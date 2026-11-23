//! Cross-request batching for `POST /v1/systemone`.
//!
//! One worker collects up to `max_batch` requests, waits at most `max_wait`,
//! groups them by model, and calls `DecisionRuntime::predict_batch` once per
//! model in a blocking task. Requests without a batchable runtime still work:
//! the default trait implementation loops over `predict`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use olaya_core::{
    DecisionRuntime, Questions, RuntimeError, RuntimeMetrics, State, SystemOneResponse,
};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Instant, timeout_at};

#[derive(Clone, Copy, Debug)]
pub struct BatchConfig {
    pub max_batch: usize,
    pub max_wait: Duration,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch: 32,
            max_wait: Duration::from_millis(2),
        }
    }
}

struct Job {
    model: String,
    state: State,
    questions: Questions,
    reply: oneshot::Sender<Result<SystemOneResponse, RuntimeError>>,
}

pub struct Batcher {
    sender: mpsc::Sender<Job>,
    metrics: RuntimeMetrics,
}

impl Batcher {
    /// Spawn the worker with the default queue capacity (8×`max_batch`).
    /// Must be called from inside a Tokio runtime.
    pub fn spawn(
        runtime: Arc<dyn DecisionRuntime>,
        config: BatchConfig,
        metrics: RuntimeMetrics,
    ) -> Arc<Self> {
        Self::spawn_with_queue(runtime, config, config.max_batch.max(1) * 8, metrics)
    }

    /// Spawn the worker with an explicit queue capacity. Must be called from
    /// inside a Tokio runtime.
    pub fn spawn_with_queue(
        runtime: Arc<dyn DecisionRuntime>,
        config: BatchConfig,
        max_queue: usize,
        metrics: RuntimeMetrics,
    ) -> Arc<Self> {
        let (sender, receiver) = mpsc::channel(max_queue.max(1));
        tokio::spawn(worker(runtime, receiver, config, metrics.clone()));
        Arc::new(Self { sender, metrics })
    }

    pub async fn predict(
        &self,
        model: &str,
        state: State,
        questions: Questions,
    ) -> Result<SystemOneResponse, RuntimeError> {
        let (reply, response) = oneshot::channel();
        self.metrics.batch_queued();
        if self
            .sender
            .send(Job {
                model: model.to_owned(),
                state,
                questions,
                reply,
            })
            .await
            .is_err()
        {
            self.metrics.batch_dequeued();
            return Err(RuntimeError::inference("batch worker stopped"));
        }
        response
            .await
            .map_err(|_| RuntimeError::inference("batch worker dropped the request"))?
    }
}

async fn worker(
    runtime: Arc<dyn DecisionRuntime>,
    mut receiver: mpsc::Receiver<Job>,
    config: BatchConfig,
    metrics: RuntimeMetrics,
) {
    let max_batch = config.max_batch.max(1);
    while let Some(first) = receiver.recv().await {
        metrics.batch_dequeued();
        let mut jobs = vec![first];
        if max_batch > 1 {
            let deadline = Instant::now() + config.max_wait;
            while jobs.len() < max_batch {
                match timeout_at(deadline, receiver.recv()).await {
                    Ok(Some(job)) => {
                        metrics.batch_dequeued();
                        jobs.push(job);
                    }
                    _ => break,
                }
            }
        }
        tracing::trace!(requests = jobs.len(), "collected batch");
        dispatch(runtime.clone(), jobs).await;
    }
}

async fn dispatch(runtime: Arc<dyn DecisionRuntime>, jobs: Vec<Job>) {
    let mut groups: HashMap<String, Vec<Job>> = HashMap::new();
    for job in jobs {
        groups.entry(job.model.clone()).or_default().push(job);
    }
    for (model, jobs) in groups {
        tracing::debug!(
            model = %model,
            requests = jobs.len(),
            "dispatching batched decision requests"
        );
        let states = jobs.iter().map(|job| job.state.clone()).collect::<Vec<_>>();
        let questions = jobs
            .iter()
            .map(|job| job.questions.clone())
            .collect::<Vec<_>>();
        let (sender, receiver) = oneshot::channel();
        let runtime = runtime.clone();
        let handle = tokio::task::spawn_blocking(move || {
            let requests = states.iter().zip(questions.iter()).collect::<Vec<_>>();
            let _ = sender.send(runtime.predict_batch(&model, &requests));
        });
        match receiver.await {
            Ok(results) if results.len() == jobs.len() => {
                for (job, result) in jobs.into_iter().zip(results) {
                    let _ = job.reply.send(result);
                }
            }
            Ok(results) => {
                let message = format!(
                    "batched predict returned {} results for {} requests",
                    results.len(),
                    jobs.len()
                );
                for job in jobs {
                    let _ = job
                        .reply
                        .send(Err(RuntimeError::inference(message.clone())));
                }
            }
            Err(_) => {
                let _ = handle.await;
                for job in jobs {
                    let _ = job
                        .reply
                        .send(Err(RuntimeError::inference("batch task failed")));
                }
            }
        }
    }
}
