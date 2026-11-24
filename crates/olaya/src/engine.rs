use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use candle_core::{DType, Device};
use indexmap::IndexMap;
use olaya_candle::tokenizer::HfTokenizer;
use olaya_candle::{Family, FamilyFactory, LoadContext, ModelError};
use olaya_core::{
    Answers, Capabilities, DecisionError, DecisionRuntime, Error, FileRole, ManifestError,
    ManifestRegistry, ManifestStore, MetricsSnapshot, ModelMetadata, QType, Questions,
    RegistryError, RepoManifest, RouterConfig, RuntimeError, RuntimeMetrics, State,
    SystemOneResponse, TokenEncoder,
};

use crate::catalog::Catalog;
use crate::families::BuiltinFamilies;
use crate::hub::{Hub, HubConfig};
use crate::registry::FsRegistry;

/// Builds an [`Engine`].
pub struct EngineBuilder {
    families: HashMap<&'static str, Arc<dyn FamilyFactory>>,
    manifests: Vec<RepoManifest>,
    device: Device,
    dtype: DType,
    tf32: bool,
    hub: Hub,
    store: Option<Arc<dyn ManifestStore>>,
    default_model: Option<String>,
    max_loaded: usize,
    metrics: Option<RuntimeMetrics>,
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self {
            families: HashMap::new(),
            manifests: Vec::new(),
            device: Device::Cpu,
            dtype: DType::F32,
            tf32: false,
            hub: Hub::new(HubConfig::from_env()),
            store: None,
            default_model: None,
            max_loaded: 3,
            metrics: None,
        }
    }
}

impl EngineBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn family(mut self, factory: Arc<dyn FamilyFactory>) -> Self {
        self.families.insert(factory.family(), factory);
        self
    }

    pub fn builtin_families(self, set: BuiltinFamilies) -> Self {
        set.register(self)
    }

    pub fn manifest(mut self, manifest: RepoManifest) -> Self {
        self.manifests.push(manifest);
        self
    }

    pub fn manifests(mut self, manifests: impl IntoIterator<Item = RepoManifest>) -> Self {
        self.manifests.extend(manifests);
        self
    }

    /// Load every `olaya.manifest.json` under `dir`.
    pub fn registry_dir(mut self, dir: impl AsRef<std::path::Path>) -> Result<Self, RegistryError> {
        self.manifests
            .extend(FsRegistry::new(dir.as_ref()).manifests()?);
        Ok(self)
    }

    /// Load manifests from a custom registry implementation.
    pub fn registry(mut self, registry: Arc<dyn ManifestRegistry>) -> Result<Self, RegistryError> {
        self.manifests.extend(registry.manifests()?);
        Ok(self)
    }

    /// Use a custom content-addressed manifest/blob store.
    pub fn store(mut self, store: Arc<dyn ManifestStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Allow TF32 tensor-core GEMMs on CUDA (ignored on CPU/Metal).
    pub fn tf32(mut self, enabled: bool) -> Self {
        self.tf32 = enabled;
        self
    }

    pub fn hub(mut self, config: HubConfig) -> Self {
        self.hub = Hub::new(config);
        self
    }

    pub fn models_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.hub.config_mut().models_dir = path.into();
        self
    }

    pub fn cache_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.hub.config_mut().cache_dir = path.into();
        self
    }

    pub fn tmp_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.hub.config_mut().tmp_dir = path.into();
        self
    }

    pub fn store_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.hub.config_mut().store_dir = path.into();
        self
    }

    pub fn default_model(mut self, name: impl Into<String>) -> Self {
        self.default_model = Some(name.into());
        self
    }

    pub fn max_loaded(mut self, max: usize) -> Self {
        self.max_loaded = max.max(1);
        self
    }

    /// Share a metrics handle with another component (for example a server).
    /// The engine creates one when this is not set.
    pub fn metrics(mut self, metrics: RuntimeMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn build(self) -> Result<Engine, ManifestError> {
        let mut catalog = Catalog::new();
        for manifest in self.manifests {
            catalog.insert(manifest)?;
        }
        let hub = match self.store {
            Some(store) => Hub::with_store(self.hub.config().clone(), store),
            None => self.hub,
        };
        #[cfg(feature = "cuda")]
        if self.tf32 && self.device.is_cuda() {
            candle_core::cuda_backend::set_gemm_reduced_precision_f32(true);
        }
        #[cfg(not(feature = "cuda"))]
        let _ = self.tf32;
        let metrics = self.metrics.unwrap_or_default();
        tracing::info!(
            models = catalog.len(),
            device = ?self.device,
            dtype = ?self.dtype,
            tf32 = self.tf32,
            "engine configured"
        );
        Ok(Engine {
            inner: Arc::new(EngineInner {
                families: self.families,
                catalog: RwLock::new(catalog),
                loaded: RwLock::new(IndexMap::new()),
                device: self.device,
                dtype: self.dtype,
                hub,
                default_model: self.default_model,
                max_loaded: self.max_loaded,
                metrics,
            }),
        })
    }
}

/// An in-process candle decision engine.
pub struct Engine {
    inner: Arc<EngineInner>,
}

pub(crate) struct EngineInner {
    families: HashMap<&'static str, Arc<dyn FamilyFactory>>,
    catalog: RwLock<Catalog>,
    loaded: RwLock<IndexMap<String, Arc<dyn Family>>>,
    device: Device,
    dtype: DType,
    hub: Hub,
    default_model: Option<String>,
    max_loaded: usize,
    metrics: RuntimeMetrics,
}

impl Engine {
    pub fn builder() -> EngineBuilder {
        EngineBuilder::new()
    }

    pub fn hub(&self) -> &Hub {
        &self.inner.hub
    }

    pub fn default_model(&self) -> Option<&str> {
        self.inner.default_model.as_deref()
    }

    /// A point-in-time copy of the runtime counters and latency histograms.
    pub fn metrics(&self) -> MetricsSnapshot {
        self.inner.metrics.snapshot()
    }

    /// The shared metrics handle, for wiring another component into the same
    /// counters before building the engine.
    pub fn metrics_handle(&self) -> RuntimeMetrics {
        self.inner.metrics.clone()
    }

    /// Run one request against `model` (or the default model when empty).
    pub fn ask(
        &self,
        model: &str,
        state: &State,
        questions: &Questions,
    ) -> Result<SystemOneResponse, Error> {
        ask_inner(&self.inner, model, state, questions)
    }

    /// Run one request and return the typed answer set.
    pub fn ask_typed(
        &self,
        model: &str,
        state: &State,
        questions: &Questions,
    ) -> Result<Answers, Error> {
        let response = self.ask(model, state, questions)?;
        Ok(Answers::new(response.answers))
    }

    /// Run independent requests, sharing a forward pass when the family
    /// supports it. Results preserve the input order.
    pub fn ask_batch(
        &self,
        model: &str,
        requests: &[(&State, &Questions)],
    ) -> Vec<Result<Answers, Error>> {
        self.ask_batch_responses(model, requests)
            .into_iter()
            .map(|result| result.map(|response| Answers::new(response.answers)))
            .collect()
    }

    /// Like [`Engine::ask_batch`] but keeps each full response (usage included).
    pub fn ask_batch_responses(
        &self,
        model: &str,
        requests: &[(&State, &Questions)],
    ) -> Vec<Result<SystemOneResponse, Error>> {
        ask_batch_inner(&self.inner, model, requests)
    }

    pub fn model(&self, name: &str) -> Result<ModelHandle, Error> {
        let (canonical, _) = self
            .inner
            .resolve(name)
            .ok_or_else(|| Error::ModelNotFound {
                model: name.to_owned(),
            })?;
        Ok(ModelHandle {
            inner: self.inner.clone(),
            name: canonical,
        })
    }

    pub fn list_models(&self) -> Vec<ModelMetadata> {
        self.inner
            .catalog
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .list()
    }

    pub fn manifests(&self) -> Vec<RepoManifest> {
        self.inner
            .catalog
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .manifests()
    }

    pub fn resolve(&self, name: &str) -> Option<RepoManifest> {
        self.inner.resolve(name).map(|(_, manifest)| manifest)
    }

    pub fn load(&self, name: &str) -> Result<(), Error> {
        load_inner(&self.inner, name, 0)
    }

    pub fn unload(&self, name: &str) -> bool {
        unload_inner(&self.inner, name, 0)
    }

    pub fn loaded_models(&self) -> Vec<String> {
        self.inner
            .loaded
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .cloned()
            .collect()
    }

    /// Pull a model reference and add it to the catalog.
    pub fn pull(&self, reference: &str) -> Result<RepoManifest, RuntimeError> {
        <Self as DecisionRuntime>::pull(self, reference)
    }

    /// Remove a model (or alias) from the catalog.
    pub fn delete(&self, name: &str) -> Result<bool, RuntimeError> {
        <Self as DecisionRuntime>::delete(self, name)
    }

    /// Add a manifest to the runtime catalog.
    pub fn add_manifest(&self, manifest: RepoManifest) -> Result<(), ManifestError> {
        self.inner
            .catalog
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(manifest)
    }
}

impl EngineInner {
    fn resolve(&self, name: &str) -> Option<(String, RepoManifest)> {
        self.catalog
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .resolve(name)
    }

    fn load_family(
        &self,
        canonical: &str,
        manifest: &RepoManifest,
    ) -> Result<Arc<dyn Family>, RuntimeError> {
        if let Some(family) = self
            .loaded
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(canonical)
        {
            tracing::trace!(model = %canonical, "family cache hit");
            return Ok(family.clone());
        }

        let started = Instant::now();
        let span = tracing::info_span!(
            "load_model",
            model = %canonical,
            family = %manifest.family,
            arch = %manifest.arch,
            device = ?self.device,
            dtype = ?self.dtype
        );
        let _enter = span.enter();
        let result = self.build_family(canonical, manifest);
        let elapsed = started.elapsed().as_secs_f64();
        match result {
            Ok(family) => {
                self.metrics.record_model_load(elapsed);
                let loaded = self
                    .loaded
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .len();
                self.metrics.set_loaded_models(loaded);
                tracing::info!(elapsed_ms = elapsed * 1000.0, "model loaded");
                Ok(family)
            }
            Err(error) => {
                self.metrics.record_model_load_error();
                tracing::warn!(
                    error = %error,
                    elapsed_ms = elapsed * 1000.0,
                    "model load failed"
                );
                Err(error)
            }
        }
    }

    fn build_family(
        &self,
        canonical: &str,
        manifest: &RepoManifest,
    ) -> Result<Arc<dyn Family>, RuntimeError> {
        let factory = self.families.get(manifest.family.as_str()).ok_or_else(|| {
            RuntimeError::FamilyNotRegistered {
                family: manifest.family.clone(),
            }
        })?;

        let model_dir = self.hub.model_dir(canonical);
        let dir = match &manifest.subfolder {
            Some(subfolder) => model_dir.join(subfolder),
            None => model_dir.clone(),
        };
        if manifest
            .files
            .iter()
            .any(|file| !dir.join(&file.path).is_file())
        {
            let store = self.hub.store();
            match store.get(canonical) {
                Ok(stored) if crate::store::matches_manifest(&stored, manifest) => {
                    tracing::info!(model = %canonical, "materializing model from the store");
                    store.materialize(&stored, &model_dir).map_err(|error| {
                        RuntimeError::Storage {
                            message: format!("materializing {canonical} from the store: {error}"),
                        }
                    })?;
                }
                Ok(_) => {
                    tracing::warn!(
                        model = %canonical,
                        "store entry does not match the registry manifest; not materializing"
                    );
                }
                Err(error) => {
                    tracing::debug!(
                        model = %canonical,
                        %error,
                        "model files are missing and no store entry exists"
                    );
                }
            }
        }
        let tokenizer_file =
            manifest
                .file(FileRole::Tokenizer)
                .ok_or_else(|| RuntimeError::LoadFailed {
                    message: format!("model {canonical:?} has no tokenizer file"),
                })?;
        let tokenizer: Arc<dyn TokenEncoder> = Arc::new(
            HfTokenizer::from_file(dir.join(&tokenizer_file.path)).map_err(map_model_error)?,
        );
        let ctx = LoadContext {
            dir: &dir,
            manifest,
            device: &self.device,
            dtype: self.dtype,
            tokenizer,
        };
        let family = factory.load(&ctx).map_err(map_model_error)?;
        let mut loaded = self
            .loaded
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loaded.insert(canonical.to_owned(), family.clone());
        while loaded.len() > self.max_loaded {
            let victim = loaded
                .keys()
                .find(|name| name.as_str() != canonical)
                .cloned();
            match victim {
                Some(name) => {
                    tracing::debug!(model = %name, "evicting model to satisfy max_loaded");
                    loaded.shift_remove(&name);
                }
                None => break,
            }
        }
        drop(loaded);
        Ok(family)
    }
}

/// A resolved model handle for repeated calls.
pub struct ModelHandle {
    inner: Arc<EngineInner>,
    name: String,
}

impl ModelHandle {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn ask(&self, state: &State, questions: &Questions) -> Result<SystemOneResponse, Error> {
        ask_inner(&self.inner, &self.name, state, questions)
    }

    pub fn ask_typed(&self, state: &State, questions: &Questions) -> Result<Answers, Error> {
        let response = self.ask(state, questions)?;
        Ok(Answers::new(response.answers))
    }

    pub fn info(&self) -> Option<RepoManifest> {
        self.inner.resolve(&self.name).map(|(_, manifest)| manifest)
    }

    pub fn unload(&self) -> bool {
        unload_inner(&self.inner, &self.name, 0)
    }
}

fn load_inner(inner: &EngineInner, name: &str, depth: usize) -> Result<(), Error> {
    if depth > 4 {
        return Err(Error::Model {
            model: name.to_owned(),
            source: RuntimeError::InvalidRequest {
                message: "router chain is too deep".to_owned(),
            },
        });
    }
    let (canonical, manifest) = inner.resolve(name).ok_or_else(|| Error::ModelNotFound {
        model: name.to_owned(),
    })?;
    if let Some(router) = &manifest.router {
        return load_inner(inner, &router.default, depth + 1);
    }
    inner
        .load_family(&canonical, &manifest)
        .map(|_| ())
        .map_err(|source| Error::Model {
            model: canonical,
            source,
        })
}

fn unload_inner(inner: &EngineInner, name: &str, depth: usize) -> bool {
    if depth > 4 {
        return false;
    }
    let Some((canonical, manifest)) = inner.resolve(name) else {
        return false;
    };
    if let Some(router) = &manifest.router {
        let mut unloaded = false;
        for target in router
            .routes
            .values()
            .chain(std::iter::once(&router.default))
        {
            unloaded |= unload_inner(inner, target, depth + 1);
        }
        return unloaded;
    }
    let mut loaded = inner
        .loaded
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let removed = loaded.shift_remove(&canonical).is_some();
    let count = loaded.len();
    drop(loaded);
    if removed {
        inner.metrics.record_model_unload(true);
        inner.metrics.set_loaded_models(count);
        tracing::info!(model = %canonical, "model unloaded");
    }
    removed
}

fn pull_inner(
    inner: &EngineInner,
    reference: &str,
    depth: usize,
    visited: &mut HashSet<String>,
) -> Result<RepoManifest, RuntimeError> {
    if depth > 4 {
        return Err(RuntimeError::InvalidRequest {
            message: "router chain is too deep".to_owned(),
        });
    }

    let resolved = inner.resolve(reference);
    if let Some((canonical, manifest)) = &resolved {
        if !visited.insert(canonical.clone()) {
            return Ok(manifest.clone());
        }
        if let Some(router) = &manifest.router {
            let targets = router_targets(router);
            tracing::info!(model = %manifest.name, targets = ?targets, "pulling router targets");
            for target in &targets {
                pull_inner(inner, target, depth + 1, visited)?;
            }
            tracing::info!(model = %manifest.name, "router targets pulled");
            return Ok(manifest.clone());
        }
    }

    let manifest = match resolved.as_ref() {
        Some((_, hint)) => inner.hub.pull(reference, Some(hint))?,
        None => {
            // Fetch the self-describing manifest first so a repository cannot
            // silently replace an existing catalog model with the same name.
            let manifest = inner.hub.fetch_manifest(reference)?;
            manifest.validate().map_err(|error| RuntimeError::Storage {
                message: format!("invalid manifest from {reference}: {error}"),
            })?;
            ensure_pull_does_not_clobber(inner, &manifest, reference)?;
            inner
                .hub
                .validate_manifest_network(&manifest, reference, true)?;
            inner.hub.pull(reference, Some(&manifest))?
        }
    };
    inner
        .catalog
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .upsert(manifest.clone())
        .map_err(|error| RuntimeError::Storage {
            message: error.to_string(),
        })?;

    if let Some(router) = &manifest.router {
        if !visited.insert(manifest.name.clone()) {
            return Ok(manifest);
        }
        let targets = router_targets(router);
        tracing::info!(model = %manifest.name, targets = ?targets, "pulling router targets");
        for target in &targets {
            pull_inner(inner, target, depth + 1, visited)?;
        }
        tracing::info!(model = %manifest.name, "router targets pulled");
    }

    Ok(manifest)
}

/// Refuse to let an uncatalogued repository replace an existing model with the
/// same canonical name or alias. Without this check, `models pull <other-url>`
/// could silently overwrite a trusted catalog entry.
fn ensure_pull_does_not_clobber(
    inner: &EngineInner,
    manifest: &RepoManifest,
    reference: &str,
) -> Result<(), RuntimeError> {
    for candidate in
        std::iter::once(manifest.name.as_str()).chain(manifest.aliases.iter().map(String::as_str))
    {
        if let Some((canonical, existing)) = inner.resolve(candidate)
            && existing != *manifest
        {
            return Err(RuntimeError::InvalidRequest {
                message: format!(
                    "refusing to replace existing model {canonical:?} with {reference}; \
                     remove it from the catalog first"
                ),
            });
        }
    }
    Ok(())
}

fn router_targets(router: &RouterConfig) -> Vec<String> {
    let mut targets = Vec::new();
    for target in router
        .routes
        .values()
        .chain(std::iter::once(&router.default))
    {
        if !targets.contains(target) {
            targets.push(target.clone());
        }
    }
    targets
}

fn ask_inner(
    inner: &EngineInner,
    model: &str,
    state: &State,
    questions: &Questions,
) -> Result<SystemOneResponse, Error> {
    let span = tracing::debug_span!("ask", model = %model, questions = questions.len());
    let _enter = span.enter();
    let started = Instant::now();
    let result = ask_inner_depth(inner, model, state, questions, 0);
    let elapsed = started.elapsed().as_secs_f64();
    match &result {
        Ok(response) => {
            inner.metrics.record_request(
                response.usage.input_tokens,
                response.usage.output_tokens,
                elapsed,
            );
            tracing::debug!(
                model = %response.model,
                input_tokens = response.usage.input_tokens,
                output_tokens = response.usage.output_tokens,
                elapsed_ms = elapsed * 1000.0,
                "decision completed"
            );
        }
        Err(error) => {
            inner.metrics.record_request_error();
            tracing::debug!(
                error = %error,
                elapsed_ms = elapsed * 1000.0,
                "decision failed"
            );
        }
    }
    result
}

fn capability_error(
    canonical: &str,
    questions: &Questions,
    capabilities: &Capabilities,
) -> Option<Error> {
    for (question_id, question) in questions.iter() {
        let qtype = QType::of(question);
        if !capabilities.supports(qtype) {
            return Some(Error::Model {
                model: canonical.to_owned(),
                source: RuntimeError::InvalidRequest {
                    message: format!(
                        "model {canonical:?} does not support {} questions",
                        qtype.name()
                    ),
                },
            });
        }
        if question.is_ordinal() && !capabilities.ordinal {
            return Some(Error::Model {
                model: canonical.to_owned(),
                source: RuntimeError::InvalidRequest {
                    message: format!("model {canonical:?} does not support ordinal choices"),
                },
            });
        }
        if let Some(limit) = capabilities.max_options {
            let options = question.num_options();
            if options > limit {
                return Some(Error::Model {
                    model: canonical.to_owned(),
                    source: RuntimeError::TooManyOptions {
                        question: question_id.clone(),
                        options,
                        limit,
                    },
                });
            }
        }
    }
    None
}

fn ask_inner_depth(
    inner: &EngineInner,
    model: &str,
    state: &State,
    questions: &Questions,
    depth: usize,
) -> Result<SystemOneResponse, Error> {
    questions.validate().map_err(Error::Validation)?;

    let requested = if model.trim().is_empty() {
        inner
            .default_model
            .as_deref()
            .ok_or_else(|| Error::ModelNotFound {
                model: model.to_owned(),
            })?
    } else {
        model
    };
    let (canonical, manifest) = inner
        .resolve(requested)
        .ok_or_else(|| Error::ModelNotFound {
            model: requested.to_owned(),
        })?;
    if let Some(router) = &manifest.router {
        if depth >= 4 {
            return Err(Error::Model {
                model: canonical,
                source: RuntimeError::InvalidRequest {
                    message: "router chain is too deep".to_owned(),
                },
            });
        }
        let target = crate::router::route_state(router, state);
        let (target_canonical, _) = inner.resolve(&target).ok_or_else(|| Error::ModelNotFound {
            model: target.clone(),
        })?;
        if target_canonical == canonical {
            return Err(Error::Model {
                model: canonical,
                source: RuntimeError::InvalidRequest {
                    message: "router routes to itself".to_owned(),
                },
            });
        }
        let mut response = ask_inner_depth(inner, &target_canonical, state, questions, depth + 1)?;
        response.model = target_canonical;
        return Ok(response);
    }
    let family = inner
        .load_family(&canonical, &manifest)
        .map_err(|source| Error::Model {
            model: canonical.clone(),
            source,
        })?;
    if let Some(error) = capability_error(&canonical, questions, &family.capabilities()) {
        return Err(error);
    }
    let mut response = family.predict(state, questions).map_err(|error| {
        let source = match error {
            ModelError::Decision(DecisionError::StateTruncated) => RuntimeError::StateTruncated {
                model: canonical.clone(),
            },
            other => map_model_error(other),
        };
        Error::Model {
            model: canonical.clone(),
            source,
        }
    })?;
    response.model = canonical;
    Ok(response)
}

fn ask_batch_inner(
    inner: &EngineInner,
    model: &str,
    requests: &[(&State, &Questions)],
) -> Vec<Result<SystemOneResponse, Error>> {
    if requests.is_empty() {
        return Vec::new();
    }
    let span = tracing::debug_span!("ask_batch", model = %model, requests = requests.len());
    let _enter = span.enter();
    let requested = if model.trim().is_empty() {
        match inner.default_model.as_deref() {
            Some(default) => default.to_owned(),
            None => {
                for _ in requests {
                    inner.metrics.record_request_error();
                }
                return requests
                    .iter()
                    .map(|_| {
                        Err(Error::ModelNotFound {
                            model: model.to_owned(),
                        })
                    })
                    .collect();
            }
        }
    } else {
        model.to_owned()
    };
    let Some((canonical, manifest)) = inner.resolve(&requested) else {
        for _ in requests {
            inner.metrics.record_request_error();
        }
        return requests
            .iter()
            .map(|_| {
                Err(Error::ModelNotFound {
                    model: requested.clone(),
                })
            })
            .collect();
    };
    if manifest.router.is_some() {
        tracing::debug!(model = %requested, "router batch falls back to per-request asks");
        return requests
            .iter()
            .map(|(state, questions)| ask_inner(inner, &requested, state, questions))
            .collect();
    }
    let family = match inner.load_family(&canonical, &manifest) {
        Ok(family) => family,
        Err(source) => {
            for _ in requests {
                inner.metrics.record_request_error();
            }
            return requests
                .iter()
                .map(|_| {
                    Err(Error::Model {
                        model: canonical.clone(),
                        source: source.clone(),
                    })
                })
                .collect();
        }
    };
    let capabilities = family.capabilities();
    let mut results: Vec<Option<Result<SystemOneResponse, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    let mut indices = Vec::new();
    let mut valid = Vec::new();
    for (index, (state, questions)) in requests.iter().enumerate() {
        if let Err(report) = questions.validate() {
            results[index] = Some(Err(Error::Validation(report)));
            continue;
        }
        if let Some(error) = capability_error(&canonical, questions, &capabilities) {
            results[index] = Some(Err(error));
            continue;
        }
        indices.push(index);
        valid.push((*state, *questions));
    }
    let mut batch_seconds = 0.0;
    if !valid.is_empty() {
        let started = Instant::now();
        let outputs = family.predict_batch(&valid);
        batch_seconds = started.elapsed().as_secs_f64();
        inner.metrics.record_batch(valid.len());
        if outputs.len() != valid.len() {
            for index in indices {
                results[index] = Some(Err(Error::Model {
                    model: canonical.clone(),
                    source: RuntimeError::Inference {
                        message: format!(
                            "batched predict returned {} results for {} requests",
                            outputs.len(),
                            valid.len()
                        ),
                    },
                }));
            }
        } else {
            for (index, output) in indices.into_iter().zip(outputs) {
                results[index] = Some(match output {
                    Ok(mut response) => {
                        response.model = canonical.clone();
                        Ok(response)
                    }
                    Err(error) => {
                        let source = match error {
                            ModelError::Decision(DecisionError::StateTruncated) => {
                                RuntimeError::StateTruncated {
                                    model: canonical.clone(),
                                }
                            }
                            other => map_model_error(other),
                        };
                        Err(Error::Model {
                            model: canonical.clone(),
                            source,
                        })
                    }
                });
            }
        }
    }
    let results = results
        .into_iter()
        .map(|result| {
            result.unwrap_or_else(|| {
                Err(Error::Model {
                    model: canonical.clone(),
                    source: RuntimeError::Inference {
                        message: "batched predict did not return this request".to_owned(),
                    },
                })
            })
        })
        .collect::<Vec<_>>();
    let ok = results.iter().filter(|result| result.is_ok()).count();
    for result in &results {
        match result {
            Ok(response) => inner.metrics.record_request(
                response.usage.input_tokens,
                response.usage.output_tokens,
                batch_seconds,
            ),
            Err(_) => inner.metrics.record_request_error(),
        }
    }
    tracing::debug!(
        model = %canonical,
        batch_size = valid.len(),
        ok,
        errors = results.len() - ok,
        elapsed_ms = batch_seconds * 1000.0,
        "decision batch completed"
    );
    results
}

fn map_model_error(error: ModelError) -> RuntimeError {
    match error {
        ModelError::Io(error) => RuntimeError::LoadFailed {
            message: error.to_string(),
        },
        ModelError::Candle(error) => RuntimeError::Inference {
            message: error.to_string(),
        },
        ModelError::Decision(error) => match error {
            DecisionError::Unsupported { message } => RuntimeError::Unsupported { message },
            DecisionError::TooManyOptions {
                question,
                options,
                limit,
            } => RuntimeError::TooManyOptions {
                question,
                options,
                limit,
            },
            DecisionError::InputTooLong { tokens, .. } => RuntimeError::InputTooLong { tokens },
            other => RuntimeError::InvalidRequest {
                message: other.to_string(),
            },
        },
        ModelError::Config(message) => RuntimeError::LoadFailed { message },
        ModelError::Tokenizer(message) => RuntimeError::LoadFailed { message },
        ModelError::Unsupported(message) => RuntimeError::Unsupported { message },
    }
}

fn runtime_error(error: Error) -> RuntimeError {
    match error {
        Error::ModelNotFound { model } => RuntimeError::ModelNotFound { model },
        Error::Model { source, .. } => source,
        Error::Validation(report) => RuntimeError::InvalidRequest {
            message: report.to_string(),
        },
        Error::Decision(error) => RuntimeError::InvalidRequest {
            message: error.to_string(),
        },
        Error::Manifest(error) => RuntimeError::LoadFailed {
            message: error.to_string(),
        },
    }
}

impl DecisionRuntime for Engine {
    fn predict(
        &self,
        model: &str,
        state: &State,
        questions: &Questions,
    ) -> Result<SystemOneResponse, RuntimeError> {
        self.ask(model, state, questions).map_err(runtime_error)
    }

    fn predict_batch(
        &self,
        model: &str,
        requests: &[(&State, &Questions)],
    ) -> Vec<Result<SystemOneResponse, RuntimeError>> {
        ask_batch_inner(&self.inner, model, requests)
            .into_iter()
            .map(|result| result.map_err(runtime_error))
            .collect()
    }

    fn list_models(&self) -> Vec<ModelMetadata> {
        Engine::list_models(self)
    }

    fn resolve(&self, name: &str) -> Option<RepoManifest> {
        self.inner.resolve(name).map(|(_, manifest)| manifest)
    }

    fn load(&self, name: &str) -> Result<(), RuntimeError> {
        Engine::load(self, name).map_err(|error| match error {
            Error::ModelNotFound { model } => RuntimeError::ModelNotFound { model },
            Error::Model { source, .. } => source,
            other => RuntimeError::LoadFailed {
                message: other.to_string(),
            },
        })
    }

    fn unload(&self, name: &str) -> bool {
        Engine::unload(self, name)
    }

    fn loaded_models(&self) -> Vec<String> {
        Engine::loaded_models(self)
    }

    fn pull(&self, reference: &str) -> Result<RepoManifest, RuntimeError> {
        let started = Instant::now();
        let mut visited = HashSet::new();
        let result = pull_inner(&self.inner, reference, 0, &mut visited);
        let elapsed = started.elapsed().as_secs_f64();
        match &result {
            Ok(manifest) => {
                self.inner.metrics.record_pull(true);
                tracing::info!(
                    model = %manifest.name,
                    revision = manifest.revision.as_deref().unwrap_or("-"),
                    elapsed_ms = elapsed * 1000.0,
                    "model pulled"
                );
            }
            Err(error) => {
                self.inner.metrics.record_pull(false);
                tracing::warn!(
                    reference = %reference,
                    error = %error,
                    elapsed_ms = elapsed * 1000.0,
                    "model pull failed"
                );
            }
        }
        result
    }

    fn delete(&self, name: &str) -> Result<bool, RuntimeError> {
        let removed = self
            .inner
            .catalog
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(name);
        if let Some(canonical) = &removed {
            let mut loaded = self
                .inner
                .loaded
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            loaded.shift_remove(canonical);
            let count = loaded.len();
            drop(loaded);
            self.inner.metrics.set_loaded_models(count);
            tracing::info!(model = %canonical, "model deleted");
        }
        self.inner.metrics.record_delete(removed.is_some());
        Ok(removed.is_some())
    }

    fn runtime_metrics(&self) -> Option<RuntimeMetrics> {
        Some(self.inner.metrics.clone())
    }
}
