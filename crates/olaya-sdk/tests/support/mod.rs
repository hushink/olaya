//! A tiny HTTP/1.1 mock server for SDK integration tests.

#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl RecordedRequest {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }

    pub fn body_text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

pub struct MockResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub delay: Duration,
}

impl MockResponse {
    pub fn json(status: u16, body: serde_json::Value) -> Self {
        Self {
            status,
            headers: vec![("Content-Type".to_owned(), "application/json".to_owned())],
            body: serde_json::to_vec(&body).expect("serialize response"),
            delay: Duration::ZERO,
        }
    }

    pub fn text(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            headers: vec![("Content-Type".to_owned(), "text/plain".to_owned())],
            body: body.into().into_bytes(),
            delay: Duration::ZERO,
        }
    }

    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }
}

pub struct MockServer {
    address: SocketAddr,
    records: Arc<Mutex<Vec<RecordedRequest>>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl MockServer {
    pub fn start<F>(handler: F) -> Self
    where
        F: Fn(&RecordedRequest, usize) -> MockResponse + Send + Sync + 'static,
    {
        let listener = bind_listener();
        let address = listener.local_addr().expect("mock address");
        listener
            .set_nonblocking(true)
            .expect("nonblocking listener");
        let records = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let handler = Arc::new(handler);
        let thread_records = records.clone();
        let thread_stop = stop.clone();
        let handle = thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        // Handle each connection on its own thread so retries
                        // are not serialized behind a slow response.
                        let handler = handler.clone();
                        let records = thread_records.clone();
                        thread::spawn(move || {
                            // Windows inherits the listener's nonblocking mode
                            // on accepted sockets; switch back to blocking.
                            let _ = stream.set_nonblocking(false);
                            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                            match read_request(stream) {
                                Ok((record, mut stream)) => {
                                    let index = {
                                        let mut records = records.lock().expect("records");
                                        records.push(record.clone());
                                        records.len() - 1
                                    };
                                    let response = handler(&record, index);
                                    if !response.delay.is_zero() {
                                        thread::sleep(response.delay);
                                    }
                                    if let Err(error) = write_response(&mut stream, response) {
                                        eprintln!("mock write error: {error}");
                                    }
                                }
                                Err(error) => eprintln!("mock read error: {error}"),
                            }
                        });
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            address,
            records,
            stop,
            handle: Some(handle),
        }
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.records.lock().expect("records").clone()
    }

    pub fn request_count(&self) -> usize {
        self.records.lock().expect("records").len()
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn bind_listener() -> TcpListener {
    let mut last_error = None;
    for _ in 0..40 {
        match TcpListener::bind("127.0.0.1:0") {
            Ok(listener) => return listener,
            Err(error) => {
                last_error = Some(error);
                thread::sleep(Duration::from_millis(25));
            }
        }
    }
    panic!("bind mock server: {}", last_error.expect("bind attempted"));
}

fn read_request(mut stream: TcpStream) -> io::Result<(RecordedRequest, TcpStream)> {
    let mut buffer = Vec::new();
    let header_end = loop {
        if let Some(position) = find(&buffer, b"\r\n\r\n") {
            break position;
        }
        let mut chunk = [0u8; 4096];
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed in headers",
            ));
        }
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.len() > 1_048_576 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "headers too large",
            ));
        }
    };
    let header_text = String::from_utf8_lossy(&buffer[..header_end]).into_owned();
    let mut lines = header_text.lines();
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_owned();
    let path = parts.next().unwrap_or_default().to_owned();
    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
        }
    }
    let mut body = buffer[header_end + 4..].to_vec();
    if headers
        .get("transfer-encoding")
        .is_some_and(|value| value.eq_ignore_ascii_case("chunked"))
    {
        body = read_chunked(&mut stream, body)?;
    } else if let Some(length) = headers
        .get("content-length")
        .and_then(|value| value.parse::<usize>().ok())
    {
        let mut chunk = [0u8; 4096];
        while body.len() < length {
            let read = stream.read(&mut chunk)?;
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "connection closed in body",
                ));
            }
            body.extend_from_slice(&chunk[..read]);
        }
        body.truncate(length);
    } else {
        body.clear();
    }
    Ok((
        RecordedRequest {
            method,
            path,
            headers,
            body,
        },
        stream,
    ))
}

fn read_chunked(stream: &mut TcpStream, mut buffer: Vec<u8>) -> io::Result<Vec<u8>> {
    let mut body = Vec::new();
    loop {
        let line = read_line(stream, &mut buffer)?;
        let size_text = line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_text, 16).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid chunk size {size_text:?}: {error}"),
            )
        })?;
        if size == 0 {
            loop {
                let trailer = read_line(stream, &mut buffer)?;
                if trailer.is_empty() {
                    break;
                }
            }
            break;
        }
        while buffer.len() < size + 2 {
            let mut chunk = [0u8; 4096];
            let read = stream.read(&mut chunk)?;
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "connection closed in chunk",
                ));
            }
            buffer.extend_from_slice(&chunk[..read]);
        }
        body.extend_from_slice(&buffer[..size]);
        buffer.drain(..size + 2);
    }
    Ok(body)
}

fn read_line(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> io::Result<String> {
    loop {
        if let Some(position) = find(buffer, b"\r\n") {
            let line = String::from_utf8_lossy(&buffer[..position]).into_owned();
            buffer.drain(..position + 2);
            return Ok(line);
        }
        let mut chunk = [0u8; 4096];
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed in line",
            ));
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn write_response(stream: &mut TcpStream, response: MockResponse) -> io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        response.status,
        reason(response.status),
        response.body.len()
    );
    for (name, value) in &response.headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(value);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(&response.body)?;
    stream.flush()
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        529 => "Overloaded",
        _ => "Status",
    }
}
