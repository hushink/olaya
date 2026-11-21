use std::path::Path;

use olaya::parse_request_value;
use olaya_conformance::{GoldenErrorCode, load_goldens};
use serde_json::Value;

/// Every request an upstream reference answered must also pass the shared
/// parser. A golden `invalid_request` may be rejected by the parser or later by
/// the family; a golden `state_truncated` must parse (truncation is a runtime
/// condition, not a syntax error).
#[test]
fn fixture_requests_match_the_parser() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let mut checked = 0;
    let mut mismatches = Vec::new();
    for family in std::fs::read_dir(&root).expect("fixtures") {
        let family = family.expect("family").path();
        if !family.is_dir() {
            continue;
        }
        for file in std::fs::read_dir(&family).expect("family files") {
            let file = file.expect("golden file").path();
            if file.extension().and_then(|value| value.to_str()) != Some("jsonl") {
                continue;
            }
            for golden in load_goldens(&file).expect("goldens load") {
                let body = Value::Object(
                    [
                        ("model".to_owned(), Value::String(golden.model.clone())),
                        ("state".to_owned(), golden.state.clone()),
                        (
                            "questions".to_owned(),
                            Value::Object(golden.questions.clone()),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                );
                let parsed = parse_request_value(body);
                match golden.error.map(|error| error.code) {
                    None
                    | Some(GoldenErrorCode::StateTruncated)
                    | Some(GoldenErrorCode::InputTooLong) => {
                        if let Err(report) = parsed {
                            mismatches.push(format!(
                                "{}: the parser rejects an answered request: {report}",
                                golden.id
                            ));
                        }
                    }
                    Some(GoldenErrorCode::InvalidRequest) => {}
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no fixtures were found");
    assert!(
        mismatches.is_empty(),
        "{} fixture(s) disagree with the parser:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    println!("{checked} fixture requests match the parser");
}
