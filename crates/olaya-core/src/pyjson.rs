//! Python-compatible `json.dumps` formatting.
//!
//! Ollaya's sequence layouts reproduce Python's default `json.dumps` exactly:
//! `", "` between items and `": "` after keys, with `ensure_ascii=False`.
//! `serde_json::to_string` is compact and would tokenize differently.

use std::fmt::Write;

use serde_json::Value;

use crate::content::Content;

/// `json.dumps(value, ensure_ascii=False)` for a JSON value.
pub fn dumps_value(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(true) => "true".to_owned(),
        Value::Bool(false) => "false".to_owned(),
        Value::Number(number) => number_repr(number),
        Value::String(text) => escape(text),
        Value::Array(items) => {
            let body = items.iter().map(dumps_value).collect::<Vec<_>>().join(", ");
            format!("[{body}]")
        }
        Value::Object(map) => {
            let body = map
                .iter()
                .map(|(key, value)| format!("{}: {}", escape(key), dumps_value(value)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{body}}}")
        }
    }
}

/// `json.dumps(content, ensure_ascii=False)` for the official `Content` union.
pub fn dumps_content(content: &Content) -> String {
    match content {
        Content::Null => "null".to_owned(),
        Content::String(text) => escape(text),
        Content::Object(map) => dumps_value(&Value::Object(map.clone())),
        Content::Array(items) => dumps_value(&Value::Array(items.clone())),
    }
}

/// Python's canonical JSON: `json.dumps(value, ensure_ascii=False,
/// sort_keys=True, separators=(",", ":"))`.
pub fn dumps_value_canonical(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(true) => "true".to_owned(),
        Value::Bool(false) => "false".to_owned(),
        Value::Number(number) => number_repr(number),
        Value::String(text) => escape(text),
        Value::Array(items) => {
            let body = items
                .iter()
                .map(dumps_value_canonical)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        Value::Object(map) => {
            let mut keys = map.keys().map(String::as_str).collect::<Vec<_>>();
            keys.sort_unstable();
            let body = keys
                .into_iter()
                .map(|key| format!("{}:{}", escape(key), dumps_value_canonical(&map[key])))
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
    }
}

/// `dumps_value_canonical` for the official `Content` union.
pub fn dumps_content_canonical(content: &Content) -> String {
    match content {
        Content::Null => "null".to_owned(),
        Content::String(text) => escape(text),
        Content::Object(map) => dumps_value_canonical(&Value::Object(map.clone())),
        Content::Array(items) => dumps_value_canonical(&Value::Array(items.clone())),
    }
}

fn escape(text: &str) -> String {
    serde_json::to_string(text).expect("a string is always serializable")
}

fn number_repr(number: &serde_json::Number) -> String {
    if let Some(value) = number.as_i64() {
        value.to_string()
    } else if let Some(value) = number.as_u64() {
        value.to_string()
    } else {
        float_repr(number.as_f64().unwrap_or(f64::NAN))
    }
}

/// Python's `repr(float)`, which is also what `json.dumps` writes for floats.
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 {
            "Infinity".to_owned()
        } else {
            "-Infinity".to_owned()
        };
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".to_owned()
        } else {
            "0.0".to_owned()
        };
    }
    // `{:e}` gives the shortest round-trip digits, the same digits Python's repr uses.
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("LowerExp always has an exponent");
    let exponent: i32 = exponent.parse().expect("LowerExp exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let decimal_point = exponent + 1; // value = 0.<digits> * 10^decimal_point

    let mut out = String::new();
    if value < 0.0 {
        out.push('-');
    }
    if decimal_point <= -4 || decimal_point > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    } else if decimal_point <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-decimal_point) as usize));
        out.push_str(&digits);
    } else if decimal_point as usize >= digits.len() {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n(
            '0',
            decimal_point as usize - digits.len(),
        ));
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decimal_point as usize]);
        out.push('.');
        out.push_str(&digits[decimal_point as usize..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_numbers_like_python_json() {
        assert_eq!(
            dumps_value(&json!({"n": 1, "f": 1.0, "e": 1e-5, "big": 1e16})),
            "{\"n\": 1, \"f\": 1.0, \"e\": 1e-05, \"big\": 1e+16}"
        );
    }

    #[test]
    fn renders_canonical_json_like_python() {
        assert_eq!(
            dumps_value_canonical(&json!({"b": 1, "a": "x", "c": [true, null]})),
            "{\"a\":\"x\",\"b\":1,\"c\":[true,null]}"
        );
        assert_eq!(
            dumps_value_canonical(&json!({"description": null, "key": "a"})),
            "{\"description\":null,\"key\":\"a\"}"
        );
    }
}
