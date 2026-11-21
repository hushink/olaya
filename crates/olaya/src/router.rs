//! Script/language routing for router manifests.

use olaya_core::{Content, RouterConfig, State, dumps_content};

/// Pick the target model for a router manifest.
pub fn route_state(router: &RouterConfig, state: &State) -> String {
    if router.strategy != "script" {
        return router.default.clone();
    }
    let class = if is_probably_english(state) {
        "english"
    } else {
        "multilingual"
    };
    router
        .routes
        .get(class)
        .cloned()
        .unwrap_or_else(|| router.default.clone())
}

/// Whether the state is predominantly Latin script, i.e. the English
/// checkpoint is the better fit.
pub fn is_probably_english(state: &State) -> bool {
    let text = match state.as_content() {
        Content::String(text) => text.clone(),
        content => dumps_content(content),
    };
    let mut latin = 0usize;
    let mut other = 0usize;
    for ch in text.chars().filter(|ch| ch.is_alphabetic()) {
        if is_latin(ch) {
            latin += 1;
        } else {
            other += 1;
        }
    }
    let total = latin + other;
    total == 0 || other * 10 <= total
}

fn is_latin(ch: char) -> bool {
    ch.is_ascii_alphabetic() || matches!(ch as u32, 0x00C0..=0x024F | 0x1E00..=0x1EFF)
}
