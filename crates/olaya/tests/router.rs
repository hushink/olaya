use std::collections::BTreeMap;

use olaya::{RouterConfig, State, route_state};
use serde_json::json;

fn laya_router() -> RouterConfig {
    RouterConfig {
        strategy: "script".to_owned(),
        routes: BTreeMap::from([
            ("english".to_owned(), "laya".to_owned()),
            ("multilingual".to_owned(), "laya-multilingual".to_owned()),
        ]),
        default: "laya".to_owned(),
    }
}

#[test]
fn english_routes_to_the_english_checkpoint() {
    let state = State::text("I was charged twice. Please refund the extra charge.");
    assert_eq!(route_state(&laya_router(), &state), "laya");
}

#[test]
fn chinese_routes_to_the_multilingual_checkpoint() {
    let state = State::text("我这个月被重复扣款了，请退还多扣的费用。");
    assert_eq!(route_state(&laya_router(), &state), "laya-multilingual");
}

#[test]
fn cyrillic_routes_to_the_multilingual_checkpoint() {
    let state = State::text("Мне дважды списали оплату за подписку.");
    assert_eq!(route_state(&laya_router(), &state), "laya-multilingual");
}

#[test]
fn json_state_is_classified_by_its_text() {
    let english = State::from_value(json!({"message": "Please refund the duplicate charge."}))
        .expect("state");
    assert_eq!(route_state(&laya_router(), &english), "laya");

    let chinese = State::from_value(json!({"message": "请退还多扣的费用。"})).expect("state");
    assert_eq!(route_state(&laya_router(), &chinese), "laya-multilingual");
}

#[test]
fn empty_or_punctuation_only_state_uses_the_default() {
    let state = State::text("123 !!!");
    assert_eq!(route_state(&laya_router(), &state), "laya");
}

#[test]
fn unknown_strategy_uses_the_default() {
    let mut router = laya_router();
    router.strategy = "custom".to_owned();
    let state = State::text("我这个月被重复扣款了。");
    assert_eq!(route_state(&router, &state), "laya");
}
