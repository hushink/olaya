//! Greedy row batching for causal backbones.
//!
//! Rows are right-padded to the longest row of their batch. The Qwen3.x trunk
//! is causal and stateless, so padding after a row's last real token cannot
//! change that row's hidden states. The budget bounds `batch * max(len)` so a
//! few long rows do not allocate an unbounded activation tensor.

use std::ops::Range;

/// Split `lengths` into contiguous groups whose padded token count stays under
/// `budget`. A single row longer than the budget becomes its own group.
pub(crate) fn groups(lengths: &[usize], budget: usize) -> Vec<Range<usize>> {
    if lengths.is_empty() {
        return Vec::new();
    }
    let budget = budget.max(1);
    let mut groups = Vec::new();
    let mut start = 0;
    let mut longest = 0;
    for (index, &length) in lengths.iter().enumerate() {
        let next_longest = longest.max(length);
        let count = index - start + 1;
        if index > start && count.saturating_mul(next_longest) > budget {
            groups.push(start..index);
            start = index;
            longest = length;
        } else {
            longest = next_longest;
        }
    }
    groups.push(start..lengths.len());
    groups
}

#[cfg(test)]
mod tests {
    use super::groups;

    #[test]
    fn groups_respect_the_padded_budget() {
        assert_eq!(groups(&[10, 10, 10, 10], 25), vec![0..2, 2..4]);
        assert_eq!(groups(&[100, 1, 1], 20), vec![0..1, 1..3]);
        assert_eq!(groups(&[], 10), Vec::<std::ops::Range<usize>>::new());
    }
}
