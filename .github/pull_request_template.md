## Summary

<!-- What changed, and why? Keep it to a few sentences. -->

## Related issue

Closes #

<!-- Use `Refs #123` for related issues that should not be closed. -->

## Type of change

<!-- release-plz derives versions and changelogs from conventional commits.
Use a conventional PR title and commit messages, e.g. `feat(sdk): ...`,
`fix(serve): ...`, `docs: ...`, `chore(ci): ...`. -->

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor or maintenance
- [ ] Performance
- [ ] Breaking change
- [ ] Dependencies or release tooling

## Affected areas

- [ ] `olaya-core`
- [ ] `olaya-candle` (encoders / families)
- [ ] `olaya` (engine / serve)
- [ ] `olaya-cli`
- [ ] `olaya-sdk`
- [ ] `registry/` or catalog
- [ ] CI, release, or tooling
- [ ] Documentation only

## Test plan

CI covers fmt, clippy, workspace tests, SDK all-features, all-families, and catalog generation; run `cargo run -p xtask -- check` locally to reproduce them. List the commands or manual steps you ran, and call out anything that needs a local model, CUDA, or Metal. For CLI, HTTP, or SDK behavior changes, include before/after output.

```console
cargo run -p xtask -- check
```

<!-- Targeted examples:
cargo test -p olaya-sdk --all-features
cargo test -p olaya --features serve --test serve_contract
cargo test --release -p olaya-conformance -- --ignored
cargo test --release -p olaya --features cuda --test dtype_parity -- --ignored
-->

## Compatibility and risk

<!-- Public API, TypeSafe wire format, manifest schema, MSRV 1.92, or behavior changes. -->

- Breaking changes / migration:
- Semver impact:
- Performance impact (include `bench` / `batch_bench` numbers for inference or batching changes):

## Model or manifest changes

<!-- Delete this section when not applicable. -->

- [ ] Manifest added or updated under `registry/v1/library/<family>/manifests/`
- [ ] `revision` pinned and `sha256` / `size` provided
- [ ] `cargo run -p xtask -- sync-registry --output target/registry-catalog.json` succeeds
- [ ] Parity or smoke evidence included for the new or changed checkpoint

## Checklist

- [ ] The PR title and commits use conventional-commit types (`feat:`, `fix:`, `docs:`, `chore:`, ...).
- [ ] I added or updated tests, or explained why no test is needed.
- [ ] I updated documentation when behavior or usage changed.
- [ ] I removed API keys, tokens, private data, and model weights from the PR.
- [ ] Public API, wire-format, or manifest-schema changes describe their impact above.
