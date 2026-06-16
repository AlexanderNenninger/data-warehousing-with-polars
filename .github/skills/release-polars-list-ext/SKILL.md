---
name: release-polars-list-ext
description: 'Create and publish a new release of polars-list-ext. Use when releasing a new version: dev, release candidate (rc), or full release. Reads the current version from Cargo.toml automatically, then asks whether to do a minor or major bump and what release type (dev/rc/full). Updates Cargo.toml, commits, creates and pushes the git tag to trigger the publish workflow.'
argument-hint: 'Optional: version number (e.g. 0.15.0)'
---

# Release polars-list-ext

## When to Use
- Publishing a new dev, rc, or stable version of the `polars-list-ext` Rust/Python extension
- Triggering the `publish-polars-list-ext` GitHub Actions workflow

## Procedure

### 1. Read the current version

Read `packages/polars-list-ext/Cargo.toml` and extract the `version` field from `[package]`.

### 2. Ask the user for release details

Tell the user the current version and present four options derived from it. For example, if the current base version is `0.14.0`:

| Option | Description | Example tag |
|--------|-------------|-------------|
| **minor** | Bump minor, full release | `polars-list-ext-v0.15.0` |
| **major** | Bump major, full release | `polars-list-ext-v1.0.0` |
| **rc** | Bump minor, release candidate | `polars-list-ext-v0.15.0rc1` |
| **dev** | Bump minor, dev pre-release | `polars-list-ext-v0.15.0.dev1` |

For rc/dev, also ask for the pre-release number (default: 1).

Strip any existing pre-release suffix from the current version before computing the bump (e.g. `0.14.0-dev.1` → base `0.14.0`).

### 3. Update the version in `packages/polars-list-ext/Cargo.toml`

Edit the `version` field under `[package]` to match the new version (use PEP 440 / semver as appropriate).

### 4. Commit the version bump

```bash
git add packages/polars-list-ext/Cargo.toml
git commit -m "chore: bump polars-list-ext to <version>"
```

### 5. Create and push the tag

```bash
git tag <tag>
git push origin HEAD <tag>
```

This triggers the `publish-polars-list-ext.yml` workflow which builds wheels for all platforms and publishes to PyPI (if `PYPI_API_TOKEN` is set).

### 6. Confirm

Tell the user the tag that was pushed and link them to:
`https://github.com/AlexanderNenninger/data-warehousing-with-polars/actions`
