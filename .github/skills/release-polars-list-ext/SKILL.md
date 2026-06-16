---
name: release-polars-list-ext
description: 'Create and publish a new release of polars-list-ext. Use when releasing a new version: dev, release candidate (rc), or full release. Asks for release type and version, updates Cargo.toml, creates and pushes the git tag to trigger the publish workflow.'
argument-hint: 'Optional: version number (e.g. 0.15.0)'
---

# Release polars-list-ext

## When to Use
- Publishing a new dev, rc, or stable version of the `polars-list-ext` Rust/Python extension
- Triggering the `publish-polars-list-ext` GitHub Actions workflow

## Procedure

### 1. Ask the user for release details

Ask:
- **Release type**: dev, rc, or full release
- **Version number**: base version (e.g. `0.15.0`)

Map to tag suffix:
| Type | Example tag |
|------|-------------|
| dev  | `polars-list-ext-v0.15.0.dev1` |
| rc   | `polars-list-ext-v0.15.0rc1` |
| full | `polars-list-ext-v0.15.0` |

For dev/rc, also ask for the pre-release number (default: 1).

### 2. Update the version in `packages/polars-list-ext/Cargo.toml`

Edit the `version` field under `[package]` to match the new version (use PEP 440 / semver as appropriate).

### 3. Commit the version bump

```bash
git add packages/polars-list-ext/Cargo.toml
git commit -m "chore: bump polars-list-ext to <version>"
```

### 4. Create and push the tag

```bash
git tag <tag>
git push origin HEAD <tag>
```

This triggers the `publish-polars-list-ext.yml` workflow which builds wheels for all platforms and publishes to PyPI (if `PYPI_API_TOKEN` is set).

### 5. Confirm

Tell the user the tag that was pushed and link them to:
`https://github.com/AlexanderNenninger/data-warehousing-with-polars/actions`
