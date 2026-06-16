---
name: release-dwp
description: 'Create and publish a new release of data-warehousing-with-polars. Use when releasing a new version: dev, release candidate (rc), or full release. Reads the current version from pyproject.toml automatically, checks existing tags, then asks whether to do a minor or major bump and what release type (dev/rc/full). Updates pyproject.toml, commits, creates and pushes the git tag to trigger the publish workflow.'
argument-hint: 'Optional: version number (e.g. 0.2.0)'
---

# Release data-warehousing-with-polars

## When to Use
- Publishing a new dev, rc, or stable version of the `data-warehousing-with-polars` Python package
- Triggering the `publish-dwp` GitHub Actions workflow

## Procedure

### 1. Read the current version

Read `packages/data-warehousing-with-polars/pyproject.toml` and extract the `version` field from `[project]`.

### 2. Check existing tags

Run `git tag --list 'dwp-v*' | sort -V` to retrieve existing release tags.

### 3. Ask the user for release details

Tell the user the current version and the existing tags, then present four options derived from it. Mark any option whose tag already exists as `(already exists)`. Strip any existing pre-release suffix from the current version before computing tags (e.g. `0.1.0.dev1` → base `0.1.0`). Apply the minor/major bump only for `minor` and `major` options.

| Option | Description | Example tag |
|--------|-------------|-------------|
| **minor** | Bump minor, full release | `dwp-v0.2.0` |
| **major** | Bump major, full release | `dwp-v1.0.0` |
| **rc** | Current version, release candidate | `dwp-v0.1.0rc1` |
| **dev** | Current version, dev pre-release | `dwp-v0.1.0.dev1` |

For rc/dev, also ask for the pre-release number (default: 1).

### 4. Update the version in `packages/data-warehousing-with-polars/pyproject.toml`

Edit the `version` field under `[project]` to match the new version (use PEP 440 format, e.g. `0.2.0`, `0.2.0rc1`, `0.2.0.dev1`).

### 5. Commit the version bump

```bash
git add packages/data-warehousing-with-polars/pyproject.toml
git commit -m "chore: bump data-warehousing-with-polars to <version>"
```

### 6. Create and push the tag

```bash
git tag <tag>
git push origin HEAD <tag>
```

This triggers the `publish-dwp.yml` workflow which builds and publishes to PyPI (if `PYPI_API_TOKEN` is set).

### 7. Confirm

Tell the user the tag that was pushed and link them to:
`https://github.com/AlexanderNenninger/data-warehousing-with-polars/actions`
