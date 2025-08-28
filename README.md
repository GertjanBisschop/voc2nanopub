# Publish Vocabulary from LinkML to Nanopub

This GitHub workflow automates the process of publishing vocabulary terms defined in a [LinkML](https://linkml.io/) schema as [Nanopublications](https://nanopub.net/). It supports signing and publishing nanopubs, generating `.htaccess` files for term redirection, and controlling publication behavior via workflow inputs.

## Workflow Overview

This workflow is designed to be **reusable** via [`workflow_call`](https://docs.github.com/en/actions/using-workflows/reusing-workflows) so it can be triggered from other workflows within the repository.

### Key features
- Checkout the correct Git reference (branch, tag, or commit)
- Parse a LinkML schema and data source
- Sign and optionally publish nanopublications
- Generate `.htaccess` redirection files for vocabulary terms

---

## Inputs

| Name                  | Required | Type      | Default        | Description                                                                                 |
|-----------------------|----------|-----------|----------------|---------------------------------------------------------------------------------------------|
| `ref`                 | ✅       | `string`  | —              | Git reference (branch, tag, or commit SHA) to checkout.                                      |
| `version`             | ❌       | `string`  | `""`            | Version of the vocabulary being published.                                                   |
| `linkml_schema`        | ✅       | `string`  | —              | Path to the LinkML schema file.                                                              |
| `data`                | ❌       | `string`  | —              | Path to the data file containing entities to publish. Alternative to `input_artefact_name`.  |
| `input_artefact_name`  | ❌       | `string`  | —              | Name of the artefact used as workflow input instead of a data file.                          |
| `changelog`           | ✅       | `string`  | —              | Path to the changelog file containing terms to be published.                                 |
| `target_name`          | ✅       | `string`  | —              | Name of the target entity list in the data file.                                             |
| `dry_run`              | ❌       | `boolean` | `false`         | If `true`, signs nanopubs but does **not** publish them.                                     |
| `output_htaccess`      | ❌       | `string`  | `htaccess.txt`  | Output path for the `.htaccess` file.                                                        |
| `output_artefact_name` | ❌       | `string`  | `htaccess.txt`  | Name of the artefact containing the generated `.htaccess` file.                              |
| `preflabel`            | ❌       | `string`  | —              | Field used to create identifiers using a hash function (default: `label`).                   |
| `id_key`               | ❌       | `string`  | —              | Field containing the URI (default: `id`).                                                    |
| `type_prefix`          | ✅       | `string`  | —              | Prefix added to identifiers following the namespace.                                        |

---

## Secrets

| Name                  | Required | Description                              |
|-----------------------|----------|------------------------------------------|
| `token`               | ✅       | GitHub token for repository checkout.     |
| `NANOPUB_ORCID_ID`     | ✅       | ORCID ID for nanopub profile.             |
| `NANOPUB_NAME`         | ✅       | Name associated with the nanopub profile. |
| `NANOPUB_PRIVATE_KEY`  | ✅       | Private key for signing nanopubs.         |
| `NANOPUB_PUBLIC_KEY`   | ✅       | Public key for nanopub profile.           |
| `NANOPUB_INTRO_URI`    | ✅       | Introduction nanopub URI.                 |

---

## Usage Example

To call this workflow from another workflow:

```yaml
name: Publish Vocabulary

on:
  push:
    branches:
      - main

jobs:
  publish-vocabulary:
    uses: ./.github/workflows/publish-vocab-nanopub.yml
    with:
      ref: main
      version: "1.0.0"
      linkml_schema: "schema/vocabulary.yaml"
      changelog: "data/changelog.yaml"
      target_name: "terms"
      type_prefix: "VOCAB"
    secrets:
      token: ${{ secrets.GITHUB_TOKEN }}
      NANOPUB_ORCID_ID: ${{ secrets.NANOPUB_ORCID_ID }}
      NANOPUB_NAME: ${{ secrets.NANOPUB_NAME }}
      NANOPUB_PRIVATE_KEY: ${{ secrets.NANOPUB_PRIVATE_KEY }}
      NANOPUB_PUBLIC_KEY: ${{ secrets.NANOPUB_PUBLIC_KEY }}
      NANOPUB_INTRO_URI: ${{ secrets.NANOPUB_INTRO_URI }}
