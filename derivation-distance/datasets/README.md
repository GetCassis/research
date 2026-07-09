# Benchmark datasets

Three public analytics datasets, each with a published context layer
(`ontology.json`: table & column descriptions, domain prose, named metrics with
SQL expressions) and the gzipped CSVs of the actual rows.

| Dataset | Data | Tables | License |
|---|---|---|---|
| [`jaffle-shop/`](jaffle-shop/) | dbt Jaffle Shop reshaped to a 6-store cafe chain | 6 | generated (Apache-2.0 tooling) |
| [`thelook/`](thelook/) | theLook e-commerce (Google BigQuery public data, synthetic) | 6 | CC BY 4.0 |
| [`olist/`](olist/) | Olist Brazilian marketplace (real, anonymized) | 11 | **CC BY-NC-SA 4.0 (non-commercial)** |

Each dataset folder holds:

- `ontology.json` — the published context model the benchmark renders contexts from.
- `SOURCE.md` — where the data comes from, how it was prepared, license.
- `data/` — the rows as gzipped CSV, one file per table (**not in git**, see below).

## Getting the data

The CSVs (~70 MB total, 23 files) are distributed as a release archive, not in git:

```bash
python datasets/download.py          # fetch + verify checksums
python datasets/download.py --check  # verify only (no network)
```

Every file is verified against `checksums.json` (SHA-256). The archive URL can be
overridden with `--url` or the `DERIVATION_DISTANCE_DATA_URL` environment variable.

## Using the data directly

Column names are UPPERCASE (matching the ontology's physical table and column names).

```sql
-- DuckDB reads .gz natively
SELECT count(*) FROM read_csv_auto('datasets/olist/data/orders.csv.gz');
```
