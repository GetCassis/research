"""DuckDB loader + introspection. See docs/CONTRACT.md.

Loads datasets/<ds>/data/*.csv.gz into data/<ds>.duckdb, one table per file,
table name = uppercased file stem (matching the ontology's physical names).
"""
import glob
import json
import os

import duckdb

import config

_schema_cache = {}


def schema_name(ds):
    """The single physical schema the ontology declares for this dataset (e.g. JAFFLE)."""
    if ds not in _schema_cache:
        o = json.load(open(config.ontology_path(ds)))
        names = {t.get("schema_name") for t in o["tables"] if t.get("schema_name")}
        _schema_cache[ds] = sorted(names)[0] if names else "MAIN"
    return _schema_cache[ds]


def _table_name(csv_path):
    stem = os.path.basename(csv_path)
    for suffix in (".csv.gz", ".csv"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.upper()


def build(ds, force=False):
    """Idempotently build data/<ds>.duckdb from the gzipped CSVs."""
    path = config.db_path(ds)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    files = sorted(glob.glob(os.path.join(config.dataset_dir(ds), "data", "*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"no CSVs for {ds} in {config.dataset_dir(ds)}/data")
    sch = schema_name(ds)
    con = duckdb.connect(path)
    try:
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{sch}"')
        existing = {
            r[1] for r in con.execute(
                "SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema=?",
                [sch],
            ).fetchall()
        }
        for f in files:
            t = _table_name(f)
            if t in existing and not force:
                continue
            con.execute(f'DROP TABLE IF EXISTS "{sch}"."{t}"')
            con.execute(
                f'CREATE TABLE "{sch}"."{t}" AS SELECT * FROM read_csv_auto(?, sample_size=-1)',
                [f],
            )
        return path
    finally:
        con.close()


def connect(ds):
    path = config.db_path(ds)
    if not os.path.exists(path):
        build(ds)
    con = duckdb.connect(path, read_only=True)
    # Resolve both qualified (JAFFLE.ORDERS) and bare (ORDERS) names.
    con.execute(f"SET search_path = '{schema_name(ds)},main'")
    return con


def schema(ds):
    """{TABLE: [{"name","type"}, ...]} from the built db."""
    sch = schema_name(ds)
    con = connect(ds)
    try:
        out = {}
        rows = con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema=? ORDER BY table_name, ordinal_position",
            [sch],
        ).fetchall()
        for table, col, typ in rows:
            out.setdefault(table, []).append({"name": col, "type": typ})
        return out
    finally:
        con.close()


def run_sql(ds, sql):
    """Execute sql read-only. Returns (columns, rows). Raises on error."""
    con = connect(ds)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return cols, rows
    finally:
        con.close()


if __name__ == "__main__":
    ds = "jaffle-shop"
    print("building", ds)
    print(build(ds))
    sch = schema(ds)
    for t, cols in sch.items():
        print(f"  {t}: {[c['name'] for c in cols]}")
    cols, rows = run_sql(ds, 'SELECT COUNT(*) FROM ORDERS')
    print("ORDERS count:", rows[0][0])
    cols, rows = run_sql(
        ds,
        'SELECT s.STORE_NAME, SUM(o.SUBTOTAL) rev FROM ORDERS o '
        'JOIN STORES s ON o.STORE_ID=s.STORE_ID GROUP BY 1 ORDER BY rev DESC LIMIT 5',
    )
    print("top stores:", rows)
