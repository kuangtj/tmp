#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect local IPM SQLite database.

Usage:
  python scripts/inspect_ipm_db.py
  python scripts/inspect_ipm_db.py --db data/db/ipm_eagle.sqlite
  python scripts/inspect_ipm_db.py --doc-id doi_xxx --show-schema --limit 10
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DB = Path("data/db/ipm_eagle.sqlite")

RAW_TABLES = [
    "raw_document",
    "raw_asset",
    "raw_text_block",
    "raw_figure",
    "raw_table",
    "planned_tasks",
]

STG_TABLES = [
    "stg_agent",
    "stg_relation",
    "stg_relation_participant",
    "stg_assay",
    "stg_structure_candidate",
    "structure_qc_result",
    "stg_component_relation",
    "stg_structure_resolution_task",
    "stg_sequence_resolution_task",
    "stg_sequence_candidate",
]

QC_TABLES = [
    "qc_issue",
    "review_task",
    "audit_event",
    "stg_resolution_event",
    "task_queue",
]

CORE_TABLES = [
    "ref",
    "agent",
    "relation",
    "relation_participant",
    "assay",
]

KNOWN_TABLES = RAW_TABLES + STG_TABLES + QC_TABLES + CORE_TABLES


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def q(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
    return list(conn.execute(sql, params).fetchall())


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def get_tables(conn: sqlite3.Connection) -> List[str]:
    return [
        r["name"]
        for r in q(
            conn,
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
    ]


def get_indexes(conn: sqlite3.Connection) -> List[str]:
    return [
        r["name"]
        for r in q(
            conn,
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
    ]


def get_columns(conn: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    if not table_exists(conn, table):
        return []
    rows = q(conn, f"PRAGMA table_info({quote_ident(table)})")
    return [dict(r) for r in rows]


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def count_rows(conn: sqlite3.Connection, table: str, doc_id: Optional[str] = None) -> int:
    if not table_exists(conn, table):
        return -1
    cols = {c["name"] for c in get_columns(conn, table)}
    if doc_id and "doc_id" in cols:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {quote_ident(table)} WHERE doc_id=?", (doc_id,)).fetchone()
    else:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {quote_ident(table)}").fetchone()
    return int(row["n"])


def distinct_counts(conn: sqlite3.Connection, table: str, col: str, doc_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    if not table_exists(conn, table):
        return []
    cols = {c["name"] for c in get_columns(conn, table)}
    if col not in cols:
        return []
    where = ""
    params: Tuple[Any, ...] = ()
    if doc_id and "doc_id" in cols:
        where = "WHERE doc_id=?"
        params = (doc_id,)
    sql = f"""
    SELECT COALESCE(CAST({quote_ident(col)} AS TEXT), '') AS value, COUNT(*) AS n
    FROM {quote_ident(table)}
    {where}
    GROUP BY COALESCE(CAST({quote_ident(col)} AS TEXT), '')
    ORDER BY n DESC, value ASC
    LIMIT {int(limit)}
    """
    return [dict(r) for r in q(conn, sql, params)]


def sample_rows(conn: sqlite3.Connection, table: str, doc_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    if not table_exists(conn, table):
        return []
    cols = {c["name"] for c in get_columns(conn, table)}
    where = ""
    params: Tuple[Any, ...] = ()
    if doc_id and "doc_id" in cols:
        where = "WHERE doc_id=?"
        params = (doc_id,)
    order_col = None
    for c in ["created_at", "updated_at", "doc_id"]:
        if c in cols:
            order_col = c
            break
    order = f"ORDER BY {quote_ident(order_col)} DESC" if order_col else ""
    sql = f"SELECT * FROM {quote_ident(table)} {where} {order} LIMIT {int(limit)}"
    out = []
    for r in q(conn, sql, params):
        d = dict(r)
        # Trim very long fields for readable JSON/MD.
        for k, v in list(d.items()):
            if isinstance(v, str) and len(v) > 500:
                d[k] = v[:500] + "...<trimmed>"
        out.append(d)
    return out


def inspect_documents(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    if not table_exists(conn, "raw_document"):
        return []
    cols = {c["name"] for c in get_columns(conn, "raw_document")}
    select_cols = [c for c in ["doc_id", "title", "doi", "pmid", "status", "source_pdf_path", "supplement_dir", "created_at"] if c in cols]
    if not select_cols:
        return []
    sql = "SELECT " + ", ".join(quote_ident(c) for c in select_cols) + " FROM raw_document ORDER BY created_at DESC"
    return [dict(r) for r in q(conn, sql)]


def inspect_per_doc(conn: sqlite3.Connection, doc_id: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"doc_id": doc_id, "counts": {}}
    for t in KNOWN_TABLES:
        n = count_rows(conn, t, doc_id=doc_id)
        if n >= 0:
            out["counts"][t] = n

    out["planned_task_types"] = distinct_counts(conn, "planned_tasks", "task_type", doc_id)
    out["planned_task_status"] = distinct_counts(conn, "planned_tasks", "status", doc_id)
    out["asset_types"] = distinct_counts(conn, "raw_asset", "asset_type", doc_id)
    out["structure_candidate_sources"] = distinct_counts(conn, "stg_structure_candidate", "source_tool", doc_id)
    out["structure_qc_decisions"] = distinct_counts(conn, "structure_qc_result", "auto_decision", doc_id)
    out["component_roles"] = distinct_counts(conn, "stg_component_relation", "component_role", doc_id)
    out["structure_task_status"] = distinct_counts(conn, "stg_structure_resolution_task", "status", doc_id)
    out["sequence_task_status"] = distinct_counts(conn, "stg_sequence_resolution_task", "status", doc_id)
    out["review_task_status"] = distinct_counts(conn, "review_task", "status", doc_id)
    return out


def inspect_db(conn: sqlite3.Connection, doc_id: Optional[str], show_schema: bool, sample_limit: int) -> Dict[str, Any]:
    tables = get_tables(conn)
    indexes = get_indexes(conn)
    counts = {t: count_rows(conn, t, doc_id=doc_id) for t in tables}
    missing_known_tables = [t for t in KNOWN_TABLES if t not in tables]

    data: Dict[str, Any] = {
        "generated_at": now(),
        "sqlite_version": conn.execute("SELECT sqlite_version() AS v").fetchone()["v"],
        "table_count": len(tables),
        "index_count": len(indexes),
        "tables": tables,
        "indexes": indexes,
        "missing_known_tables": missing_known_tables,
        "row_counts": counts,
        "documents": inspect_documents(conn),
        "selected_doc": doc_id,
    }

    docs = [doc_id] if doc_id else [d.get("doc_id") for d in data["documents"] if d.get("doc_id")]
    data["per_doc"] = {d: inspect_per_doc(conn, d) for d in docs}

    data["samples"] = {}
    for t in [
        "raw_document",
        "planned_tasks",
        "stg_agent",
        "stg_relation",
        "stg_relation_participant",
        "stg_assay",
        "stg_structure_resolution_task",
        "stg_sequence_resolution_task",
        "stg_structure_candidate",
        "stg_component_relation",
        "stg_sequence_candidate",
        "qc_issue",
        "review_task",
    ]:
        rows = sample_rows(conn, t, doc_id=doc_id, limit=sample_limit)
        if rows:
            data["samples"][t] = rows

    if show_schema:
        data["schema"] = {t: get_columns(conn, t) for t in tables}

    return data


def md_table(rows: List[List[Any]], headers: List[str]) -> str:
    def cell(x: Any) -> str:
        s = "" if x is None else str(x)
        return s.replace("\n", " ").replace("|", "\\|")
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(cell(x) for x in r) + " |")
    return "\n".join(out)


def render_markdown(data: Dict[str, Any], db_path: Path) -> str:
    lines: List[str] = []
    lines.append(f"# IPM SQLite Database Inspection")
    lines.append("")
    lines.append(f"- DB: `{db_path}`")
    lines.append(f"- Generated: `{data['generated_at']}`")
    lines.append(f"- SQLite: `{data['sqlite_version']}`")
    lines.append(f"- Tables: `{data['table_count']}`")
    lines.append(f"- Indexes: `{data['index_count']}`")
    if data.get("selected_doc"):
        lines.append(f"- Selected doc_id: `{data['selected_doc']}`")
    lines.append("")

    if data["missing_known_tables"]:
        lines.append("## Missing expected tables")
        lines.append("")
        lines.append("`" + "`, `".join(data["missing_known_tables"]) + "`")
        lines.append("")

    lines.append("## Row counts")
    rows = [[t, data["row_counts"][t]] for t in data["tables"]]
    lines.append(md_table(rows, ["table", "rows"]))
    lines.append("")

    lines.append("## Documents")
    docs = data.get("documents", [])
    if docs:
        headers = list(docs[0].keys())
        lines.append(md_table([[d.get(h, "") for h in headers] for d in docs], headers))
    else:
        lines.append("No rows in raw_document or table missing.")
    lines.append("")

    for doc_id, doc_data in data.get("per_doc", {}).items():
        lines.append(f"## Per-doc summary: `{doc_id}`")
        lines.append("")
        cnt_rows = [[k, v] for k, v in doc_data.get("counts", {}).items()]
        lines.append(md_table(cnt_rows, ["table", "rows for doc"])); lines.append("")
        for key in [
            "planned_task_types",
            "planned_task_status",
            "asset_types",
            "structure_candidate_sources",
            "structure_qc_decisions",
            "component_roles",
            "structure_task_status",
            "sequence_task_status",
            "review_task_status",
        ]:
            vals = doc_data.get(key) or []
            if vals:
                lines.append(f"### {key}")
                lines.append(md_table([[x.get("value", ""), x.get("n", 0)] for x in vals], ["value", "n"]))
                lines.append("")

    if data.get("samples"):
        lines.append("## Samples")
        for table, rows in data["samples"].items():
            lines.append(f"### `{table}`")
            lines.append("```json")
            lines.append(json.dumps(rows, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")

    if data.get("schema"):
        lines.append("## Schema")
        for table, cols in data["schema"].items():
            lines.append(f"### `{table}`")
            lines.append(md_table([[c["cid"], c["name"], c["type"], c["notnull"], c["dflt_value"], c["pk"]] for c in cols], ["cid", "name", "type", "notnull", "default", "pk"]))
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect local IPM SQLite database")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="SQLite DB path")
    ap.add_argument("--doc-id", default="", help="Inspect one doc_id only")
    ap.add_argument("--show-schema", action="store_true", help="Include PRAGMA table_info for all tables")
    ap.add_argument("--limit", type=int, default=5, help="Sample rows per table")
    ap.add_argument("--out-dir", default="data/staging/db_inspect", help="Output directory")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect(db_path)
    try:
        data = inspect_db(conn, args.doc_id or None, args.show_schema, args.limit)
    finally:
        conn.close()

    json_path = out_dir / "db_inspect.json"
    md_path = out_dir / "db_inspect.md"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(data, db_path), encoding="utf-8")

    print(render_markdown(data, db_path))
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
