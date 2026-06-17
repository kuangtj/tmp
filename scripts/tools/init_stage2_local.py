#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initialize local SQLite database for the IPM literature extraction pipeline.

New pipeline design:
PDF + Supplement
  -> parse_pdf
  -> plan_paper_vlm
  -> extract_supplement_direct_structures
  -> extract_ipm_knowledge
  -> extract_assays
  -> build_missing_info_tasks
       - structure resolution tasks
       - sequence resolution tasks
  -> supplement sequence / participant standardization
  -> unresolved structure resolution by UniParser/VLM/reconstruction
  -> global_qc
  -> core load

Core tables are exactly:
  ref
  agent
  relation
  relation_participant
  assay

Usage:
  python scripts/init_stage2_local.py --reset
  python scripts/init_stage2_local.py --db data/db/ipm_eagle.sqlite --reset
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


DEFAULT_DB = Path("data/db/ipm_eagle.sqlite")
DEFAULT_DIRS = [
    "data/db",
    "data/raw",
    "data/work",
    "data/staging",
    "data/review",
    "data/core",
    "logs",
]


SCHEMA_SQL = r"""
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

-- ============================================================
-- Raw layer
-- ============================================================

CREATE TABLE IF NOT EXISTS raw_document (
    doc_id              TEXT PRIMARY KEY,
    title               TEXT,
    doi                 TEXT,
    pmid                TEXT,
    journal             TEXT,
    year                TEXT,
    source_pdf_path     TEXT,
    supplement_dir      TEXT,
    status              TEXT DEFAULT 'created',
    metadata_json       TEXT DEFAULT '{}',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_asset (
    asset_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    asset_type          TEXT,
    page_no             INTEGER,
    figure_ref          TEXT,
    table_ref           TEXT,
    file_path           TEXT,
    metadata_json       TEXT DEFAULT '{}',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS raw_text_block (
    block_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    page_no             INTEGER,
    section             TEXT,
    text                TEXT,
    metadata_json       TEXT DEFAULT '{}',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS raw_figure (
    figure_id           TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    page_no             INTEGER,
    figure_ref          TEXT,
    file_path           TEXT,
    caption             TEXT,
    metadata_json       TEXT DEFAULT '{}',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS raw_table (
    table_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    page_no             INTEGER,
    table_ref           TEXT,
    file_path           TEXT,
    table_json          TEXT DEFAULT '{}',
    metadata_json       TEXT DEFAULT '{}',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS planned_tasks (
    task_id             TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    asset_id            TEXT,
    asset_type          TEXT,
    task_type           TEXT,
    agents_json         TEXT DEFAULT '[]',
    priority            INTEGER DEFAULT 50,
    reason              TEXT,
    status              TEXT DEFAULT 'pending',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE
);

-- ============================================================
-- Staging: IPM knowledge extraction
-- ============================================================

CREATE TABLE IF NOT EXISTS stg_agent (
    stg_id              TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    name                TEXT NOT NULL,
    normalized_name     TEXT,
    aliases_json        TEXT DEFAULT '[]',
    agent_type          TEXT,
    canonical_smiles    TEXT,
    structure_status    TEXT DEFAULT 'missing',
    structure_json      TEXT DEFAULT '{}',
    sequence_json       TEXT DEFAULT '{}',
    ids_json            TEXT DEFAULT '{}',
    evidence_json       TEXT DEFAULT '[]',
    record_json         TEXT DEFAULT '{}',
    confidence          REAL,
    review_required     INTEGER DEFAULT 1,
    qc_reasons_json     TEXT DEFAULT '[]',
    status              TEXT DEFAULT 'pending_qc',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS stg_relation (
    relation_id         TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    relation_type       TEXT,
    modality            TEXT,
    outcome_class       TEXT,
    mechanism_route     TEXT,
    intended_effect     TEXT,
    relation_name       TEXT,
    evidence_text       TEXT,
    evidence_source     TEXT,
    source_asset_id     TEXT,
    source_block_id     TEXT,
    source_page_no      INTEGER,
    record_json         TEXT DEFAULT '{}',
    raw_output          TEXT DEFAULT '{}',
    confidence          REAL,
    review_required     INTEGER DEFAULT 1,
    qc_reasons_json     TEXT DEFAULT '[]',
    status              TEXT DEFAULT 'pending_qc',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS stg_relation_participant (
    participant_id          TEXT PRIMARY KEY,
    relation_id             TEXT,
    doc_id                  TEXT NOT NULL,
    entity_name             TEXT NOT NULL,
    canonical_name          TEXT,
    entity_type             TEXT,
    role                    TEXT,
    role_detail             TEXT,
    species                 TEXT,
    taxon_id                TEXT,
    agent_stg_id            TEXT,
    ids_json                TEXT DEFAULT '{}',
    sequence_json           TEXT DEFAULT '{}',
    structure_json          TEXT DEFAULT '{}',
    standardization_json    TEXT DEFAULT '{}',
    evidence_text           TEXT,
    source_asset_id         TEXT,
    source_block_id         TEXT,
    source_page_no          INTEGER,
    raw_output              TEXT DEFAULT '{}',
    confidence              REAL,
    review_required         INTEGER DEFAULT 1,
    qc_reasons_json         TEXT DEFAULT '[]',
    status                  TEXT DEFAULT 'pending_qc',
    created_at              TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at              TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE,
    FOREIGN KEY (relation_id) REFERENCES stg_relation(relation_id) ON DELETE CASCADE,
    FOREIGN KEY (agent_stg_id) REFERENCES stg_agent(stg_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS stg_assay (
    assay_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    relation_id         TEXT,
    agent_stg_id        TEXT,
    agent_name          TEXT,
    target_name         TEXT,
    assay_category      TEXT,
    assay_type          TEXT,
    assay_format        TEXT,
    primary_metric      TEXT,
    primary_value       TEXT,
    primary_qualifier   TEXT,
    primary_unit        TEXT,
    secondary_metrics_json TEXT DEFAULT '{}',
    cell_line           TEXT,
    species             TEXT,
    dose                TEXT,
    dose_unit           TEXT,
    treatment_time      TEXT,
    treatment_time_unit TEXT,
    condition_json      TEXT DEFAULT '{}',
    evidence_text       TEXT,
    source_asset_id     TEXT,
    source_table_id     TEXT,
    source_figure_id    TEXT,
    source_page_no      INTEGER,
    record_json         TEXT DEFAULT '{}',
    raw_output          TEXT DEFAULT '{}',
    confidence          REAL,
    review_required     INTEGER DEFAULT 1,
    qc_reasons_json     TEXT DEFAULT '[]',
    status              TEXT DEFAULT 'pending_qc',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE,
    FOREIGN KEY (relation_id) REFERENCES stg_relation(relation_id) ON DELETE SET NULL,
    FOREIGN KEY (agent_stg_id) REFERENCES stg_agent(stg_id) ON DELETE SET NULL
);

-- ============================================================
-- Staging: compound structure candidates and component graph
-- ============================================================

CREATE TABLE IF NOT EXISTS stg_structure_candidate (
    candidate_id        TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    asset_id            TEXT,
    source_task_id      TEXT,
    image_path          TEXT,
    image_name          TEXT,
    candidate_index     INTEGER,
    compound_name       TEXT,
    molecule_label      TEXT,
    smiles              TEXT,
    canonical_smiles    TEXT,
    rdkit_valid         INTEGER,
    rdkit_error         TEXT,
    source_tool         TEXT,
    smiles_source       TEXT,
    bbox_json           TEXT DEFAULT '{}',
    raw_context_json    TEXT DEFAULT '{}',
    raw_output          TEXT DEFAULT '{}',
    raw_json            TEXT DEFAULT '{}',
    status              TEXT DEFAULT 'pending_qc',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS structure_qc_result (
    candidate_id        TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    asset_id            TEXT,
    qc_score            REAL,
    structure_class     TEXT,
    auto_decision       TEXT,
    qc_flags_json       TEXT DEFAULT '[]',
    vlm_qc_json         TEXT DEFAULT '{}',
    review_xlsx_path    TEXT,
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (candidate_id) REFERENCES stg_structure_candidate(candidate_id) ON DELETE CASCADE,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS stg_component_relation (
    relation_id         TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    asset_id            TEXT,
    candidate_id        TEXT NOT NULL,
    compound_name       TEXT,
    component_role      TEXT,
    relation_type       TEXT,
    evidence_text       TEXT,
    figure_ref          TEXT,
    confidence          REAL,
    review_required     INTEGER DEFAULT 1,
    raw_output          TEXT DEFAULT '{}',
    status              TEXT DEFAULT 'pending_qc',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE,
    FOREIGN KEY (candidate_id) REFERENCES stg_structure_candidate(candidate_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS stg_structure_resolution_task (
    task_id                 TEXT PRIMARY KEY,
    doc_id                  TEXT NOT NULL,
    agent_stg_id            TEXT,
    agent_name              TEXT NOT NULL,
    normalized_agent_name   TEXT,
    reason                  TEXT,
    priority                TEXT DEFAULT 'medium',
    status                  TEXT DEFAULT 'pending',
    resolution_stage        TEXT DEFAULT 'pending',
    candidate_pages_json    TEXT DEFAULT '[]',
    matched_candidate_id    TEXT,
    final_smiles            TEXT,
    review_required         INTEGER DEFAULT 1,
    raw_output              TEXT DEFAULT '{}',
    created_at              TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at              TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE,
    FOREIGN KEY (agent_stg_id) REFERENCES stg_agent(stg_id) ON DELETE SET NULL
);

-- ============================================================
-- Staging: biological sequence / ID resolution
-- ============================================================

CREATE TABLE IF NOT EXISTS stg_sequence_resolution_task (
    task_id                 TEXT PRIMARY KEY,
    doc_id                  TEXT NOT NULL,
    participant_id          TEXT,
    entity_name             TEXT NOT NULL,
    canonical_name          TEXT,
    entity_type             TEXT,
    role_hint               TEXT,
    species_hint            TEXT,
    sequence_need_type      TEXT,
    priority                TEXT DEFAULT 'medium',
    status                  TEXT DEFAULT 'pending',
    result_json             TEXT DEFAULT '{}',
    review_required         INTEGER DEFAULT 1,
    created_at              TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at              TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE,
    FOREIGN KEY (participant_id) REFERENCES stg_relation_participant(participant_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS stg_sequence_candidate (
    candidate_id            TEXT PRIMARY KEY,
    doc_id                  TEXT NOT NULL,
    participant_id          TEXT,
    entity_name             TEXT,
    canonical_name          TEXT,
    sequence_type           TEXT,
    sequence_scope          TEXT,
    sequence                TEXT,
    sequence_length         INTEGER,
    accession               TEXT,
    accession_type          TEXT,
    organism                TEXT,
    taxon_id                TEXT,
    source_tool             TEXT,
    source_file             TEXT,
    source_table_id         TEXT,
    evidence_text           TEXT,
    confidence              REAL,
    review_required         INTEGER DEFAULT 1,
    raw_output              TEXT DEFAULT '{}',
    status                  TEXT DEFAULT 'pending_qc',
    created_at              TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at              TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE,
    FOREIGN KEY (participant_id) REFERENCES stg_relation_participant(participant_id) ON DELETE SET NULL
);

-- ============================================================
-- QC / review / audit
-- ============================================================

CREATE TABLE IF NOT EXISTS qc_issue (
    issue_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    object_type         TEXT,
    object_id           TEXT,
    severity            TEXT,
    issue_code          TEXT,
    message             TEXT,
    auto_fixable        INTEGER DEFAULT 0,
    payload_json        TEXT DEFAULT '{}',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS review_task (
    review_id           TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    object_type         TEXT,
    object_id           TEXT,
    reason              TEXT,
    status              TEXT DEFAULT 'pending',
    human_decision      TEXT,
    human_note          TEXT,
    payload_json        TEXT DEFAULT '{}',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audit_event (
    audit_id            TEXT PRIMARY KEY,
    doc_id              TEXT,
    object_type         TEXT,
    object_id           TEXT,
    action              TEXT,
    before_json         TEXT DEFAULT '{}',
    after_json          TEXT DEFAULT '{}',
    note                TEXT,
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stg_resolution_event (
    event_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    task_id             TEXT,
    task_type           TEXT,
    stage               TEXT,
    status              TEXT,
    message             TEXT,
    payload_json        TEXT DEFAULT '{}',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS task_queue (
    task_id             TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    queue_name          TEXT,
    task_type           TEXT,
    payload_json        TEXT DEFAULT '{}',
    status              TEXT DEFAULT 'pending',
    error               TEXT,
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES raw_document(doc_id) ON DELETE CASCADE
);

-- ============================================================
-- Core layer
-- Core table names are exactly: ref, agent, relation,
-- relation_participant, assay
-- ============================================================

CREATE TABLE IF NOT EXISTS ref (
    ref_id              TEXT PRIMARY KEY,
    doc_id              TEXT UNIQUE,
    doi                 TEXT,
    pmid                TEXT,
    title               TEXT,
    journal             TEXT,
    year                TEXT,
    source_pdf_path     TEXT,
    supplement_dir      TEXT,
    record_json         TEXT DEFAULT '{}',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent (
    agent_id            TEXT PRIMARY KEY,
    ref_id              TEXT,
    doc_id              TEXT,
    name                TEXT NOT NULL,
    normalized_name     TEXT,
    aliases_json        TEXT DEFAULT '[]',
    agent_type          TEXT,
    canonical_smiles    TEXT,
    structure_status    TEXT,
    structure_json      TEXT DEFAULT '{}',
    sequence_json       TEXT DEFAULT '{}',
    ids_json            TEXT DEFAULT '{}',
    source_stg_id       TEXT,
    record_json         TEXT DEFAULT '{}',
    provenance_json     TEXT DEFAULT '{}',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ref_id) REFERENCES ref(ref_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS relation (
    relation_id         TEXT PRIMARY KEY,
    ref_id              TEXT,
    doc_id              TEXT,
    relation_type       TEXT,
    modality            TEXT,
    outcome_class       TEXT,
    mechanism_route     TEXT,
    intended_effect     TEXT,
    relation_name       TEXT,
    evidence_text       TEXT,
    source_stg_id       TEXT,
    record_json         TEXT DEFAULT '{}',
    provenance_json     TEXT DEFAULT '{}',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ref_id) REFERENCES ref(ref_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS relation_participant (
    participant_id      TEXT PRIMARY KEY,
    relation_id         TEXT NOT NULL,
    agent_id            TEXT,
    ref_id              TEXT,
    doc_id              TEXT,
    entity_name         TEXT NOT NULL,
    canonical_name      TEXT,
    entity_type         TEXT,
    role                TEXT,
    role_detail         TEXT,
    species             TEXT,
    taxon_id            TEXT,
    ids_json            TEXT DEFAULT '{}',
    sequence_json       TEXT DEFAULT '{}',
    structure_json      TEXT DEFAULT '{}',
    source_stg_id       TEXT,
    record_json         TEXT DEFAULT '{}',
    provenance_json     TEXT DEFAULT '{}',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (relation_id) REFERENCES relation(relation_id) ON DELETE CASCADE,
    FOREIGN KEY (agent_id) REFERENCES agent(agent_id) ON DELETE SET NULL,
    FOREIGN KEY (ref_id) REFERENCES ref(ref_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS assay (
    assay_id            TEXT PRIMARY KEY,
    relation_id         TEXT,
    agent_id            TEXT,
    ref_id              TEXT,
    doc_id              TEXT,
    agent_name          TEXT,
    target_name         TEXT,
    assay_category      TEXT,
    assay_type          TEXT,
    assay_format        TEXT,
    primary_metric      TEXT,
    primary_value       TEXT,
    primary_qualifier   TEXT,
    primary_unit        TEXT,
    secondary_metrics_json TEXT DEFAULT '{}',
    cell_line           TEXT,
    species             TEXT,
    dose                TEXT,
    dose_unit           TEXT,
    treatment_time      TEXT,
    treatment_time_unit TEXT,
    condition_json      TEXT DEFAULT '{}',
    evidence_text       TEXT,
    source_stg_id       TEXT,
    record_json         TEXT DEFAULT '{}',
    provenance_json     TEXT DEFAULT '{}',
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (relation_id) REFERENCES relation(relation_id) ON DELETE SET NULL,
    FOREIGN KEY (agent_id) REFERENCES agent(agent_id) ON DELETE SET NULL,
    FOREIGN KEY (ref_id) REFERENCES ref(ref_id) ON DELETE SET NULL
);
"""


INDEX_SQL = r"""
CREATE INDEX IF NOT EXISTS idx_raw_asset_doc            ON raw_asset(doc_id);
CREATE INDEX IF NOT EXISTS idx_raw_asset_page           ON raw_asset(doc_id, page_no);
CREATE INDEX IF NOT EXISTS idx_raw_text_doc_page        ON raw_text_block(doc_id, page_no);
CREATE INDEX IF NOT EXISTS idx_raw_figure_doc_page      ON raw_figure(doc_id, page_no);
CREATE INDEX IF NOT EXISTS idx_raw_table_doc_page       ON raw_table(doc_id, page_no);
CREATE INDEX IF NOT EXISTS idx_planned_doc_type         ON planned_tasks(doc_id, task_type, status);
CREATE INDEX IF NOT EXISTS idx_planned_asset            ON planned_tasks(asset_id);

CREATE INDEX IF NOT EXISTS idx_stg_agent_doc_name       ON stg_agent(doc_id, name);
CREATE INDEX IF NOT EXISTS idx_stg_agent_doc_type       ON stg_agent(doc_id, agent_type);
CREATE INDEX IF NOT EXISTS idx_stg_relation_doc         ON stg_relation(doc_id);
CREATE INDEX IF NOT EXISTS idx_stg_rel_part_doc         ON stg_relation_participant(doc_id);
CREATE INDEX IF NOT EXISTS idx_stg_rel_part_relation    ON stg_relation_participant(relation_id);
CREATE INDEX IF NOT EXISTS idx_stg_rel_part_role        ON stg_relation_participant(doc_id, role);
CREATE INDEX IF NOT EXISTS idx_stg_assay_doc            ON stg_assay(doc_id);
CREATE INDEX IF NOT EXISTS idx_stg_assay_relation       ON stg_assay(relation_id);
CREATE INDEX IF NOT EXISTS idx_stg_assay_agent_name     ON stg_assay(doc_id, agent_name);

CREATE INDEX IF NOT EXISTS idx_struct_candidate_doc     ON stg_structure_candidate(doc_id);
CREATE INDEX IF NOT EXISTS idx_struct_candidate_asset   ON stg_structure_candidate(doc_id, asset_id);
CREATE INDEX IF NOT EXISTS idx_struct_candidate_tool    ON stg_structure_candidate(doc_id, source_tool);
CREATE INDEX IF NOT EXISTS idx_struct_candidate_name    ON stg_structure_candidate(doc_id, compound_name);
CREATE INDEX IF NOT EXISTS idx_struct_qc_doc_decision   ON structure_qc_result(doc_id, auto_decision);
CREATE INDEX IF NOT EXISTS idx_component_doc_compound   ON stg_component_relation(doc_id, compound_name);
CREATE INDEX IF NOT EXISTS idx_component_candidate      ON stg_component_relation(candidate_id);
CREATE INDEX IF NOT EXISTS idx_component_role           ON stg_component_relation(doc_id, component_role);
CREATE INDEX IF NOT EXISTS idx_struct_task_doc_status   ON stg_structure_resolution_task(doc_id, status, resolution_stage);
CREATE INDEX IF NOT EXISTS idx_struct_task_agent        ON stg_structure_resolution_task(doc_id, agent_name);

CREATE INDEX IF NOT EXISTS idx_seq_task_doc_status      ON stg_sequence_resolution_task(doc_id, status);
CREATE INDEX IF NOT EXISTS idx_seq_task_participant     ON stg_sequence_resolution_task(participant_id);
CREATE INDEX IF NOT EXISTS idx_seq_candidate_doc        ON stg_sequence_candidate(doc_id);
CREATE INDEX IF NOT EXISTS idx_seq_candidate_part       ON stg_sequence_candidate(participant_id);
CREATE INDEX IF NOT EXISTS idx_seq_candidate_entity     ON stg_sequence_candidate(doc_id, entity_name);

CREATE INDEX IF NOT EXISTS idx_qc_issue_doc             ON qc_issue(doc_id, object_type, object_id);
CREATE INDEX IF NOT EXISTS idx_review_doc_status        ON review_task(doc_id, status);
CREATE INDEX IF NOT EXISTS idx_resolution_event_task    ON stg_resolution_event(task_id, task_type);
CREATE INDEX IF NOT EXISTS idx_task_queue_doc_status    ON task_queue(doc_id, queue_name, status);

CREATE INDEX IF NOT EXISTS idx_ref_doc                  ON ref(doc_id);
CREATE INDEX IF NOT EXISTS idx_ref_doi                  ON ref(doi);
CREATE INDEX IF NOT EXISTS idx_agent_doc_name           ON agent(doc_id, name);
CREATE INDEX IF NOT EXISTS idx_agent_smiles             ON agent(canonical_smiles);
CREATE INDEX IF NOT EXISTS idx_relation_doc             ON relation(doc_id);
CREATE INDEX IF NOT EXISTS idx_relation_part_rel        ON relation_participant(relation_id);
CREATE INDEX IF NOT EXISTS idx_relation_part_doc_role   ON relation_participant(doc_id, role);
CREATE INDEX IF NOT EXISTS idx_assay_doc                ON assay(doc_id);
CREATE INDEX IF NOT EXISTS idx_assay_relation           ON assay(relation_id);
CREATE INDEX IF NOT EXISTS idx_assay_agent              ON assay(agent_id);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize local IPM SQLite database.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path.")
    parser.add_argument("--reset", action="store_true", help="Delete existing database and recreate schema.")
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Only reset SQLite DB. Do not delete data/work, data/staging, data/review, logs.",
    )
    return parser.parse_args()


def remove_sqlite_files(db_path: Path) -> None:
    candidates = [
        db_path,
        Path(str(db_path) + "-wal"),
        Path(str(db_path) + "-shm"),
        Path(str(db_path) + "-journal"),
    ]
    for path in candidates:
        if path.exists():
            path.unlink()


def ensure_dirs() -> None:
    for d in DEFAULT_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)


def vacuum_and_check(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA integrity_check;")
    result = cur.fetchone()[0]
    if result != "ok":
        raise RuntimeError(f"SQLite integrity_check failed: {result}")


def list_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    ).fetchall()
    return [r[0] for r in rows]


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)

    ensure_dirs()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reset:
        remove_sqlite_files(db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executescript(INDEX_SQL)
        conn.commit()
        vacuum_and_check(conn)
        tables = list_tables(conn)
    finally:
        conn.close()

    print(f"SQLite initialized: {db_path}")
    print(f"reset: {args.reset}")
    print(f"tables: {len(tables)}")
    for t in tables:
        print(f"  - {t}")


if __name__ == "__main__":
    main()
