#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn


SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS raw_document (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    doi TEXT,
    pmid TEXT,
    source_pdf_path TEXT,
    supplement_dir TEXT,
    status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_asset (
    asset_id TEXT PRIMARY KEY,
    doc_id TEXT,
    asset_type TEXT,
    page_no INTEGER,
    figure_ref TEXT,
    table_ref TEXT,
    file_path TEXT,
    metadata_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_text_block (
    block_id TEXT PRIMARY KEY,
    doc_id TEXT,
    page_no INTEGER,
    section TEXT,
    text TEXT,
    metadata_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_figure (
    figure_id TEXT PRIMARY KEY,
    doc_id TEXT,
    page_no INTEGER,
    figure_ref TEXT,
    file_path TEXT,
    caption TEXT,
    metadata_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_table (
    table_id TEXT PRIMARY KEY,
    doc_id TEXT,
    page_no INTEGER,
    table_ref TEXT,
    file_path TEXT,
    table_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS planned_tasks (
    task_id TEXT PRIMARY KEY,
    doc_id TEXT,
    asset_id TEXT,
    asset_type TEXT,
    task_type TEXT,
    agents_json TEXT,
    priority TEXT,
    reason TEXT,
    status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stg_structure_candidate (
    candidate_id TEXT PRIMARY KEY,
    doc_id TEXT,
    asset_id TEXT,
    image_path TEXT,
    image_name TEXT,
    candidate_index INTEGER,
    smiles TEXT,
    canonical_smiles TEXT,
    rdkit_valid INTEGER,
    rdkit_error TEXT,
    smiles_source TEXT,
    raw_json TEXT,
    status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    source_tool TEXT,
    raw_output TEXT,
    molecule_label TEXT,
    bbox_json TEXT,
    raw_context_json TEXT
);

CREATE TABLE IF NOT EXISTS structure_qc_result (
    candidate_id TEXT PRIMARY KEY,
    doc_id TEXT,
    asset_id TEXT,
    qc_score REAL,
    auto_decision TEXT,
    qc_flags_json TEXT,
    vlm_qc_json TEXT,
    review_xlsx_path TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stg_component_relation (
    relation_id TEXT PRIMARY KEY,
    doc_id TEXT,
    asset_id TEXT,
    candidate_id TEXT,
    compound_name TEXT,
    component_role TEXT,
    relation_type TEXT,
    evidence_text TEXT,
    figure_ref TEXT,
    confidence REAL,
    review_required INTEGER,
    raw_output TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stg_agent (
    agent_id TEXT PRIMARY KEY,
    doc_id TEXT,
    name TEXT,
    agent_type TEXT,
    canonical_smiles TEXT,
    structure_diagram_refs TEXT,
    evidence_span TEXT,
    status TEXT,
    raw_output TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stg_relation (
    stg_id TEXT PRIMARY KEY,
    relation_id TEXT,
    doc_id TEXT,
    relation_key TEXT,
    relation_type TEXT,
    relation_subtype TEXT,
    modality TEXT,
    inducer_name TEXT,
    matched_agent_name TEXT,
    mechanism_route TEXT,
    relation_basis TEXT,
    proximity_topology TEXT,
    effector_status TEXT,
    outcome_class TEXT,
    mechanism_tags TEXT,
    participants_json TEXT,
    evidence_span TEXT,
    confidence REAL,
    status TEXT,
    review_required INTEGER,
    qc_reasons TEXT,
    qc_warnings TEXT,
    record_json TEXT,
    raw_output TEXT,
    source_task_id TEXT,
    asset_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stg_relation_participant (
    participant_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    participant_key TEXT NOT NULL,
    name TEXT NOT NULL,
    canonical_name TEXT,
    entity_type TEXT,
    relation_ids_json TEXT,
    role_entries_json TEXT,
    is_effector INTEGER,
    is_primary_readout_target INTEGER,
    ids_json TEXT,
    sequence_json TEXT,
    structure_json TEXT,
    standardization_json TEXT,
    evidence_span TEXT,
    evidence_spans_json TEXT,
    confidence REAL,
    status TEXT,
    review_required INTEGER,
    qc_reasons TEXT,
    qc_warnings TEXT,
    raw_output TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doc_id, participant_key)
);

CREATE TABLE IF NOT EXISTS stg_assay (
    stg_id TEXT PRIMARY KEY,
    doc_id TEXT,
    assay_id TEXT,
    relation_id TEXT,
    relation_key TEXT,
    tk TEXT,
    inducer_name TEXT,
    target_name TEXT,
    assay_category TEXT,
    assay_platform TEXT,
    assay_type TEXT,
    system_type TEXT,
    cell_line TEXT,
    species TEXT,
    primary_metric TEXT,
    qualifier TEXT,
    primary_value TEXT,
    primary_unit TEXT,
    polarity TEXT,
    negative_reason TEXT,
    dose TEXT,
    time TEXT,
    figure_ref TEXT,
    evidence_span TEXT,
    condition_json TEXT,
    record_json TEXT,
    raw_output TEXT,
    confidence REAL,
    status TEXT,
    review_required INTEGER,
    qc_reasons TEXT,
    qc_warnings TEXT,
    source_task_id TEXT,
    asset_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS task_queue (
    task_id TEXT PRIMARY KEY,
    doc_id TEXT,
    queue_name TEXT,
    task_type TEXT,
    payload_json TEXT,
    status TEXT DEFAULT 'queued',
    priority INTEGER DEFAULT 0,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    error TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_task_queue_status
ON task_queue(queue_name, status, priority, created_at);

CREATE INDEX IF NOT EXISTS idx_task_queue_doc
ON task_queue(doc_id, task_type, status);

CREATE INDEX IF NOT EXISTS idx_raw_document_status
ON raw_document(status);

CREATE INDEX IF NOT EXISTS idx_raw_asset_doc
ON raw_asset(doc_id, asset_type, page_no);

CREATE INDEX IF NOT EXISTS idx_raw_text_doc
ON raw_text_block(doc_id, page_no, section);

CREATE INDEX IF NOT EXISTS idx_raw_figure_doc
ON raw_figure(doc_id, page_no, figure_ref);

CREATE INDEX IF NOT EXISTS idx_raw_table_doc
ON raw_table(doc_id, page_no, table_ref);

CREATE INDEX IF NOT EXISTS idx_planned_tasks_doc_type
ON planned_tasks(doc_id, task_type, status);

CREATE INDEX IF NOT EXISTS idx_stg_structure_doc_asset
ON stg_structure_candidate(doc_id, asset_id, source_tool, status);

CREATE INDEX IF NOT EXISTS idx_stg_component_doc_compound
ON stg_component_relation(doc_id, compound_name);

CREATE INDEX IF NOT EXISTS idx_stg_agent_doc_name
ON stg_agent(doc_id, name);

CREATE INDEX IF NOT EXISTS idx_stg_relation_doc_key
ON stg_relation(doc_id, relation_key);

CREATE INDEX IF NOT EXISTS idx_stg_relation_part_doc_key
ON stg_relation_participant(doc_id, participant_key);

CREATE INDEX IF NOT EXISTS idx_stg_assay_doc_relation
ON stg_assay(doc_id, relation_key);

CREATE INDEX IF NOT EXISTS idx_stg_assay_doc_inducer
ON stg_assay(doc_id, inducer_name);
"""


def main():
    conn = get_conn()
    conn.executescript(SCHEMA_SQL)
    conn.commit()

    tables = [
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]

    conn.close()

    print("Initialized pipeline database.")
    print("Tables:")
    for t in tables:
        print(" -", t)


if __name__ == "__main__":
    main()
