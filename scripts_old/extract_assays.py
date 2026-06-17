#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import uuid
import argparse
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple
import base64

from tqdm import tqdm
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn


STAGE11_TASK_TYPES = [
    "assay_table",
    "supplementary_assay_table",
]

ASSAY_CATEGORIES = {
    "binding",
    "ternary_complex",
    "ubiquitination",
    "degradation",
    "trafficking",
    "phosphorylation",
    "editing",
    "PPI",
    "immune_function",
    "toxicity",
    "other",
}

ASSAY_PLATFORMS = {
    "biochemical",
    "cell_based",
    "in_vivo",
    "ex_vivo",
    "computational",
    "other",
}

PRIMARY_METRICS = {
    "Kd",
    "Ki",
    "IC50",
    "EC50",
    "DC50",
    "Dmax",
    "Fold_Change",
    "Percent_Effect",
    "Half_Life",
    "Other",
}

QUALIFIERS = {"=", "<", ">", "~"}

POLARITIES = {
    "positive",
    "negative",
    "inconclusive",
}

NEGATIVE_REASONS = {
    "no_binding",
    "no_ternary",
    "no_effect",
    "weak_effect",
    "toxicity_confounded",
    "failed_delivery",
    "failed_expression",
    "other",
}


def uid(prefix: str, *parts: Any) -> str:
    return prefix + "_" + uuid.uuid5(
        uuid.NAMESPACE_URL,
        "|".join(str(x) for x in parts),
    ).hex[:16]


def clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def table_cols(conn, table: str) -> set:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def ensure_column(conn, table: str, col: str, typ: str) -> None:
    if col not in table_cols(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def require_cols(conn, table: str, cols: List[str]) -> None:
    actual = table_cols(conn, table)
    missing = [c for c in cols if c not in actual]
    if missing:
        raise RuntimeError(f"{table} missing columns: {missing}; actual={sorted(actual)}")


def ensure_tables(conn) -> None:
    conn.execute("""
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
    )
    """)

    needed = {
        "stg_id": "TEXT",
        "doc_id": "TEXT",
        "assay_id": "TEXT",
        "relation_id": "TEXT",
        "relation_key": "TEXT",
        "tk": "TEXT",

        "inducer_name": "TEXT",
        "target_name": "TEXT",

        "assay_category": "TEXT",
        "assay_platform": "TEXT",
        "assay_type": "TEXT",
        "system_type": "TEXT",
        "cell_line": "TEXT",
        "species": "TEXT",

        "primary_metric": "TEXT",
        "qualifier": "TEXT",
        "primary_value": "TEXT",
        "primary_unit": "TEXT",
        "polarity": "TEXT",
        "negative_reason": "TEXT",

        "dose": "TEXT",
        "time": "TEXT",
        "figure_ref": "TEXT",
        "evidence_span": "TEXT",

        "condition_json": "TEXT",
        "record_json": "TEXT",
        "raw_output": "TEXT",

        "confidence": "REAL",
        "status": "TEXT",
        "review_required": "INTEGER",
        "qc_reasons": "TEXT",
        "qc_warnings": "TEXT",

        "source_task_id": "TEXT",
        "asset_id": "TEXT",
        "created_at": "TEXT",
    }

    for col, typ in needed.items():
        ensure_column(conn, "stg_assay", col, typ)

    conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_stg_assay_doc_relation
    ON stg_assay(doc_id, relation_key)
    """)

    conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_stg_assay_doc_inducer
    ON stg_assay(doc_id, inducer_name)
    """)

    conn.commit()


def preflight(conn) -> None:
    require_cols(conn, "planned_tasks", [
        "task_id", "doc_id", "asset_id", "asset_type",
        "task_type", "agents_json", "priority", "reason", "status",
    ])

    require_cols(conn, "raw_asset", [
        "asset_id", "doc_id", "asset_type", "page_no",
        "figure_ref", "table_ref", "file_path", "metadata_json",
    ])

    require_cols(conn, "raw_text_block", [
        "block_id", "doc_id", "page_no", "section", "text",
    ])

    require_cols(conn, "raw_figure", [
        "figure_id", "doc_id", "page_no", "figure_ref", "caption",
    ])

    require_cols(conn, "raw_table", [
        "table_id", "doc_id", "page_no", "table_ref", "table_json",
    ])

    require_cols(conn, "stg_relation", [
        "relation_id", "doc_id", "relation_key", "inducer_name",
        "modality", "mechanism_route", "participants_json",
    ])


def norm_name(x: Any) -> str:
    x = clean(x).lower()
    x = re.sub(r"^(compound|cpd\.?|protac)\s+", "", x)
    x = re.sub(r"-treated\b.*$", "", x)
    x = re.sub(r"\btreated\b.*$", "", x)
    x = re.sub(r"\btreatment\b.*$", "", x)
    x = re.sub(r"\([^)]*\)", "", x)
    x = re.sub(r"\s+", "", x)
    return x


def normalize_compound_name(x: str) -> str:
    x = clean(x)
    x = re.sub(r"(?i)^compound\s+", "", x).strip()
    x = re.sub(r"(?i)^cpd\.?\s+", "", x).strip()
    x = re.sub(r"(?i)^protac\s+", "", x).strip()
    x = re.sub(r"(?i)-treated\b.*$", "", x).strip()
    x = re.sub(r"(?i)\btreated\b.*$", "", x).strip()
    x = re.sub(r"(?i)\btreatment\b.*$", "", x).strip()
    x = re.sub(r"\([^)]*\)", "", x).strip()
    return x


def load_relations(conn, doc_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows = conn.execute("""
    SELECT
        relation_id,
        relation_key,
        inducer_name,
        modality,
        mechanism_route,
        relation_basis,
        outcome_class,
        mechanism_tags,
        participants_json,
        status,
        review_required
    FROM stg_relation
    WHERE doc_id=?
      AND COALESCE(relation_key, '') != ''
    ORDER BY inducer_name, relation_key
    """, (doc_id,)).fetchall()

    rels = []
    by_key = {}

    target_roles = {
        "primary_target",
        "degradation_target",
        "stabilization_target",
        "regulated_target",
        "immune_target_antigen",
        "cargo",
        "substrate",
    }

    for r in rows:
        d = dict(r)
        try:
            parts = json.loads(d.get("participants_json") or "[]")
        except Exception:
            parts = []

        d["participants"] = parts
        d["target_names"] = [
            clean(p.get("name"))
            for p in parts
            if clean(p.get("name")) and clean(p.get("participant_role")) in target_roles
        ]
        d["participant_names"] = [
            clean(p.get("name"))
            for p in parts
            if clean(p.get("name"))
        ]
        rels.append(d)
        by_key[d["relation_key"]] = d

    return rels, by_key


def relation_prompt_payload(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in relations:
        out.append({
            "relation_id": r["relation_id"],
            "relation_key": r["relation_key"],
            "inducer_name": r["inducer_name"],
            "modality": r["modality"],
            "mechanism_route": r["mechanism_route"],
            "relation_basis": r.get("relation_basis", ""),
            "target_names": r.get("target_names", []),
            "participants": [
                {
                    "name": clean(p.get("name")),
                    "participant_role": clean(p.get("participant_role")),
                    "functional_role": clean(p.get("functional_role")),
                }
                for p in r.get("participants", [])
                if clean(p.get("name"))
            ],
        })
    return out


def load_tasks(conn, doc_id: str, task_types: List[str], limit: int = 0):
    marks = ",".join(["?"] * len(task_types))
    params = [doc_id] + task_types

    sql = f"""
    SELECT
        p.task_id,
        p.doc_id,
        p.asset_id,
        p.asset_type AS task_asset_type,
        p.task_type,
        p.agents_json,
        p.priority,
        p.reason,
        p.status,

        a.asset_type,
        a.page_no,
        a.figure_ref,
        a.table_ref,
        a.file_path,
        a.metadata_json

    FROM planned_tasks p
    JOIN raw_asset a
      ON p.asset_id = a.asset_id

    WHERE p.doc_id=?
      AND p.task_type IN ({marks})
      AND p.status='planned'

    ORDER BY
        CASE p.priority
            WHEN 'high' THEN 1
            WHEN 'medium' THEN 2
            ELSE 3
        END,
        COALESCE(a.page_no, 999999),
        p.task_type,
        p.task_id
    """

    if limit:
        sql += f" LIMIT {int(limit)}"

    return conn.execute(sql, params).fetchall()


def load_page_text(conn, doc_id: str, page_no: int) -> str:
    if page_no is None:
        return ""

    rows = conn.execute("""
    SELECT section, text
    FROM raw_text_block
    WHERE doc_id=?
      AND page_no=?
      AND section IN ('Text', 'Section-header', 'Title')
    ORDER BY rowid
    """, (doc_id, page_no)).fetchall()

    parts = []
    for r in rows:
        section = clean(r["section"])
        text = clean(r["text"])
        if text:
            parts.append(f"[{section}]\n{text}" if section else text)

    return "\n\n".join(parts)


def load_figures_for_asset(conn, doc_id: str, page_no: int, figure_ref: str):
    if page_no is None:
        return []

    if figure_ref:
        rows = conn.execute("""
        SELECT figure_id, figure_ref, caption
        FROM raw_figure
        WHERE doc_id=?
          AND page_no=?
          AND figure_ref=?
        """, (doc_id, page_no, figure_ref)).fetchall()
    else:
        rows = conn.execute("""
        SELECT figure_id, figure_ref, caption
        FROM raw_figure
        WHERE doc_id=?
          AND page_no=?
        """, (doc_id, page_no)).fetchall()

    return [dict(r) for r in rows]


def load_tables_for_asset(conn, doc_id: str, page_no: int, table_ref: str):
    if table_ref and page_no is not None:
        rows = conn.execute("""
        SELECT table_id, page_no, table_ref, table_json
        FROM raw_table
        WHERE doc_id=?
          AND page_no=?
          AND table_ref=?
        """, (doc_id, page_no, table_ref)).fetchall()
    elif table_ref:
        rows = conn.execute("""
        SELECT table_id, page_no, table_ref, table_json
        FROM raw_table
        WHERE doc_id=?
          AND table_ref=?
        """, (doc_id, table_ref)).fetchall()
    elif page_no is not None:
        rows = conn.execute("""
        SELECT table_id, page_no, table_ref, table_json
        FROM raw_table
        WHERE doc_id=?
          AND page_no=?
        """, (doc_id, page_no)).fetchall()
    else:
        rows = []

    return [dict(r) for r in rows]


def build_task_context(conn, task, max_chars: int) -> str:
    doc_id = task["doc_id"]
    page_no = task["page_no"]

    obj = {
        "planned_task": {
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "priority": task["priority"],
            "reason": task["reason"],
            "asset_id": task["asset_id"],
            "asset_type": task["asset_type"],
        },
        "asset": {
            "page_no": page_no,
            "figure_ref": task["figure_ref"],
            "table_ref": task["table_ref"],
            "file_path": task["file_path"],
            "metadata_json": task["metadata_json"],
        },
        "figures": load_figures_for_asset(conn, doc_id, page_no, task["figure_ref"]),
        "tables": load_tables_for_asset(conn, doc_id, page_no, task["table_ref"]),
        "page_text": load_page_text(conn, doc_id, page_no),
    }

    return jdump(obj)[:max_chars]


IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
}

TEXT_SUPP_SUFFIXES = {
    ".txt", ".csv", ".tsv", ".json", ".jsonl", ".md"
}


def parse_json_maybe(x: Any) -> Dict[str, Any]:
    if not x:
        return {}
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x)
    except Exception:
        return {}


def resolve_path(path: Any) -> str:
    path = clean(path)
    if not path:
        return ""

    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p

    return str(p)


def is_existing_image(path: Any) -> bool:
    path = resolve_path(path)
    if not path:
        return False
    p = Path(path)
    return p.exists() and p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES


def is_existing_text_supp(path: Any) -> bool:
    path = resolve_path(path)
    if not path:
        return False
    p = Path(path)
    return p.exists() and p.is_file() and p.suffix.lower() in TEXT_SUPP_SUFFIXES


def image_to_data_url(path: str) -> str:
    path = resolve_path(path)
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def collect_image_paths_from_metadata(metadata_json: Any) -> List[str]:
    meta = parse_json_maybe(metadata_json)
    paths = []

    candidate_keys = [
        "image_path",
        "page_image",
        "asset_image",
        "crop_path",
        "crop_image",
        "figure_image",
        "table_image",
        "overlay_path",
        "file_path",
    ]

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in candidate_keys and isinstance(v, str):
                    if is_existing_image(v):
                        paths.append(resolve_path(v))
                else:
                    walk(v)
        elif isinstance(x, list):
            for i in x:
                walk(i)

    walk(meta)

    return paths


def collect_task_image_paths(task: Dict[str, Any]) -> List[str]:
    paths = []

    fp = task.get("file_path")
    if is_existing_image(fp):
        paths.append(resolve_path(fp))

    paths.extend(collect_image_paths_from_metadata(task.get("metadata_json")))

    out = []
    seen = set()
    for x in paths:
        x = resolve_path(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    return out


def read_text_supplement_preview(path: str, max_chars: int = 30000) -> str:
    path = resolve_path(path)
    p = Path(path)

    if not p.exists() or not p.is_file():
        return ""

    if p.suffix.lower() not in TEXT_SUPP_SUFFIXES:
        return ""

    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = p.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        return ""

    return text[:max_chars]


def dedupe_keep_order(xs: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in xs:
        x = resolve_path(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def truncate_text(x: Any, n: int) -> str:
    x = str(x or "")
    return x[:n] + ("..." if len(x) > n else "")


def build_integrated_assay_context(
    conn,
    doc_id: str,
    task_types: List[str],
    max_chars: int = 160000,
    limit: int = 0,
    max_images: int = 16,
) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """
    Build one integrated Stage 11 assay evidence packet.

    This version includes:
    - text blocks
    - figure captions
    - table_json
    - supplementary text previews for csv/tsv/txt/json/jsonl/md
    - real image paths from raw_asset.file_path and metadata_json

    The image paths are later sent to the VLM as image_url inputs.
    """

    tasks = load_tasks(conn, doc_id, task_types, limit=limit)

    source_tasks = []
    image_paths = []

    for t in tasks:
        task_images = collect_task_image_paths(dict(t))
        image_paths.extend(task_images)

        source_tasks.append({
            "task_id": t["task_id"],
            "asset_id": t["asset_id"],
            "task_type": t["task_type"],
            "priority": t["priority"],
            "reason": t["reason"],
            "asset_type": t["asset_type"],
            "page_no": t["page_no"],
            "figure_ref": t["figure_ref"],
            "table_ref": t["table_ref"],
            "file_path": t["file_path"],
            "image_paths": task_images,
        })

    blocks = []
    total = 0

    def add_block(title: str, payload: Any, max_payload_chars: int = 20000) -> None:
        nonlocal total

        if isinstance(payload, str):
            body = truncate_text(payload, max_payload_chars)
        else:
            body = truncate_text(jdump(payload), max_payload_chars)

        if not clean(body):
            return

        block = f"\n\n## {title}\n{body}"

        if total + len(block) > max_chars:
            return

        blocks.append(block)
        total += len(block)

    add_block("SOURCE_TASKS", source_tasks, max_payload_chars=30000)

    seen_pages = set()
    seen_figures = set()
    seen_tables = set()
    seen_supp_files = set()

    tasks_sorted = sorted(
        tasks,
        key=lambda x: (
            999999 if x["page_no"] is None else x["page_no"],
            x["task_type"] or "",
            x["task_id"] or "",
        )
    )

    for task in tasks_sorted:
        page_no = task["page_no"]

        # Page text around assay assets.
        if page_no is not None and page_no not in seen_pages:
            seen_pages.add(page_no)
            page_text = load_page_text(conn, doc_id, page_no)
            add_block(
                f"PAGE_TEXT page_no={page_no}",
                page_text,
                max_payload_chars=18000,
            )

        # Figure caption/context. The actual image is sent separately via image_url
        # if raw_asset.file_path or metadata_json contains image path.
        figures = load_figures_for_asset(
            conn,
            doc_id,
            page_no,
            task["figure_ref"],
        )
        for fig in figures:
            fig_key = (
                fig.get("figure_id"),
                fig.get("figure_ref"),
                fig.get("caption"),
            )
            if fig_key in seen_figures:
                continue
            seen_figures.add(fig_key)

            add_block(
                f"FIGURE_CONTEXT page_no={page_no} figure_ref={fig.get('figure_ref')}",
                fig,
                max_payload_chars=12000,
            )

        # Table JSON is the main structured source.
        tables = load_tables_for_asset(
            conn,
            doc_id,
            page_no,
            task["table_ref"],
        )
        for tab in tables:
            tab_key = (
                tab.get("table_id"),
                tab.get("page_no"),
                tab.get("table_ref"),
            )
            if tab_key in seen_tables:
                continue
            seen_tables.add(tab_key)

            add_block(
                f"TABLE_CONTEXT page_no={tab.get('page_no')} table_ref={tab.get('table_ref')}",
                tab,
                max_payload_chars=50000,
            )

        # Supplementary files:
        # - csv/tsv/txt/json are read as text preview
        # - images are sent as image_url
        # - PDFs must already be parsed/rendered elsewhere
        fp = resolve_path(task["file_path"])
        if fp and fp not in seen_supp_files:
            seen_supp_files.add(fp)

            if is_existing_text_supp(fp):
                preview = read_text_supplement_preview(fp, max_chars=40000)
                add_block(
                    f"SUPPLEMENTARY_TEXT_FILE asset_id={task['asset_id']} path={fp}",
                    preview,
                    max_payload_chars=40000,
                )

            elif is_existing_image(fp):
                image_paths.append(fp)
                add_block(
                    f"SUPPLEMENTARY_IMAGE_FILE asset_id={task['asset_id']}",
                    {
                        "path": fp,
                        "note": "This image is also provided as multimodal image input.",
                    },
                    max_payload_chars=3000,
                )

            elif fp.lower().endswith(".pdf"):
                add_block(
                    f"SUPPLEMENTARY_PDF_NOT_INLINE asset_id={task['asset_id']}",
                    {
                        "path": fp,
                        "note": (
                            "PDF files are not directly sent to the VLM in this script. "
                            "Parse or render supplementary PDF pages into raw_text_block/raw_table/raw_asset images first."
                        ),
                    },
                    max_payload_chars=3000,
                )

    image_paths = dedupe_keep_order(image_paths)[:max_images]

    add_block(
        "MULTIMODAL_IMAGE_INPUTS",
        [
            {"image_index": i + 1, "path": path}
            for i, path in enumerate(image_paths)
        ],
        max_payload_chars=12000,
    )

    return "".join(blocks).strip(), source_tasks, image_paths

def build_prompt(context: str, relations: List[Dict[str, Any]]) -> str:
    relation_payload = relation_prompt_payload(relations)

    return f"""
You are extracting Stage 11 assay measurement records from ONE integrated assay evidence context.

Return JSONL only.
No markdown. No commentary.
One JSON object per line.
If there is no assay measurement relevant to candidate relations, return nothing.

Allowed rt:
assay

Stage 11 extracts assay/effect/measurement facts and links each assay to an existing Stage 10 relation.

Candidate relations:
{json.dumps(relation_payload, ensure_ascii=False)}

Task:
Extract measured assay records from the integrated evidence context.

The context may contain:
- main text
- figure captions
- Western blot figure context
- dose-response context
- SAR tables
- assay tables
- supplementary assay tables
- supplementary text files
- multimodal image inputs listed under MULTIMODAL_IMAGE_INPUTS and attached as image_url

Core rule:
Stage 11 records measurement facts.
Do NOT judge activity as good/bad/positive/negative.
Do NOT output polarity.
Do NOT output negative_reason.
Do NOT output review_required.
Do NOT explain whether the assay is successful or failed.

Deduplication rule:
The same assay measurement may be mentioned in text, table, figure caption, and supplement.
Output only ONE assay record for the same measurement event.

Treat records as the same measurement event when they share:
- same relation / inducer
- same target
- same assay_category
- same assay_type or biologically equivalent assay type
- same system/cell line/species
- same dose and time, when stated
- same primary_metric
- same primary_value and unit

When duplicate evidence exists:
- prefer the most structured quantitative source, usually table > figure caption > main text
- keep the clearest evidence_span
- put extra source notes in condition_json.supporting_sources if useful
- do not create separate rows just because evidence appears in multiple places

Each assay record should describe ONE measured readout under ONE condition for ONE inducer and ONE linked relation.

Required linking:
- Prefer exact relation_key from Candidate relations.
- If relation_key is unclear, output inducer_name and target_name exactly; validator will link.
- Do not invent relation_key.
- Do not create new relations.

Keep only necessary fields:
- relation_key
- inducer_name
- target_name
- assay_category
- assay_platform
- assay_type
- system_type
- cell_line
- species
- dose
- time
- primary_metric
- qualifier
- primary_value
- primary_unit
- figure_ref
- evidence_span
- condition_json
- confidence

Allowed assay_category:
{sorted(ASSAY_CATEGORIES)}

Allowed assay_platform:
{sorted(ASSAY_PLATFORMS)}

Allowed primary_metric:
{sorted(PRIMARY_METRICS)}

Allowed qualifier:
{sorted(QUALIFIERS)}

Metric rules:
- Use DC50 only for DC50 values.
- Use Dmax only for Dmax values.
- Use IC50 only for IC50 values.
- Use EC50 only for EC50 values.
- Use Percent_Effect for degradation percentage, inhibition percentage, apoptosis percentage, viability percentage, or other percent effect.
- Use Fold_Change only when the context explicitly reports fold change.
- Use Other for band intensity, qualitative WB change, N.D., not determined, no obvious degradation, or non-standard metrics.

Value rules:
- primary_value should preserve the paper-native value as a compact string.
- Keep "N.D.", "not determined", ">1", "<0.1", "no obvious degradation", "weak", etc. as primary_value when those are the reported measurement.
- Use qualifier only when a numeric inequality is stated.
- If the value is "N.D." or qualitative, qualifier should be "=" and primary_unit can be empty.

Condition rules:
- Keep proteasome inhibitors, MG132, CHX, MLN4924, bortezomib, carfilzomib, 2-DG, ATP, pretreatment, washout, rescue conditions, replicates, statistics, and controls in condition_json.
- Do not use these perturbation reagents as inducer_name.
- inducer_name must be the IPM compound or final inducer.

Assay category examples:
- c-Met degradation, WB degradation, DC50, Dmax, degradation percentage -> degradation
- Kd, Ki, IC50, binding affinity, target inhibition -> binding
- ternary complex formation, AlphaLISA/TR-FRET/NanoBRET ternary assay -> ternary_complex
- ubiquitination -> ubiquitination
- cell viability, proliferation, apoptosis -> toxicity or other
- docking-only without measured value -> do not output assay

Do not extract:
- synthesis
- reaction conditions
- intermediates
- NMR
- HRMS
- yield
- chemical characterization
- pure structure-only rows without assay measurement

Schema:
{{"rt":"assay","relation_key":"","inducer_name":"22b","target_name":"c-Met","assay_category":"degradation","assay_platform":"cell_based","assay_type":"Western blot","system_type":"cell_line","cell_line":"EBC-1","species":"human","dose":"100 nM","time":"24 h","primary_metric":"DC50","qualifier":"=","primary_value":"0.59","primary_unit":"nM","figure_ref":"Figure 3","evidence_span":"","condition_json":{{}},"confidence":0.0}}

Integrated assay evidence context:
{context}
Important visual rule:
If multimodal images are attached, inspect them together with the text context. Use image evidence for WB bands, dose-response curves, figure panels, and table screenshots when the text/table_json is incomplete.
""".strip()

def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    image_paths: List[str] | None = None,
) -> str:
    image_paths = image_paths or []

    content = [{"type": "text", "text": prompt}]

    for path in image_paths:
        try:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(path)
                },
            })
        except Exception as e:
            content.append({
                "type": "text",
                "text": f"[IMAGE_LOAD_ERROR path={path} error={e}]",
            })

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    return resp.choices[0].message.content or ""


def parse_jsonl(text: str) -> List[Dict[str, Any]]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:jsonl|json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            x = json.loads(line)
            if isinstance(x, dict):
                out.append(x)
        except Exception:
            pass

    if out:
        return out

    try:
        x = json.loads(text)
        if isinstance(x, list):
            return [i for i in x if isinstance(i, dict)]
        if isinstance(x, dict):
            return [x]
    except Exception:
        pass

    return []


def evidence_ok(evidence: str, context: str) -> bool:
    ev = clean(evidence)
    if not ev or len(ev) < 8:
        return False

    ctx = clean(context)
    if ev in ctx:
        return True

    tokens = [t for t in re.split(r"\W+", ev) if len(t) >= 4]
    if len(tokens) < 4:
        return False

    ctx_lower = ctx.lower()
    hit = sum(1 for t in tokens[:16] if t.lower() in ctx_lower)
    return hit >= min(6, len(tokens))


def normalize_allowed(value: Any, allowed: set, default: str, reasons: List[str], reason: str) -> str:
    value = clean(value)
    if not value:
        return default
    if value not in allowed:
        reasons.append(reason)
        return default
    return value


def best_relation_match(
    rec: Dict[str, Any],
    relations: List[Dict[str, Any]],
    relation_by_key: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    key = clean(rec.get("relation_key"))
    if key and key in relation_by_key:
        return relation_by_key[key]

    inducer = normalize_compound_name(clean(rec.get("inducer_name")))
    target = clean(rec.get("target_name"))

    if not inducer:
        return None

    target_norm = norm_name(target)

    best = None
    best_score = 0

    for r in relations:
        score = 0

        if norm_name(r.get("inducer_name")) == norm_name(inducer):
            score += 5

        rel_targets = r.get("target_names") or []
        rel_targets_norm = {norm_name(x) for x in rel_targets}

        if target_norm and target_norm in rel_targets_norm:
            score += 4
        elif not target_norm and len(rel_targets) == 1:
            score += 1

        if score > best_score:
            best = r
            best_score = score

    if best_score >= 5:
        return best

    return None

def assay_signature(rec: Dict[str, Any]) -> str:
    """
    Stable dedup signature for one assay measurement event.

    Do NOT include evidence_span or figure_ref.
    The same measurement may be supported by text/table/figure/supplement,
    but should still become one assay row.
    """
    condition_json = rec.get("condition_json")
    if not isinstance(condition_json, dict):
        condition_json = {}

    perturbations = condition_json.get("perturbations") or condition_json.get("pretreat") or ""
    control = condition_json.get("control") or ""

    fields = [
        rec.get("relation_key", ""),
        rec.get("inducer_name", ""),
        rec.get("target_name", ""),
        rec.get("assay_category", ""),
        rec.get("assay_platform", ""),
        rec.get("assay_type", ""),
        rec.get("system_type", ""),
        rec.get("cell_line", ""),
        rec.get("species", ""),
        rec.get("dose", ""),
        rec.get("time", ""),
        rec.get("primary_metric", ""),
        rec.get("qualifier", ""),
        rec.get("primary_value", ""),
        rec.get("primary_unit", ""),
        perturbations,
        control,
    ]
    return "|".join(clean(x) for x in fields)


def validate_assay(
    rec: Dict[str, Any],
    context: str,
    relations: List[Dict[str, Any]],
    relation_by_key: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    if clean(rec.get("rt")) != "assay":
        return None

    reasons = []
    warnings = []

    rec["inducer_name"] = normalize_compound_name(rec.get("inducer_name"))
    rec["target_name"] = clean(rec.get("target_name"))

    matched = best_relation_match(rec, relations, relation_by_key)
    if matched:
        rec["relation_id"] = matched["relation_id"]
        rec["relation_key"] = matched["relation_key"]
        rec["inducer_name"] = matched["inducer_name"]

        if not rec["target_name"] and len(matched.get("target_names", [])) == 1:
            rec["target_name"] = matched["target_names"][0]
    else:
        rec["relation_id"] = ""
        rec["relation_key"] = clean(rec.get("relation_key"))
        reasons.append("relation_not_linked")

    if not rec["inducer_name"]:
        reasons.append("missing_inducer_name")

    if not rec["target_name"]:
        warnings.append("missing_target_name")

    # Enum problems are normalized but not treated as hard review blockers.
    assay_category_raw = clean(rec.get("assay_category"))
    if assay_category_raw not in ASSAY_CATEGORIES:
        if assay_category_raw:
            warnings.append("invalid_assay_category_normalized")
        rec["assay_category"] = "other"
    else:
        rec["assay_category"] = assay_category_raw

    assay_platform_raw = clean(rec.get("assay_platform"))
    if assay_platform_raw not in ASSAY_PLATFORMS:
        if assay_platform_raw:
            warnings.append("invalid_assay_platform_normalized")
        rec["assay_platform"] = "other"
    else:
        rec["assay_platform"] = assay_platform_raw

    metric_raw = clean(rec.get("primary_metric"))
    if metric_raw not in PRIMARY_METRICS:
        if metric_raw:
            warnings.append("invalid_primary_metric_normalized")
        rec["primary_metric"] = "Other"
    else:
        rec["primary_metric"] = metric_raw

    qualifier_raw = clean(rec.get("qualifier"))
    if qualifier_raw not in QUALIFIERS:
        if qualifier_raw:
            warnings.append("invalid_qualifier_normalized")
        rec["qualifier"] = "="
    else:
        rec["qualifier"] = qualifier_raw or "="

    for k in [
        "assay_type", "system_type", "cell_line", "species",
        "dose", "time", "primary_value", "primary_unit",
        "figure_ref", "evidence_span",
    ]:
        rec[k] = clean(rec.get(k))

    if not rec["primary_value"]:
        warnings.append("missing_primary_value")

    evidence = clean(rec.get("evidence_span"))
    if not evidence:
        warnings.append("missing_evidence_span")
    elif not evidence_ok(evidence, context):
        # Non-blocking only. Table serialization/OCR often breaks exact matching.
        warnings.append("evidence_weak_match")

    condition_json = rec.get("condition_json")
    if not isinstance(condition_json, dict):
        condition_json = {}
    rec["condition_json"] = condition_json

    # Stage 11 no longer judges activity.
    # Keep columns for compatibility, but do not ask LLM to output them.
    rec["polarity"] = ""
    rec["negative_reason"] = ""

    try:
        rec["confidence"] = float(rec.get("confidence", 0) or 0)
    except Exception:
        rec["confidence"] = 0.0

    rec["assay_signature"] = assay_signature(rec)
    rec["assay_id"] = uid("assay", rec["assay_signature"])

    rec["qc_reasons"] = reasons
    rec["qc_warnings"] = warnings

    # Ignore model-provided review_required.
    # Only hard linking/identity failures require review.
    rec["review_required"] = bool(reasons)

    if rec["review_required"]:
        rec["status"] = "review_required"
    elif warnings:
        rec["status"] = "auto_pass_with_warning"
    else:
        rec["status"] = "auto_pass"

    return rec

def upsert_assay(conn, doc_id: str, rec: Dict[str, Any], task) -> None:
    stg_id = uid("stg_assay", doc_id, rec["assay_signature"])

    conn.execute("""
    INSERT OR REPLACE INTO stg_assay
    (
        stg_id, doc_id, assay_id,
        relation_id, relation_key, tk,
        inducer_name, target_name,

        assay_category, assay_platform, assay_type,
        system_type, cell_line, species,

        primary_metric, qualifier, primary_value, primary_unit,
        polarity, negative_reason,

        dose, time, figure_ref, evidence_span,
        condition_json, record_json, raw_output,

        confidence, status, review_required,
        qc_reasons, qc_warnings,

        source_task_id, asset_id
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        stg_id,
        doc_id,
        rec["assay_id"],

        rec.get("relation_id", ""),
        rec.get("relation_key", ""),
        rec.get("relation_key", ""),

        rec.get("inducer_name", ""),
        rec.get("target_name", ""),

        rec.get("assay_category", ""),
        rec.get("assay_platform", ""),
        rec.get("assay_type", ""),

        rec.get("system_type", ""),
        rec.get("cell_line", ""),
        rec.get("species", ""),

        rec.get("primary_metric", ""),
        rec.get("qualifier", ""),
        rec.get("primary_value", ""),
        rec.get("primary_unit", ""),

        rec.get("polarity", ""),
        rec.get("negative_reason", ""),

        rec.get("dose", ""),
        rec.get("time", ""),
        rec.get("figure_ref", ""),
        rec.get("evidence_span", ""),

        jdump(rec.get("condition_json", {})),
        jdump(rec),
        jdump(rec),

        rec.get("confidence", 0),
        rec.get("status", ""),
        int(rec.get("review_required", False)),

        jdump(rec.get("qc_reasons", [])),
        jdump(rec.get("qc_warnings", [])),

        task["task_id"],
        task["asset_id"],
    ))


def rec_key(rec: Dict[str, Any]) -> Tuple[str, str]:
    return ("assay", rec["assay_signature"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "ipm-llm"))
    ap.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    ap.add_argument("--task-types", default=",".join(STAGE11_TASK_TYPES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-context-chars", type=int, default=640000)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--max-images", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    task_types = [x.strip() for x in args.task_types.split(",") if x.strip()]

    client = OpenAI(
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
    )

    conn = get_conn()
    ensure_tables(conn)
    preflight(conn)

    if args.overwrite:
        conn.execute("DELETE FROM stg_assay WHERE doc_id=?", (args.doc_id,))
        conn.commit()

    relations, relation_by_key = load_relations(conn, args.doc_id)

    integrated_context, source_tasks, image_paths = build_integrated_assay_context(
        conn=conn,
        doc_id=args.doc_id,
        task_types=task_types,
        max_chars=args.max_context_chars,
        limit=args.limit,
        max_images=args.max_images)

    out_dir = Path("data/staging") / args.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    context_path = out_dir / "stage11_integrated_assay_context.txt"
    raw_path = out_dir / "stage11_assays_raw_llm.jsonl"
    valid_path = out_dir / "stage11_assays_validated.jsonl"
    report_path = out_dir / "stage11_assays_report.json"

    context_path.write_text(integrated_context, encoding="utf-8")

    stats = Counter()
    seen = set()
    valid_records = []

    pseudo_task = {
        "task_id": "stage11_integrated_assay_context",
        "asset_id": "stage11_integrated_assay_context",
        "task_type": "integrated_assay_context",
    }

    if not integrated_context.strip():
        stats["empty_integrated_context"] += 1
        raw_text = ""
    else:
        prompt = build_prompt(integrated_context, relations)

        try:
            raw_text = call_llm(
                client=client,
                model=args.llm_model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                image_paths=image_paths
            )
        except Exception as e:
            stats["llm_error"] += 1
            raw_text = ""
            raw_path.write_text(jdump({
                "doc_id": args.doc_id,
                "mode": "integrated_assay_context",
                "error": str(e),
                "num_source_tasks": len(source_tasks),
                "context_chars": len(integrated_context),
                "context_path": str(context_path),
            }) + "\n", encoding="utf-8")

    with open(raw_path, "w", encoding="utf-8") as fraw, open(valid_path, "w", encoding="utf-8") as fvalid:
        fraw.write(jdump({
            "doc_id": args.doc_id,
            "mode": "integrated_assay_context",
            "num_source_tasks": len(source_tasks),
            "source_tasks": source_tasks,
            "context_chars": len(integrated_context),
            "context_path": str(context_path),
            "num_images": len(image_paths),
            "image_paths": image_paths,   
            "raw_text": raw_text,
        }) + "\n")

        for rec in parse_jsonl(raw_text):
            rec = validate_assay(rec, integrated_context, relations, relation_by_key)
            if not rec:
                stats["invalid_or_unsupported_rt"] += 1
                continue

            rec["_source_task_id"] = pseudo_task["task_id"]
            rec["_source_asset_id"] = pseudo_task["asset_id"]
            rec["_source_task_type"] = pseudo_task["task_type"]
            rec["_source_task_ids"] = [x["task_id"] for x in source_tasks]
            rec["_source_asset_ids"] = sorted({x["asset_id"] for x in source_tasks})

            key = rec_key(rec)
            if key in seen:
                stats["deduped"] += 1
                continue

            seen.add(key)
            valid_records.append(rec)
            fvalid.write(jdump(rec) + "\n")

            stats["rt:assay"] += 1
            stats[f"status:{rec.get('status', '')}"] += 1
            stats[f"category:{rec.get('assay_category', '')}"] += 1
            stats[f"metric:{rec.get('primary_metric', '')}"] += 1

            if rec.get("relation_id"):
                stats["relation_linked"] += 1
            else:
                stats["relation_not_linked"] += 1

            if rec.get("review_required"):
                stats["review_required"] += 1
            else:
                stats["auto_pass"] += 1

            if not args.dry_run:
                upsert_assay(conn, args.doc_id, rec, pseudo_task)

        if not args.dry_run:
            conn.commit()

    conn.close()

    report = {
        "doc_id": args.doc_id,
        "mode": "integrated_assay_context",
        "num_source_tasks": len(source_tasks),
        "num_candidate_relations": len(relations),
        "task_types": task_types,
        "context_chars": len(integrated_context),
        "num_images": len(image_paths),
        "image_paths": image_paths,
        "context_path": str(context_path),
        "num_valid_records": len(valid_records),
        "stats": dict(stats),
        "raw_llm_jsonl": str(raw_path),
        "validated_jsonl": str(valid_path),
        "tables": ["stg_assay"],
        "note": (
            "Stage 11 integrates related text, figures, tables, and supplementary assay contexts into one request, "
            "then extracts deduplicated assay measurement records linked to Stage 10 relation_key/relation_id."
        ),
    }

    report_path.write_text(jdump(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    
if __name__ == "__main__":
    main()