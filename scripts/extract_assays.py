#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract assay/activity/effect measurements and link them to existing stg_relation.

Output:
- stg_assay

Rules:
- Do not create new IPM relations here.
- Link assays to existing relation candidates when possible.
- Record measurement facts; do not judge good/bad design quality.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn

ASSAY_CATEGORIES = {"binding", "ternary_complex", "ubiquitination", "degradation", "trafficking", "phosphorylation", "editing", "PPI", "immune_function", "viability", "toxicity", "other"}
PRIMARY_METRICS = {"Kd", "Ki", "IC50", "EC50", "DC50", "Dmax", "Fold_Change", "Percent_Effect", "Half_Life", "Other"}
GENERIC_BAD = {"compound", "compounds", "molecule", "molecules", "protac", "protacs", "series", "hyt molecules"}
SYNTHESIS_RE = re.compile(r"\b(NMR|HRMS|yield|synthesis|synthesized|purification|LC-MS|HPLC purity)\b", re.I)


def uid(prefix: str, *parts: Any) -> str:
    return prefix + "_" + hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def jdump(x: Any) -> str:
    return json.dumps(x if x is not None else {}, ensure_ascii=False, default=str)


def clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def evidence_ok(evidence: str, context: str) -> bool:
    ev = clean(evidence)
    if len(ev) < 5:
        return False
    if ev in context:
        return True
    compact_ev = re.sub(r"\s+", " ", ev).lower()
    compact_ctx = re.sub(r"\s+", " ", context).lower()
    if compact_ev in compact_ctx:
        return True
    toks = [t.lower() for t in re.split(r"\W+", compact_ev) if len(t) >= 3]
    return len(toks) >= 3 and sum(1 for t in toks[:12] if t in compact_ctx) >= min(5, len(toks))


def is_generic_name(x: str) -> bool:
    s = clean(x).lower().strip(" .,:;()[]{}")
    return (not s) or s in GENERIC_BAD or len(s) > 80


def extract_json_records(text: str) -> List[Dict[str, Any]]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:jsonl|json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    out = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            pass
    if out:
        return out
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("assays"), list):
            return [x for x in obj["assays"] if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        return []
    return []


def load_relations(conn, doc_id: str) -> List[Dict[str, Any]]:
    rels = []
    rows = conn.execute("SELECT * FROM stg_relation WHERE doc_id=? ORDER BY created_at", (doc_id,)).fetchall()
    for r in rows:
        d = dict(r)
        parts = conn.execute("SELECT entity_name, role, entity_type FROM stg_relation_participant WHERE relation_id=? ORDER BY role", (r["relation_id"],)).fetchall()
        d["participants"] = [dict(p) for p in parts]
        rels.append(d)
    return rels


def load_assay_context(conn, doc_id: str, task_types: List[str], max_context_chars: int) -> str:
    placeholders = ",".join("?" for _ in task_types)
    asset_rows = conn.execute(
        f"""
        SELECT p.task_id, p.task_type, a.asset_id, a.asset_type, a.page_no, a.figure_ref, a.table_ref, a.file_path, a.metadata_json
        FROM planned_tasks p
        LEFT JOIN raw_asset a ON a.asset_id=p.asset_id
        WHERE p.doc_id=? AND p.task_type IN ({placeholders})
        ORDER BY p.priority DESC, a.page_no
        """,
        (doc_id, *task_types),
    ).fetchall()
    asset_ids = {r["asset_id"] for r in asset_rows if r["asset_id"]}

    chunks = []
    # Include selected table JSON/caption/image metadata.
    for r in asset_rows:
        chunks.append(f"[PLANNED_ASSET task_type={r['task_type']} asset_id={r['asset_id']} page={r['page_no']} figure_ref={r['figure_ref']} table_ref={r['table_ref']} file={r['file_path']}]\nmetadata={r['metadata_json'] or ''}")
        if r["asset_type"] == "table_image" or r["asset_type"] == "supplementary_table":
            t = conn.execute("SELECT table_id, page_no, table_ref, table_json FROM raw_table WHERE table_id=?", (r["asset_id"],)).fetchone()
            if t:
                chunks.append(f"[TABLE table_id={t['table_id']} page={t['page_no']} ref={t['table_ref']}]\n{(t['table_json'] or '')[:10000]}")
        if r["asset_type"] == "figure_image":
            f = conn.execute("SELECT figure_id, page_no, figure_ref, caption FROM raw_figure WHERE figure_id=?", (r["asset_id"],)).fetchone()
            if f:
                chunks.append(f"[FIGURE figure_id={f['figure_id']} page={f['page_no']} ref={f['figure_ref']}]\n{f['caption'] or ''}")

    # Add all captions/tables if planner produced too little.
    if len("\n".join(chunks)) < 2000:
        for t in conn.execute("SELECT table_id, page_no, table_ref, table_json FROM raw_table WHERE doc_id=?", (doc_id,)).fetchall():
            chunks.append(f"[TABLE table_id={t['table_id']} page={t['page_no']} ref={t['table_ref']}]\n{(t['table_json'] or '')[:10000]}")
        for f in conn.execute("SELECT figure_id, page_no, figure_ref, caption FROM raw_figure WHERE doc_id=?", (doc_id,)).fetchall():
            chunks.append(f"[FIGURE figure_id={f['figure_id']} page={f['page_no']} ref={f['figure_ref']}]\n{f['caption'] or ''}")

    # Also add nearby text blocks from relevant pages.
    pages = {r["page_no"] for r in asset_rows if r["page_no"] is not None}
    if pages:
        ph = ",".join("?" for _ in pages)
        rows = conn.execute(f"SELECT block_id, page_no, section, text FROM raw_text_block WHERE doc_id=? AND page_no IN ({ph}) ORDER BY page_no", (doc_id, *sorted(pages))).fetchall()
    else:
        rows = conn.execute("SELECT block_id, page_no, section, text FROM raw_text_block WHERE doc_id=? ORDER BY page_no LIMIT 200", (doc_id,)).fetchall()
    for r in rows:
        txt = clean(r["text"])
        if txt:
            chunks.append(f"[TEXT block_id={r['block_id']} page={r['page_no']} section={r['section']}]\n{txt}")

    return "\n\n".join(chunks)[:max_context_chars]


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i+chunk_size])
        i += max(1, chunk_size-overlap)
    return out


def relation_payload(rels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rels:
        out.append({
            "relation_id": r["relation_id"],
            "modality": r.get("modality", ""),
            "outcome_class": r.get("outcome_class", ""),
            "mechanism_route": r.get("mechanism_route", ""),
            "relation_name": r.get("relation_name", ""),
            "participants": r.get("participants", []),
            "evidence_text": r.get("evidence_text", "")[:300],
        })
    return out


def build_prompt(context: str, rels: List[Dict[str, Any]]) -> str:
    return f"""
You are extracting assay/activity/effect measurements for an induced proximity medicine database.

Return JSONL only. One JSON object per line. No markdown.

Hard rules:
1. Link each assay to an existing relation_id from Relation candidates whenever possible.
2. Do NOT create new relations here.
3. Do not extract synthesis, NMR, HRMS, purity, or yield information.
4. Do not judge whether the compound is good or bad. Record measurement facts only.
5. Each output should represent one measurement event. Do not duplicate the same event.
6. evidence_text must be a short verbatim span from the context.
7. Keep DC50, Dmax, IC50, EC50, Kd, Ki, Percent_Effect distinct.

Allowed assay_category:
{sorted(ASSAY_CATEGORIES)}
Allowed primary_metric:
{sorted(PRIMARY_METRICS)}

Output schema per line:
{{
  "rt":"assay",
  "relation_id":"existing relation_id or empty if uncertain",
  "agent_name":"final compound/construct tested",
  "target_name":"target/readout target",
  "assay_category":"degradation|binding|ternary_complex|...|other",
  "assay_type":"Western blot|HiBiT|TR-FRET|SPR|cell viability|...",
  "assay_format":"biochemical|cell_based|in_vivo|ex_vivo|other",
  "primary_metric":"DC50|Dmax|IC50|EC50|Kd|Ki|Percent_Effect|Fold_Change|Other",
  "primary_value":"value only",
  "primary_qualifier":"=|>|<|~|not determined|",
  "primary_unit":"nM|uM|%|fold|h|",
  "cell_line":"",
  "species":"",
  "dose":"",
  "dose_unit":"",
  "treatment_time":"",
  "treatment_time_unit":"",
  "evidence_text":"verbatim evidence span",
  "confidence":0.0,
  "review_required":true
}}

Relation candidates:
{json.dumps(relation_payload(rels), ensure_ascii=False)[:20000]}

Context:
{context}
""".strip()


def validate_assay(rec: Dict[str, Any], context: str, relation_ids: set) -> Optional[Dict[str, Any]]:
    if rec.get("rt") not in {"assay", "measurement"}:
        return None
    reasons = []
    relation_id = clean(rec.get("relation_id"))
    if relation_id and relation_id not in relation_ids:
        relation_id = ""
        reasons.append("relation_id_not_found")
    agent_name = clean(rec.get("agent_name"))
    if is_generic_name(agent_name):
        reasons.append("missing_or_generic_agent_name")
    category = clean(rec.get("assay_category")) or "other"
    if category not in ASSAY_CATEGORIES:
        category = "other"; reasons.append("invalid_assay_category")
    metric = clean(rec.get("primary_metric")) or "Other"
    if metric not in PRIMARY_METRICS:
        metric = "Other"; reasons.append("invalid_primary_metric")
    evidence = clean(rec.get("evidence_text"))
    if not evidence_ok(evidence, context):
        reasons.append("evidence_text_not_found_or_empty")
    if SYNTHESIS_RE.search(evidence):
        reasons.append("synthesis_like_evidence")
    try:
        conf = float(rec.get("confidence", 0.0))
    except Exception:
        conf = 0.0; reasons.append("invalid_confidence")
    rec.update({
        "relation_id": relation_id,
        "agent_name": agent_name,
        "target_name": clean(rec.get("target_name")),
        "assay_category": category,
        "assay_type": clean(rec.get("assay_type")),
        "assay_format": clean(rec.get("assay_format")),
        "primary_metric": metric,
        "primary_value": clean(rec.get("primary_value")),
        "primary_qualifier": clean(rec.get("primary_qualifier")),
        "primary_unit": clean(rec.get("primary_unit")),
        "cell_line": clean(rec.get("cell_line")),
        "species": clean(rec.get("species")),
        "dose": clean(rec.get("dose")),
        "dose_unit": clean(rec.get("dose_unit")),
        "treatment_time": clean(rec.get("treatment_time")),
        "treatment_time_unit": clean(rec.get("treatment_time_unit")),
        "evidence_text": evidence,
        "confidence": conf,
        "qc_reasons": reasons,
        "review_required": bool(rec.get("review_required", False) or reasons or conf < 0.75),
    })
    return rec


def find_agent_stg_id(conn, doc_id: str, name: str) -> str:
    if not name:
        return ""
    row = conn.execute("SELECT stg_id FROM stg_agent WHERE doc_id=? AND lower(name)=lower(?) LIMIT 1", (doc_id, name)).fetchone()
    return row["stg_id"] if row else ""


def insert_assay(conn, doc_id: str, rec: Dict[str, Any]) -> None:
    agent_stg_id = find_agent_stg_id(conn, doc_id, rec.get("agent_name", ""))
    sig = "|".join([doc_id, rec.get("relation_id", ""), rec.get("agent_name", ""), rec.get("target_name", ""), rec.get("assay_category", ""), rec.get("assay_type", ""), rec.get("primary_metric", ""), rec.get("primary_value", ""), rec.get("primary_unit", ""), rec.get("evidence_text", "")[:120]])
    assay_id = uid("assay", sig)
    conn.execute(
        """
        INSERT OR REPLACE INTO stg_assay
        (assay_id, doc_id, relation_id, agent_stg_id, agent_name, target_name, assay_category,
         assay_type, assay_format, primary_metric, primary_value, primary_qualifier, primary_unit,
         cell_line, species, dose, dose_unit, treatment_time, treatment_time_unit,
         evidence_text, record_json, raw_output, confidence, review_required, qc_reasons_json, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (assay_id, doc_id, rec.get("relation_id", ""), agent_stg_id, rec.get("agent_name", ""), rec.get("target_name", ""), rec.get("assay_category", "other"), rec.get("assay_type", ""), rec.get("assay_format", ""), rec.get("primary_metric", "Other"), rec.get("primary_value", ""), rec.get("primary_qualifier", ""), rec.get("primary_unit", ""), rec.get("cell_line", ""), rec.get("species", ""), rec.get("dose", ""), rec.get("dose_unit", ""), rec.get("treatment_time", ""), rec.get("treatment_time_unit", ""), rec.get("evidence_text", ""), jdump(rec), jdump(rec), rec.get("confidence", 0.0), int(rec.get("review_required", True)), jdump(rec.get("qc_reasons", [])), "pending_qc"),
    )


def call_llm(client: OpenAI, model: str, prompt: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(model=model, temperature=0, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "ipm-llm"))
    ap.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    ap.add_argument("--task-types", default="assay_table,supplementary_assay_table,western_blot_figure,dose_response_curve")
    ap.add_argument("--max-context-chars", type=int, default=120000)
    ap.add_argument("--chunk-size", type=int, default=16000)
    ap.add_argument("--overlap", type=int, default=1200)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-images", type=int, default=0)  # reserved; this version is text/table-driven
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = get_conn()
    rels = load_relations(conn, args.doc_id)
    if not rels:
        raise SystemExit("No stg_relation records found. Run extract_ipm_knowledge.py first.")
    relation_ids = {r["relation_id"] for r in rels}
    if args.overwrite:
        conn.execute("DELETE FROM stg_assay WHERE doc_id=?", (args.doc_id,))
        conn.commit()

    task_types = [x.strip() for x in args.task_types.split(",") if x.strip()]
    context = load_assay_context(conn, args.doc_id, task_types, args.max_context_chars)
    chunks = chunk_text(context, args.chunk_size, args.overlap)
    if args.limit:
        chunks = chunks[:args.limit]

    out_dir = ROOT / "data" / "staging" / args.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "assays_raw_llm.jsonl"
    val_path = out_dir / "assays_validated.jsonl"
    report_path = out_dir / "assay_extraction_report.json"

    client = OpenAI(base_url=args.llm_base_url, api_key=args.llm_api_key)
    stats = Counter()
    seen = set()

    with raw_path.open("w", encoding="utf-8") as fraw, val_path.open("w", encoding="utf-8") as fval:
        for i, ctx in enumerate(tqdm(chunks, desc="Extract assays")):
            prompt = build_prompt(ctx, rels)
            try:
                raw = call_llm(client, args.llm_model, prompt, args.max_tokens)
            except Exception as e:
                stats["llm_error"] += 1
                fraw.write(jdump({"chunk": i, "error": str(e)}) + "\n")
                continue
            fraw.write(jdump({"chunk": i, "raw_text": raw}) + "\n")
            for rec in extract_json_records(raw):
                stats["raw_records"] += 1
                rec = validate_assay(rec, ctx, relation_ids)
                if not rec:
                    stats["invalid_record"] += 1
                    continue
                key = (rec.get("relation_id"), rec.get("agent_name"), rec.get("target_name"), rec.get("assay_type"), rec.get("primary_metric"), rec.get("primary_value"), rec.get("evidence_text", "")[:80])
                if key in seen:
                    stats["deduped"] += 1
                    continue
                seen.add(key)
                fval.write(jdump(rec) + "\n")
                stats["assays"] += 1
                if rec.get("review_required"):
                    stats["review_required"] += 1
                else:
                    stats["accepted"] += 1
                if not args.dry_run:
                    insert_assay(conn, args.doc_id, rec)
                    conn.commit()

    report = {"doc_id": args.doc_id, "num_chunks": len(chunks), "stats": dict(stats), "raw_llm_jsonl": str(raw_path), "validated_jsonl": str(val_path), "tables": ["stg_assay"]}
    report_path.write_text(jdump(report), encoding="utf-8")
    conn.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
