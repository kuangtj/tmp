#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an article-level molecule index before relation extraction.

This step solves a common failure mode in the current pipeline: individual
assets/pages are classified independently, while compound naming is global.
The script asks the LLM to reconcile aliases, compound ranges, named molecules,
targets, E3 ligases/effectors, and whether a molecule is a final IPM agent.

Outputs:
- data/staging/{doc_id}/global_molecule_index.json
- stg_agent rows with aliases_json / record_json / evidence_json populated
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn


PROMPT_TEMPLATE = """
You are building a GLOBAL molecule index for one induced-proximity medicine paper.

Return JSON only. No markdown. No commentary.

The current downstream pipeline fails when one molecule has multiple local names
across figures, captions, tables, text, and supplementary files. Your task is to
resolve article-level molecule identity before relation extraction.

Extract final IPM agents and useful named molecule aliases from the whole article
context. Prefer final PROTACs, molecular glues, HyT molecules, degraders,
engagers, biologics, oligos, or other induced-proximity agents. Do not promote
warheads, linkers, E3 ligands, target ligands, intermediates, reagents, or
building blocks to final agents unless the context explicitly says they are final
IPM agents.

Important scope rule:
- This step is ARTICLE-LEVEL molecule indexing, not experiment-level relation extraction.
- A single inducer may appear in multiple experiments with different targets,
  target domains, effectors, cell contexts, or mechanistic settings.
- Therefore target_names and effector_names here must be treated as
  article-level possible participants, not an exclusive or complete per-experiment truth.
- If the article mentions multiple plausible targets for one inducer, keep all of
  them here instead of choosing only one favorite target.

For compound ranges, expand them when the context clearly lists a small numeric
range, for example 3-9 -> 3, 4, 5, 6, 7, 8, 9 and 14a-14f -> 14a, 14b, 14c,
14d, 14e, 14f. Keep aliases that appear in the paper, such as "compound 22b",
"PROTAC 22b", "22b-treated", or a named molecule.

Output schema:
{
  "molecules": [
    {
      "canonical_name": "22b",
      "aliases": ["compound 22b", "PROTAC 22b"],
      "is_final_ipm_agent": true,
      "agent_type": "heterobifunctional|glue|HyT|antibody|oligo|small_molecule|other",
      "modality": "PROTAC|Molecular_Glue_Degrader|Hydrophobic_Tagging_Degrader|Other",
      "target_names": ["c-Met"],
      "effector_names": ["CRBN"],
      "mechanism_route": "E3_ligase_recruitment|hydrophobic_tagging|unknown|other",
      "evidence_span": "short verbatim article span",
      "confidence": 0.0,
      "review_required": true
    }
  ]
}

Quality rules:
- canonical_name must be the compact article-level name used for downstream relation extraction.
- aliases should include all visible local variants likely to appear in assay text.
- target_names and effector_names are article-level candidate participants that may
  be propagated from series-level design evidence.
- When multiple targets/effectors are discussed for one inducer across the paper,
  include all plausible names here.
- If a compound is only a ligand/linker/intermediate, include it only if useful as an alias object and set is_final_ipm_agent=false.
- evidence_span must be copied from the context.

ARTICLE CONTEXT:
{context}
""".strip()

RANGE_RE = re.compile(r"\b(\d+)([a-z]?)\s*[-–−]\s*(\d+)([a-z]?)\b", re.I)
COMPOUND_NAME_RE = re.compile(r"\b(?:compound|compounds|PROTACs?|degraders?|molecules?|derivatives?|analogs?|analogues?)\s+([0-9][0-9a-zA-Z]*(?:\s*(?:[-–−]|,|/|;|\band\b|\bor\b)\s*[0-9][0-9a-zA-Z]*)*)", re.I)
GENERIC_BAD = {
    "compound", "compounds", "protac", "protacs", "molecule", "molecules",
    "linker", "warhead", "ligand", "e3 ligand", "target ligand", "intermediate",
}


def uid(prefix: str, *parts: Any) -> str:
    return prefix + "_" + hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


def clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def jdump(x: Any) -> str:
    return json.dumps(x if x is not None else {}, ensure_ascii=False, default=str)


def norm_name(x: Any) -> str:
    s = clean(x).lower()
    s = re.sub(r"^(?:compound|cmpd|protac|molecule|degrader)\s+", "", s)
    s = re.sub(r"(?:-treated|\s+treated\s+cells?|\s+treatment)$", "", s)
    return s.strip(" .,:;()[]{}")


def is_bad_name(x: Any) -> bool:
    s = norm_name(x)
    return not s or s in GENERIC_BAD or len(s) > 100


def expand_range_token(token: str) -> List[str]:
    token = clean(token)
    m = RANGE_RE.fullmatch(token)
    if not m:
        return [token] if token else []
    start_n, start_s, end_n, end_s = m.groups()
    if start_s.lower() and end_s.lower() and start_n == end_n:
        a, b = ord(start_s.lower()), ord(end_s.lower())
        if a <= b and b - a <= 30:
            return [f"{start_n}{chr(c)}" for c in range(a, b + 1)]
    if not start_s and not end_s:
        a, b = int(start_n), int(end_n)
        if a <= b and b - a <= 100:
            return [str(i) for i in range(a, b + 1)]
    return [token]


def expand_name_expression(expr: str) -> List[str]:
    parts = re.split(r"\s*(?:,|/|;|\band\b|\bor\b)\s*", clean(expr), flags=re.I)
    out = []
    for part in parts:
        for name in expand_range_token(part):
            if name:
                out.append(name)
    return out


def heuristic_compound_names(text: str) -> List[str]:
    names = set()
    for m in COMPOUND_NAME_RE.finditer(text or ""):
        token = clean(m.group(1))
        for name in expand_name_expression(token):
            if not is_bad_name(name):
                names.add(norm_name(name))
    return sorted(names, key=lambda x: (len(x), x))


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[i:])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            continue
    return {}


def evidence_ok(evidence: str, context: str) -> bool:
    ev = clean(evidence)
    if not ev:
        return False
    if ev in context:
        return True
    ev2 = re.sub(r"\s+", " ", ev).lower()
    ctx2 = re.sub(r"\s+", " ", context).lower()
    if ev2 in ctx2:
        return True
    toks = [t.lower() for t in re.split(r"\W+", ev2) if len(t) >= 4]
    return len(toks) >= 4 and sum(1 for t in toks[:14] if t in ctx2) >= min(6, len(toks))


def load_global_context(conn, doc_id: str, max_chars: int) -> str:
    chunks: List[str] = []
    for r in conn.execute(
        """
        SELECT block_id, page_no, section, text
        FROM raw_text_block
        WHERE doc_id=?
        ORDER BY CASE WHEN page_no IS NULL THEN 999999 ELSE page_no END, block_id
        """,
        (doc_id,),
    ).fetchall():
        txt = clean(r["text"])
        if txt:
            chunks.append(f"[TEXT block_id={r['block_id']} page={r['page_no']} section={r['section']}]\n{txt}")

    for r in conn.execute(
        "SELECT figure_id,page_no,figure_ref,caption FROM raw_figure WHERE doc_id=? ORDER BY page_no,figure_id",
        (doc_id,),
    ).fetchall():
        cap = clean(r["caption"])
        if cap:
            chunks.append(f"[FIGURE figure_id={r['figure_id']} page={r['page_no']} ref={r['figure_ref']}]\n{cap}")

    for r in conn.execute(
        "SELECT table_id,page_no,table_ref,table_json FROM raw_table WHERE doc_id=? ORDER BY page_no,table_id",
        (doc_id,),
    ).fetchall():
        s = r["table_json"] or ""
        if s:
            chunks.append(f"[TABLE table_id={r['table_id']} page={r['page_no']} ref={r['table_ref']}]\n{s[:16000]}")

    text = "\n\n".join(chunks)
    if len(text) <= max_chars:
        return text

    # Keep the beginning, tables/supplementary material near the end, and enough middle context.
    head = int(max_chars * 0.45)
    tail = int(max_chars * 0.35)
    mid = max_chars - head - tail
    start = text[:head]
    middle_start = max(0, len(text) // 2 - mid // 2)
    middle = text[middle_start:middle_start + mid]
    end = text[-tail:]
    return "\n\n".join([start, middle, end])


def normalize_molecule(rec: Dict[str, Any], context: str) -> Optional[Dict[str, Any]]:
    name = clean(rec.get("canonical_name"))
    if is_bad_name(name):
        return None
    aliases = rec.get("aliases") if isinstance(rec.get("aliases"), list) else []
    aliases = [clean(x) for x in aliases if clean(x) and not is_bad_name(x)]
    aliases = sorted(set(aliases + [name, norm_name(name)]), key=lambda x: (len(x), x.lower()))
    evidence = clean(rec.get("evidence_span"))
    reasons = []
    if not evidence_ok(evidence, context):
        reasons.append("evidence_span_not_found_or_empty")
    try:
        confidence = max(0.0, min(1.0, float(rec.get("confidence", 0.0))))
    except Exception:
        confidence = 0.0
        reasons.append("invalid_confidence")
    return {
        "canonical_name": norm_name(name),
        "display_name": name,
        "aliases": aliases,
        "is_final_ipm_agent": bool(rec.get("is_final_ipm_agent", False)),
        "agent_type": clean(rec.get("agent_type")) or "small_molecule",
        "modality": clean(rec.get("modality")) or "Other",
        "target_names": [clean(x) for x in (rec.get("target_names") or []) if clean(x)],
        "effector_names": [clean(x) for x in (rec.get("effector_names") or []) if clean(x)],
        "mechanism_route": clean(rec.get("mechanism_route")) or "unknown",
        "evidence_span": evidence,
        "confidence": confidence,
        "review_required": bool(rec.get("review_required", False) or reasons or confidence < 0.75),
        "qc_reasons": sorted(set(reasons)),
        "raw_record": rec,
    }


def merge_heuristics(molecules: List[Dict[str, Any]], names: List[str]) -> List[Dict[str, Any]]:
    by_norm = {norm_name(m["canonical_name"]): m for m in molecules}
    for name in names:
        key = norm_name(name)
        if key in by_norm:
            continue
        by_norm[key] = {
            "canonical_name": key,
            "display_name": name,
            "aliases": [name, key],
            "is_final_ipm_agent": False,
            "agent_type": "small_molecule",
            "modality": "Other",
            "target_names": [],
            "effector_names": [],
            "mechanism_route": "unknown",
            "evidence_span": "",
            "confidence": 0.25,
            "review_required": True,
            "qc_reasons": ["heuristic_name_only"],
            "raw_record": {"source": "heuristic_compound_names"},
        }
    return sorted(by_norm.values(), key=lambda x: (not x.get("is_final_ipm_agent"), x["canonical_name"]))


def upsert_agent(conn, doc_id: str, mol: Dict[str, Any]) -> None:
    stg_id = uid("agent", doc_id, mol["canonical_name"])
    record = {
        "source": "build_global_molecule_index",
        "is_final_ipm_agent": mol["is_final_ipm_agent"],
        "modality": mol["modality"],
        "target_names": mol["target_names"],
        "effector_names": mol["effector_names"],
        "mechanism_route": mol["mechanism_route"],
        "qc_reasons": mol["qc_reasons"],
    }
    conn.execute(
        """
        INSERT INTO stg_agent
        (stg_id, doc_id, name, normalized_name, aliases_json, agent_type, structure_status,
         evidence_json, record_json, confidence, review_required, qc_reasons_json, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stg_id) DO UPDATE SET
            aliases_json=excluded.aliases_json,
            agent_type=excluded.agent_type,
            evidence_json=excluded.evidence_json,
            record_json=excluded.record_json,
            confidence=MAX(COALESCE(stg_agent.confidence,0), COALESCE(excluded.confidence,0)),
            review_required=CASE WHEN stg_agent.review_required=1 OR excluded.review_required=1 THEN 1 ELSE 0 END,
            qc_reasons_json=excluded.qc_reasons_json,
            status=excluded.status,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            stg_id,
            doc_id,
            mol["canonical_name"],
            norm_name(mol["canonical_name"]),
            jdump(mol["aliases"]),
            mol["agent_type"],
            "missing",
            jdump([{"evidence_text": mol["evidence_span"], "source": "global_molecule_index"}]),
            jdump(record),
            mol["confidence"],
            int(mol["review_required"]),
            jdump(mol["qc_reasons"]),
            "global_indexed" if mol["is_final_ipm_agent"] else "alias_candidate",
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "ipm-vlm"))
    ap.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    ap.add_argument("--max-context-chars", type=int, default=180000)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = get_conn()
    doc = conn.execute("SELECT doc_id FROM raw_document WHERE doc_id=?", (args.doc_id,)).fetchone()
    if not doc:
        raise SystemExit(f"doc_id not found: {args.doc_id}")

    context = load_global_context(conn, args.doc_id, args.max_context_chars)
    heuristic_names = heuristic_compound_names(context)
    client = OpenAI(base_url=args.llm_base_url, api_key=args.llm_api_key)
    raw = client.chat.completions.create(
        model=args.llm_model,
        temperature=0,
        max_tokens=args.max_tokens,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.replace("{context}", context)}],
    ).choices[0].message.content or ""

    obj = extract_json_object(raw)
    molecules = []
    for rec in obj.get("molecules", []) if isinstance(obj.get("molecules"), list) else []:
        if isinstance(rec, dict):
            mol = normalize_molecule(rec, context)
            if mol:
                molecules.append(mol)
    molecules = merge_heuristics(molecules, heuristic_names)

    out_dir = ROOT / "data" / "staging" / args.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "doc_id": args.doc_id,
        "num_molecules": len(molecules),
        "num_final_ipm_agents": sum(1 for m in molecules if m.get("is_final_ipm_agent")),
        "heuristic_names": heuristic_names,
        "molecules": molecules,
        "raw_llm_text": raw,
    }
    out_path = out_dir / "global_molecule_index.json"
    out_path.write_text(jdump(out), encoding="utf-8")

    if not args.dry_run:
        for mol in molecules:
            upsert_agent(conn, args.doc_id, mol)
        conn.commit()
    conn.close()

    print(json.dumps({
        "doc_id": args.doc_id,
        "global_molecule_index": str(out_path),
        "num_molecules": out["num_molecules"],
        "num_final_ipm_agents": out["num_final_ipm_agents"],
        "dry_run": bool(args.dry_run),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
