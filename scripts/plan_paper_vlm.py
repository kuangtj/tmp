#!/usr/bin/env python3
import os
import re
import sys
import json
import uuid
import base64
import argparse
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn
from ipm_eagle.db.local_queue import enqueue


TASK_TYPES = [
    "structure_figure",
    "sar_table",
    "assay_table",
    "mechanism_figure",
    "western_blot_figure",
    "dose_response_curve",
    "text_evidence",
    "supplementary_structure_table",
    "supplementary_assay_table",
    "mixed_or_uncertain",
    "irrelevant",
]


AGENTS = {
    "structure_figure": ["SupplementStructureAgent", "TargetedStructureResolutionAgent"],
    "sar_table": ["SupplementStructureAgent", "TargetedStructureResolutionAgent"],
    "supplementary_structure_table": ["SupplementStructureAgent", "TargetedStructureResolutionAgent"],

    "assay_table": ["AssayExtractionAgent", "IPMKnowledgeExtractionAgent"],
    "supplementary_assay_table": ["AssayExtractionAgent", "IPMKnowledgeExtractionAgent"],
    "western_blot_figure": ["AssayExtractionAgent", "IPMKnowledgeExtractionAgent"],
    "dose_response_curve": ["AssayExtractionAgent", "IPMKnowledgeExtractionAgent"],

    "mechanism_figure": ["IPMKnowledgeExtractionAgent"],
    "text_evidence": ["IPMKnowledgeExtractionAgent", "AssayAgent"],
    "mixed_or_uncertain": ["PlannerReviewAgent"],
    "irrelevant": [],
}

PRIORITY_SCORE = {"high": 90, "medium": 50, "low": 10}


def uid(prefix, *parts):
    return prefix + "_" + uuid.uuid5(uuid.NAMESPACE_URL, "|".join(map(str, parts))).hex[:16]


def load_json(s):
    try:
        return json.loads(s or "{}")
    except Exception:
        return {}


def image_to_data_url(path):
    p = Path(path)
    if not p.exists():
        return None

    suffix = p.suffix.lower()
    mime = "image/png"
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"

    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def short_text(s, n=5000):
    s = s or ""
    return s[:n]


def read_text_file(path, n=4000):
    p = Path(path)
    if not p.exists():
        return ""
    if p.suffix.lower() in [".txt", ".md", ".json", ".csv"]:
        return p.read_text(encoding="utf-8", errors="ignore")[:n]
    return ""


def get_doc_context(conn, doc_id):
    rows = conn.execute(
        """
        SELECT section, text
        FROM raw_text_block
        WHERE doc_id=?
          AND section IN ('title', 'abstract')
        """,
        (doc_id,),
    ).fetchall()

    ctx = {"title": "", "abstract": ""}
    for r in rows:
        ctx[r["section"]] = r["text"] or ""

    return ctx


def get_asset_context(conn, asset):
    asset_id = asset["asset_id"]
    asset_type = asset["asset_type"] or ""
    file_path = asset["file_path"] or ""
    meta = load_json(asset["metadata_json"])

    parts = [
        f"asset_id: {asset_id}",
        f"asset_type: {asset_type}",
        f"page_no: {asset['page_no']}",
        f"figure_ref: {asset['figure_ref'] or ''}",
        f"table_ref: {asset['table_ref'] or ''}",
        f"file_path: {file_path}",
        f"metadata: {json.dumps(meta, ensure_ascii=False)[:2000]}",
    ]

    fig = conn.execute(
        "SELECT figure_ref, caption, metadata_json FROM raw_figure WHERE figure_id=?",
        (asset_id,),
    ).fetchone()
    if fig:
        parts += [
            f"figure_ref_db: {fig['figure_ref'] or ''}",
            f"figure_caption: {fig['caption'] or ''}",
            f"figure_metadata: {(fig['metadata_json'] or '')[:2000]}",
        ]

    tab = conn.execute(
        "SELECT table_ref, file_path, table_json FROM raw_table WHERE table_id=?",
        (asset_id,),
    ).fetchone()
    if tab:
        parts += [
            f"table_ref_db: {tab['table_ref'] or ''}",
            f"table_json: {(tab['table_json'] or '')[:3000]}",
            f"table_file_preview: {read_text_file(tab['file_path'] or '', 3000)}",
        ]

    return "\n".join(parts)


def extract_json(text):
    """
    Robust JSON parser for VLM output.
    Never raises. If JSON is truncated or invalid, recover key fields.
    """
    raw = text or ""
    text = raw.strip()

    # remove markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # parse first complete JSON object
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    # try simple brace repair
    if "{" in text:
        frag = text[text.find("{"):]
        need_close = frag.count("{") - frag.count("}")
        if need_close > 0:
            repaired = frag + ("}" * need_close)
            try:
                return json.loads(repaired)
            except Exception:
                pass

    # regex recovery for truncated JSON
    task_type = "mixed_or_uncertain"
    for t in TASK_TYPES:
        if re.search(rf'"task_type"\s*:\s*"{re.escape(t)}"', raw):
            task_type = t
            break
        if t in raw:
            task_type = t
            break

    priority = "medium"
    m = re.search(r'"priority"\s*:\s*"(high|medium|low)', raw, re.I)
    if m:
        priority = m.group(1).lower()

    reason = ""
    m = re.search(r'"reason"\s*:\s*"([^"]*)', raw, re.S)
    if m:
        reason = " ".join(m.group(1).split())[:300]

    if not reason:
        reason = "VLM output JSON was invalid or truncated; recovered fields when possible."

    return {
        "task_type": task_type,
        "priority": priority,
        "reason": reason + " [json_recovered]",
        "raw_invalid_json": raw[:1000],
    }
    

def vlm_classify(client, model, asset, doc_ctx, asset_ctx,max_tokens=2048):
    file_path = asset["file_path"] or ""
    data_url = None

    if Path(file_path).suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
        data_url = image_to_data_url(file_path)

    prompt = f"""
You are a Paper Planner for an induced proximity medicine database pipeline.

Classify this asset into exactly ONE task_type.

This planner is not used to run full-document structure recognition immediately.
It creates candidate evidence/page pools for later IPM knowledge extraction, assay extraction, and missing-agent structure/sequence resolution.

Allowed task_type:
{json.dumps(TASK_TYPES, ensure_ascii=False)}

Definitions:
- structure_figure: main-paper chemical structure drawings, molecule panels, compound structures, PROTAC/MG/HyT structures, linker/warhead/E3 ligand/scaffold/R-group structures. These are candidate pages for later missing-agent structure resolution.
- sar_table: main-paper table with compound names, R-groups, chemical structures, SAR, activity of compound series.
- assay_table: main-paper table with biological assay data, DC50, Dmax, IC50, EC50, Kd, Ki, degradation, viability, binding, cell line, dose/time.
- mechanism_figure: conceptual mechanism/pathway/proximity/ternary complex/UPS/lysosome/autophagy schematic.
- western_blot_figure: Western blot/immunoblot/protein band figure.
- dose_response_curve: dose-response plot, degradation curve, viability curve, IC50/DC50/Dmax plot.
- text_evidence: page image or text asset useful as general evidence, but not primarily structure/table/assay figure.
- supplementary_structure_table: supplementary table/file with compound names and complete structures, SMILES, InChI, SDF/MolBlock, R-groups, SAR, or structure images. This has highest priority for direct structure extraction.
- supplementary_assay_table: supplementary table with assay measurements.
- mixed_or_uncertain: useful but ambiguous, multiple roles, or insufficient information.
- irrelevant: not useful for downstream extraction.

Rules:
1. Do not miss text_evidence, assay_table, supplementary_assay_table, structure_figure, or supplementary_structure_table.
2. Supplementary tables should be high priority if useful.
3. If uncertain but potentially useful, use mixed_or_uncertain, not irrelevant.
4. Return JSON only.

Output schema:
{{
  "task_type": "...",
  "priority": "high|medium|low",
  "reason": "brief reason"
}}

Paper context:
Title: {short_text(doc_ctx.get("title", ""), 1000)}
Abstract: {short_text(doc_ctx.get("abstract", ""), 2000)}

Asset context:
{short_text(asset_ctx, 6000)}
""".strip()

    content = [{"type": "text", "text": prompt}]
    if data_url:
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=max_tokens
    )

    raw = resp.choices[0].message.content
    obj = extract_json(raw)

    task_type = obj.get("task_type", "mixed_or_uncertain")
    if task_type not in TASK_TYPES:
        task_type = "mixed_or_uncertain"

    priority = obj.get("priority", "medium")
    if priority not in ["high", "medium", "low"]:
        priority = "medium"

    return {
        "task_type": task_type,
        "priority": priority,
        "reason": obj.get("reason", ""),
        "raw_vlm": raw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=os.getenv("VLLM_MODEL", "ipm-vlm"))
    ap.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--enqueue", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--include-page-images", action="store_true")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    conn = get_conn()

    doc = conn.execute(
        "SELECT doc_id, title, status FROM raw_document WHERE doc_id=?",
        (args.doc_id,),
    ).fetchone()

    if not doc:
        raise SystemExit(f"doc_id not found: {args.doc_id}")

    doc_ctx = get_doc_context(conn, args.doc_id)
    if not doc_ctx.get("title"):
        doc_ctx["title"] = doc["title"] or ""

    assets = conn.execute(
        """
        SELECT *
        FROM raw_asset
        WHERE doc_id=?
        ORDER BY page_no, asset_type, asset_id
        """,
        (args.doc_id,),
    ).fetchall()

    tasks = []

    for a in assets:
        if not args.include_page_images and a["asset_type"] == "page_image":
            continue

        asset_ctx = get_asset_context(conn, a)
        result = vlm_classify(client, args.model, a, doc_ctx, asset_ctx, max_tokens=args.max_tokens)

        task_type = result["task_type"]
        priority_label = result.get("priority", "medium")
        if priority_label not in ["high", "medium", "low"]:
            priority_label = "medium"
        priority = PRIORITY_SCORE[priority_label]
        reason = result["reason"]
        agents = AGENTS.get(task_type, [])

        task_id = uid("task", args.doc_id, a["asset_id"], task_type)

        task = {
            "task_id": task_id,
            "doc_id": args.doc_id,
            "asset_id": a["asset_id"],
            "asset_type": a["asset_type"],
            "file_path": a["file_path"],
            "task_type": task_type,
            "agents": agents,
            "priority": priority,
            "priority_label": priority_label,
            "reason": reason,
        }

        tasks.append(task)

        conn.execute(
            """
            INSERT OR REPLACE INTO planned_tasks
            (task_id, doc_id, asset_id, asset_type, task_type, agents_json, priority, reason, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                args.doc_id,
                a["asset_id"],
                a["asset_type"],
                task_type,
                json.dumps(agents, ensure_ascii=False),
                priority,
                reason,
                "planned",
            ),
        )

        if args.enqueue and task_type != "irrelevant":
            if task_type in ["structure_figure", "sar_table", "supplementary_structure_table"]:
                queue_name = "structure_queue"
            elif task_type in ["mixed_or_uncertain"]:
                queue_name = "vlm_queue"
            else:
                queue_name = "text_queue"

            enqueue(
                doc_id=args.doc_id,
                queue_name=queue_name,
                task_type=task_type,
                payload=task,
            )

        print(json.dumps({
            "asset_id": a["asset_id"],
            "asset_type": a["asset_type"],
            "task_type": task_type,
            "priority": priority,
            "priority_label": priority_label,
            "reason": reason,
        }, ensure_ascii=False))

    out_dir = Path("data/work") / args.doc_id / "planner"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "task_plan.json"
    out_path.write_text(
        json.dumps({"doc_id": args.doc_id, "tasks": tasks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    conn.commit()
    conn.close()

    stat = {}
    for t in tasks:
        stat[t["task_type"]] = stat.get(t["task_type"], 0) + 1

    print(json.dumps({
        "doc_id": args.doc_id,
        "task_plan": str(out_path),
        "num_tasks": len(tasks),
        "stat": stat,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
