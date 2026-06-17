#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8: Align UniParser molecule candidates to compound/component names.

Scope:
- UniParser only: source_tool='uniparser'
- Page-level alignment: group by raw_context_json.page_key / page_image
- Preserve component relations for downstream reconstruction
- No ChemEAGLE / MolParser / direct table compatibility code

Output:
- stg_component_relation
- data/staging/{doc_id}/uniparser_align/uniparser_component_relations.jsonl
- data/staging/{doc_id}/uniparser_align/uniparser_alignment_report.json
- data/staging/{doc_id}/uniparser_align/galleries/*.png
"""

import os
import re
import sys
import json
import uuid
import base64
import argparse
import textwrap
from pathlib import Path
from collections import defaultdict, Counter

from PIL import Image, ImageDraw
from tqdm import tqdm
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn


COMPONENT_ROLES = {
    "full",
    "scaffold",
    "warhead",
    "E3_ligand",
    "linker",
    "R_group",
    "unknown_component",
}

RELATION_TYPES = {
    "exact_full_structure",
    "component_of",
    "variable_series_template",
    "shared_scaffold",
    "ambiguous",
    "no_relation",
}

GENERIC_NAMES = {
    "compound", "compounds", "molecule", "molecules",
    "protac", "protacs", "hyt", "hyt molecules",
    "degrader", "degraders", "analog", "analogs", "analogue", "analogues",
    "series", "template", "templates", "scaffold", "scaffolds",
    "linker", "linkers", "warhead", "warheads",
    "e3 ligand", "e3 ligands", "binder", "binders", "ligand", "ligands",
    "r group", "r groups", "substituent", "substituents",
}

TEXT_SECTIONS = {
    "Title", "Section-header", "Text", "Caption", "Footnote",
    "supplementary_Title", "supplementary_Section-header",
    "supplementary_Text", "supplementary_Caption", "supplementary_Footnote",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def uid(prefix, *parts):
    raw = "|".join(str(x) for x in parts)
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, raw).hex[:16]}"


def jload(x, default=None):
    if default is None:
        default = {}
    try:
        return json.loads(x or "")
    except Exception:
        return default


def jdump(x):
    return json.dumps(x, ensure_ascii=False, default=str)


def row_get(row, key, default=""):
    try:
        v = row[key]
        return default if v is None else v
    except Exception:
        return default


def normalize_text(x):
    x = str(x or "")
    x = re.sub(r"\s+", " ", x).strip()
    return x.strip(" ,;:.()[]{}")


def compact_key(x):
    x = normalize_text(x).lower()
    return re.sub(r"[^a-z0-9]+", "", x)


def is_generic_name(x):
    x = normalize_text(x).lower()
    x = re.sub(r"\s+", " ", x)
    return x in GENERIC_NAMES or compact_key(x) in {compact_key(v) for v in GENERIC_NAMES}


def looks_like_concrete_name(x):
    x = normalize_text(x)
    if not x or is_generic_name(x):
        return False
    if len(x) > 80:
        return False
    # compound numbers such as 1, 12a, 7b
    if re.fullmatch(r"[A-Za-z]?\d+[A-Za-z]?", x):
        return True
    # named agents such as Tepotinib, ARV-825, dBET1
    if re.search(r"[A-Za-z]{2,}[A-Za-z0-9-]*", x):
        return True
    return False


def resolve_path(path):
    p = Path(str(path or ""))
    if not str(p):
        return p
    if not p.is_absolute():
        p = ROOT / p
    return p


def image_to_data_url(path):
    p = resolve_path(path)
    if not p.exists() or p.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    mime = "image/png"
    if p.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"alignments": []}


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------

def ensure_table(conn):
    conn.execute("""
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
    )
    """)
    conn.commit()


def load_uniparser_candidates(conn, doc_id, only_auto_pass=False, limit=0):
    where = [
        "c.doc_id=?",
        "c.source_tool='uniparser'",
        "COALESCE(q.auto_decision, '') != 'review_uniparser_fixed_numeric_repeat_error'",
    ]
    params = [doc_id]

    if only_auto_pass:
        where.append("q.auto_decision='auto_pass_uniparser_valid_no_fixed_numeric_repeat'")

    sql = f"""
    SELECT
        c.*,
        q.qc_score,
        q.auto_decision,
        q.qc_flags_json,
        q.vlm_qc_json
    FROM stg_structure_candidate c
    LEFT JOIN structure_qc_result q
      ON c.candidate_id = q.candidate_id
    WHERE {' AND '.join(where)}
    ORDER BY c.asset_id, c.candidate_index, c.candidate_id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


def insert_component_relation(conn, doc_id, row, alignment, page_meta):
    raw = jload(row_get(row, "raw_output", ""), {})
    ctx = jload(row_get(row, "raw_context_json", ""), {})

    relation_id = uid(
        "ucr",
        doc_id,
        row_get(row, "candidate_id"),
        alignment.get("compound_name", ""),
        alignment.get("component_name", ""),
        alignment.get("component_role", ""),
        alignment.get("relation_type", ""),
        alignment.get("row_label", ""),
        alignment.get("column_label", ""),
    )

    raw_output = {
        "alignment_type": "uniparser_page_component_alignment",
        "page": page_meta,
        "alignment": alignment,
        "candidate": {
            "candidate_id": row_get(row, "candidate_id"),
            "asset_id": row_get(row, "asset_id"),
            "candidate_index": row_get(row, "candidate_index"),
            "image_path": row_get(row, "image_path"),
            "smiles": row_get(row, "smiles"),
            "canonical_smiles": row_get(row, "canonical_smiles"),
            "molecule_label": row_get(row, "molecule_label"),
            "bbox_json": jload(row_get(row, "bbox_json", ""), {}),
            "raw_output": raw,
            "raw_context_json": ctx,
            "stage7_auto_decision": row_get(row, "auto_decision"),
            "stage7_qc_score": row_get(row, "qc_score"),
            "stage7_qc_flags": jload(row_get(row, "qc_flags_json", ""), []),
        },
        "reconstruct_hint": {
            "reconstruct_ready": bool(alignment.get("reconstruct_ready", False)),
            "is_full_molecule": alignment.get("component_role") == "full",
            "is_component": alignment.get("component_role") != "full",
            "component_name": alignment.get("component_name", ""),
            "row_label": alignment.get("row_label", ""),
            "column_label": alignment.get("column_label", ""),
            "series_name": alignment.get("series_name", ""),
            "parent_template_bbox_id": alignment.get("parent_template_bbox_id", ""),
        },
    }

    compound_name = alignment.get("compound_name", "")
    if not compound_name and looks_like_concrete_name(alignment.get("row_label", "")):
        compound_name = normalize_text(alignment.get("row_label", ""))

    conn.execute(
        """
        INSERT OR REPLACE INTO stg_component_relation
        (relation_id, doc_id, asset_id, candidate_id, compound_name,
         component_role, relation_type, evidence_text, figure_ref,
         confidence, review_required, raw_output)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            relation_id,
            doc_id,
            row_get(row, "asset_id"),
            row_get(row, "candidate_id"),
            compound_name,
            alignment.get("component_role", "unknown_component"),
            alignment.get("relation_type", "ambiguous"),
            alignment.get("evidence_text", "")[:500],
            page_meta.get("figure_ref", "") or page_meta.get("table_ref", "") or f"page_{page_meta.get('source_page_no', '')}",
            float(alignment.get("confidence", 0) or 0),
            int(bool(alignment.get("review_required", True))),
            jdump(raw_output),
        ),
    )

    out = {
        "relation_id": relation_id,
        "doc_id": doc_id,
        "asset_id": row_get(row, "asset_id"),
        "candidate_id": row_get(row, "candidate_id"),
        "compound_name": compound_name,
        "component_name": alignment.get("component_name", ""),
        "component_role": alignment.get("component_role", "unknown_component"),
        "relation_type": alignment.get("relation_type", "ambiguous"),
        "row_label": alignment.get("row_label", ""),
        "column_label": alignment.get("column_label", ""),
        "series_name": alignment.get("series_name", ""),
        "parent_template_bbox_id": alignment.get("parent_template_bbox_id", ""),
        "reconstruct_ready": bool(alignment.get("reconstruct_ready", False)),
        "evidence_text": alignment.get("evidence_text", "")[:500],
        "confidence": float(alignment.get("confidence", 0) or 0),
        "review_required": bool(alignment.get("review_required", True)),
        "page_key": page_meta.get("page_key", ""),
        "page_image": page_meta.get("page_image", ""),
    }
    return out


# -----------------------------------------------------------------------------
# UniParser candidate/page helpers
# -----------------------------------------------------------------------------

def row_context(row):
    return jload(row_get(row, "raw_context_json", ""), {})


def candidate_page_key(row):
    ctx = row_context(row)
    return (
        ctx.get("page_key")
        or ctx.get("page_image")
        or ctx.get("source_page_image")
        or row_get(row, "asset_id")
    )


def candidate_page_image(row):
    ctx = row_context(row)
    for k in ["page_image", "source_page_image", "crop_page_image_used"]:
        p = ctx.get(k)
        if p and resolve_path(p).exists():
            return str(resolve_path(p))
    # fallback to candidate crop when page image is missing
    p = row_get(row, "image_path")
    return str(resolve_path(p)) if p else ""


def candidate_source_page_no(row):
    ctx = row_context(row)
    for k in ["source_page_no", "page_no", "page"]:
        v = ctx.get(k)
        if v not in [None, ""]:
            try:
                return int(v)
            except Exception:
                return v
    return ""


def group_by_page_key(candidates):
    groups = defaultdict(list)
    for r in candidates:
        groups[candidate_page_key(r)].append(r)
    return groups


def find_bbox_from_row(row):
    raw = jload(row_get(row, "raw_output", ""), {})
    bbox = jload(row_get(row, "bbox_json", ""), {})

    for obj in [raw, bbox]:
        if isinstance(obj, dict):
            for k in ["float_xyxy", "xyxy", "bbox", "box", "position"]:
                v = obj.get(k)
                if v:
                    return v
        elif isinstance(obj, list) and obj:
            return obj
    return None


def bbox_to_xyxy(bbox, w, h):
    if bbox is None:
        return None

    vals = None
    if isinstance(bbox, dict):
        vals = [
            bbox.get("x0", bbox.get("left")),
            bbox.get("y0", bbox.get("top")),
            bbox.get("x1", bbox.get("right")),
            bbox.get("y1", bbox.get("bottom")),
        ]
    elif isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
        vals = bbox
    elif isinstance(bbox, list) and bbox and isinstance(bbox[0], (list, tuple)):
        xs = [p[0] for p in bbox if len(p) >= 2]
        ys = [p[1] for p in bbox if len(p) >= 2]
        if xs and ys:
            vals = [min(xs), min(ys), max(xs), max(ys)]

    if not vals or any(v is None for v in vals):
        return None

    x0, y0, x1, y1 = map(float, vals)
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        x0, x1 = x0 * w, x1 * w
        y0, y1 = y0 * h, y1 * h

    x0 = max(0, min(w - 1, int(round(x0))))
    y0 = max(0, min(h - 1, int(round(y0))))
    x1 = max(0, min(w, int(round(x1))))
    y1 = max(0, min(h, int(round(y1))))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def candidate_is_markush(row):
    raw = jload(row_get(row, "raw_output", ""), {})
    if isinstance(raw, dict):
        return bool(raw.get("is_markush", False))
    return False


# -----------------------------------------------------------------------------
# Page context
# -----------------------------------------------------------------------------

def collect_page_context(conn, doc_id, page_no, max_chars=16000):
    chunks = []
    captions = []
    tables = []

    page_values = [page_no]
    # In practice raw_text_block may be 1-based while UniParser page may be 0-based.
    # This keeps the implementation simple but avoids empty context.
    if isinstance(page_no, int):
        page_values = sorted({page_no, page_no + 1, page_no - 1})

    qmarks = ",".join("?" for _ in page_values)

    rows = conn.execute(
        f"""
        SELECT page_no, section, text
        FROM raw_text_block
        WHERE doc_id=?
          AND page_no IN ({qmarks})
        ORDER BY page_no, block_id
        """,
        [doc_id] + page_values,
    ).fetchall()
    for r in rows:
        section = row_get(r, "section", "")
        txt = row_get(r, "text", "")
        if section in TEXT_SECTIONS and txt:
            chunks.append(f"[{section} p{row_get(r, 'page_no')}] {txt}")

    rows = conn.execute(
        f"""
        SELECT page_no, figure_ref, caption
        FROM raw_figure
        WHERE doc_id=?
          AND page_no IN ({qmarks})
        ORDER BY page_no, figure_id
        """,
        [doc_id] + page_values,
    ).fetchall()
    for r in rows:
        cap = row_get(r, "caption", "")
        if cap:
            captions.append({
                "page_no": row_get(r, "page_no"),
                "figure_ref": row_get(r, "figure_ref"),
                "caption": cap[:2000],
            })

    rows = conn.execute(
        f"""
        SELECT page_no, table_ref, table_json
        FROM raw_table
        WHERE doc_id=?
          AND page_no IN ({qmarks})
        ORDER BY page_no, table_id
        """,
        [doc_id] + page_values,
    ).fetchall()
    for r in rows:
        tj = row_get(r, "table_json", "")
        if tj:
            tables.append({
                "page_no": row_get(r, "page_no"),
                "table_ref": row_get(r, "table_ref"),
                "table_json": str(tj)[:4000],
            })

    return {
        "source_page_no": page_no,
        "page_text": "\n".join(chunks)[:max_chars],
        "figure_captions": captions[:20],
        "tables": tables[:10],
    }


def compact_uniparser_objects(rows):
    out = []
    for r in rows:
        raw = jload(row_get(r, "raw_output", ""), {})
        if isinstance(raw, dict):
            out.append({
                "candidate_id": row_get(r, "candidate_id"),
                "class": raw.get("class", "molecule"),
                "confidence": raw.get("confidence", ""),
                "float_xyxy": raw.get("float_xyxy", find_bbox_from_row(r)),
                "is_markush": raw.get("is_markush", False),
                "str": raw.get("str", ""),
            })
    return out


# -----------------------------------------------------------------------------
# Gallery creation
# -----------------------------------------------------------------------------

def draw_text_box(draw, xy, text, fill=(220, 0, 0)):
    x, y = xy
    text = str(text)
    try:
        box = draw.textbbox((x, y), text)
        tw = box[2] - box[0]
        th = box[3] - box[1]
    except Exception:
        tw = max(40, 8 * len(text))
        th = 14
    draw.rectangle([x, y, x + tw + 8, y + th + 6], fill=fill)
    draw.text((x + 4, y + 3), text, fill=(255, 255, 255))


def resize_keep(im, max_w, max_h):
    im = im.convert("RGB")
    scale = min(max_w / im.width, max_h / im.height, 1.0)
    nw = max(1, int(im.width * scale))
    nh = max(1, int(im.height * scale))
    im2 = im.resize((nw, nh))
    canvas = Image.new("RGB", (max_w, max_h), (255, 255, 255))
    canvas.paste(im2, ((max_w - nw) // 2, (max_h - nh) // 2))
    return canvas


def crop_with_padding(im, xyxy, pad_ratio=0.20, min_pad=40):
    if not xyxy:
        return im.copy()
    x0, y0, x1, y1 = xyxy
    pad = max(min_pad, int(max(x1 - x0, y1 - y0) * pad_ratio))
    x0p = max(0, x0 - pad)
    y0p = max(0, y0 - pad)
    x1p = min(im.width, x1 + pad)
    y1p = min(im.height, y1 + pad)
    return im.crop([x0p, y0p, x1p, y1p]).convert("RGB")


def wrap_lines(text, width=48, max_lines=6):
    text = str(text or "")
    lines = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=width) or [""])
    return lines[:max_lines]


def build_page_gallery(page_image, rows, out_dir, page_key, chunk_index=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    page_path = resolve_path(page_image)
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(page_key))[:100]
    out_path = out_dir / f"{safe_key}_{chunk_index:03d}.png"

    if not page_path.exists():
        raise FileNotFoundError(f"page image not found: {page_image}")

    page = Image.open(page_path).convert("RGB")
    overlay = page.copy()
    od = ImageDraw.Draw(overlay)

    meta = []
    for i, r in enumerate(rows, start=1):
        bbox_id = f"B{i:03d}"
        bbox = find_bbox_from_row(r)
        xyxy = bbox_to_xyxy(bbox, page.width, page.height)
        if xyxy:
            od.rectangle(xyxy, outline=(220, 0, 0), width=5)
            draw_text_box(od, (xyxy[0], max(0, xyxy[1] - 26)), bbox_id)

        meta.append({
            "bbox_id": bbox_id,
            "candidate_id": row_get(r, "candidate_id"),
            "asset_id": row_get(r, "asset_id"),
            "candidate_index": row_get(r, "candidate_index"),
            "bbox": bbox,
            "xyxy": xyxy,
            "smiles": row_get(r, "smiles"),
            "canonical_smiles": row_get(r, "canonical_smiles"),
            "is_markush": candidate_is_markush(r),
            "stage7_auto_decision": row_get(r, "auto_decision"),
            "stage7_qc_score": row_get(r, "qc_score"),
        })

    left_max_w = 1200
    left_max_h = 1600
    left_img = resize_keep(overlay, left_max_w, left_max_h)

    row_h = 190
    right_w = 900
    header_h = 70
    H = max(left_img.height, header_h + row_h * max(1, len(rows)))
    W = left_img.width + right_w + 30

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(left_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    sep_x = left_img.width + 12
    draw.line([sep_x, 0, sep_x, H], fill=(180, 180, 180), width=2)
    draw.text((sep_x + 15, 18), f"UniParser page alignment gallery | {page_key} | chunk {chunk_index}", fill=(0, 0, 0))
    draw.text((sep_x + 15, 42), "For each B-box, align full molecule or component/template name for reconstruction.", fill=(80, 80, 80))

    for i, (r, m) in enumerate(zip(rows, meta)):
        y = header_h + i * row_h
        crop = crop_with_padding(page, m["xyxy"])
        crop_thumb = resize_keep(crop, 240, 150)
        canvas.paste(crop_thumb, (sep_x + 15, y + 20))
        draw_text_box(draw, (sep_x + 18, y + 22), m["bbox_id"])

        text_x = sep_x + 275
        text_y = y + 18
        lines = [
            f"bbox_id: {m['bbox_id']}",
            f"candidate_id: {str(m['candidate_id'])[:42]}",
            f"is_markush: {m['is_markush']}",
            f"qc: {str(m['stage7_auto_decision'])[:55]}",
            f"smiles: {str(m['smiles'])[:90]}",
        ]
        for line in lines:
            for wl in wrap_lines(line, width=76, max_lines=2):
                draw.text((text_x, text_y), wl, fill=(0, 0, 0))
                text_y += 18
        draw.line([sep_x + 10, y + row_h - 5, W - 15, y + row_h - 5], fill=(220, 220, 220), width=1)

    canvas.save(out_path)
    return str(out_path), meta


# -----------------------------------------------------------------------------
# VLM alignment
# -----------------------------------------------------------------------------

def call_vlm_alignment(client, model, gallery_path, candidate_meta, page_context, uniparser_objects):
    prompt = f"""
You are aligning UniParser molecule candidates from one medicinal chemistry / IPM paper page.

Input image:
- Left: original page image with UniParser molecule bounding boxes labeled B001, B002, ...
- Right: per-candidate crop gallery with bbox_id, candidate_id, SMILES, Markush flag, and QC decision.

Task:
For EVERY candidate in Candidate structures, align it to the visible compound/component meaning on this page.
A candidate can be:
- a complete compound structure
- a scaffold/template
- an R-group substituent
- a linker
- a warhead / target binder
- an E3 ligand
- another component needed for reconstruction
- no useful relation

Important rules:
1. Component fragments are important. Do NOT discard R-groups, linkers, scaffolds, Markush templates, or dummy-atom structures.
2. Use table row/column positions, nearby text, captions, and visible labels to infer row_label, column_label, component_name, and role.
3. Do NOT use generic labels as compound_name: PROTACs, HyT molecules, molecules, compounds, analogs, series, linker, scaffold, warhead, E3 ligand.
4. Do NOT put multiple compounds in one compound_name.
5. If a generic label is visible, put it in series_name or component_name, not compound_name.
6. For table/SAR pages, row_label and column_label are critical for reconstruction.
7. Full exact molecule can be auto-aligned only if the compound_name is concrete and visible/inferable from row/label.
8. For components, set reconstruct_ready=true when the row/column/component identity is useful for later molecule reconstruction, even if review_required=true.

Allowed component_role:
{json.dumps(sorted(COMPONENT_ROLES), ensure_ascii=False)}

Allowed relation_type:
{json.dumps(sorted(RELATION_TYPES), ensure_ascii=False)}

Candidate structures:
{json.dumps(candidate_meta, ensure_ascii=False)}

Page context:
{json.dumps(page_context, ensure_ascii=False)[:20000]}

UniParser molecule objects on this page:
{json.dumps(uniparser_objects, ensure_ascii=False)[:16000]}

Return compact JSON only. Output exactly one alignment object for each candidate_id.

Schema:
{{
  "alignments": [
    {{
      "bbox_id": "B001",
      "candidate_id": "...",
      "compound_name": "",
      "component_name": "",
      "component_role": "full|scaffold|warhead|E3_ligand|linker|R_group|unknown_component",
      "relation_type": "exact_full_structure|component_of|variable_series_template|shared_scaffold|ambiguous|no_relation",
      "row_label": "",
      "column_label": "",
      "series_name": "",
      "parent_template_bbox_id": "",
      "evidence_text": "short visible evidence",
      "reconstruct_ready": true,
      "confidence": 0.0,
      "review_required": true
    }}
  ]
}}
""".strip()

    content = [{"type": "text", "text": prompt}]
    u = image_to_data_url(gallery_path)
    if u:
        content.append({"type": "image_url", "image_url": {"url": u}})

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )
    obj = extract_json(resp.choices[0].message.content)
    if not isinstance(obj.get("alignments"), list):
        obj["alignments"] = []
    return obj


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

def normalize_alignment(aln, meta_by_candidate, meta_by_bbox):
    if not isinstance(aln, dict):
        return None

    candidate_id = normalize_text(aln.get("candidate_id", ""))
    bbox_id = normalize_text(aln.get("bbox_id", ""))

    if candidate_id not in meta_by_candidate and bbox_id in meta_by_bbox:
        candidate_id = meta_by_bbox[bbox_id]["candidate_id"]
    if candidate_id not in meta_by_candidate:
        return None

    compound_name = normalize_text(aln.get("compound_name", ""))
    component_name = normalize_text(aln.get("component_name", ""))
    row_label = normalize_text(aln.get("row_label", ""))
    column_label = normalize_text(aln.get("column_label", ""))
    series_name = normalize_text(aln.get("series_name", ""))

    if compound_name and is_generic_name(compound_name):
        if not series_name:
            series_name = compound_name
        compound_name = ""

    # If VLM puts concrete row label but leaves compound_name empty, use row_label.
    if not compound_name and looks_like_concrete_name(row_label):
        compound_name = row_label

    component_role = normalize_text(aln.get("component_role", "unknown_component"))
    if component_role not in COMPONENT_ROLES:
        component_role = "unknown_component"

    relation_type = normalize_text(aln.get("relation_type", "ambiguous"))
    if relation_type not in RELATION_TYPES:
        relation_type = "ambiguous"

    try:
        confidence = float(aln.get("confidence", 0) or 0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence_text = normalize_text(aln.get("evidence_text", ""))[:500]
    reconstruct_ready = bool(aln.get("reconstruct_ready", False))

    # Conservative rule:
    # - only high-confidence full exact structures can bypass review
    # - all components are preserved but require review before final reconstruction load
    auto_full = (
        component_role == "full"
        and relation_type == "exact_full_structure"
        and looks_like_concrete_name(compound_name)
        and confidence >= 0.85
    )
    review_required = not auto_full

    if relation_type in {"ambiguous", "no_relation"}:
        review_required = True
    if not compound_name and component_role == "full":
        review_required = True
    if component_role != "full":
        review_required = True

    return {
        "bbox_id": bbox_id,
        "candidate_id": candidate_id,
        "compound_name": compound_name,
        "component_name": component_name,
        "component_role": component_role,
        "relation_type": relation_type,
        "row_label": row_label,
        "column_label": column_label,
        "series_name": series_name,
        "parent_template_bbox_id": normalize_text(aln.get("parent_template_bbox_id", "")),
        "evidence_text": evidence_text,
        "reconstruct_ready": reconstruct_ready,
        "confidence": confidence,
        "review_required": review_required,
        "source": "vlm_uniparser_page_alignment",
    }


def fallback_alignment(meta):
    return {
        "bbox_id": meta.get("bbox_id", ""),
        "candidate_id": meta.get("candidate_id", ""),
        "compound_name": "",
        "component_name": "",
        "component_role": "unknown_component",
        "relation_type": "ambiguous",
        "row_label": "",
        "column_label": "",
        "series_name": "",
        "parent_template_bbox_id": "",
        "evidence_text": "No confident UniParser page alignment returned by VLM.",
        "reconstruct_ready": False,
        "confidence": 0.0,
        "review_required": True,
        "source": "fallback_unaligned_candidate",
    }


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield i // n, seq[i:i + n]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--vlm-base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--vlm-model", default=os.getenv("VLLM_MODEL", "ipm-vlm"))
    ap.add_argument("--vlm-api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--only-auto-pass", action="store_true", help="Use only auto_pass_uniparser_valid_no_fixed_numeric_repeat candidates.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-candidates-per-gallery", type=int, default=12)
    args = ap.parse_args()

    client = OpenAI(base_url=args.vlm_base_url, api_key=args.vlm_api_key)

    conn = get_conn()
    ensure_table(conn)

    out_dir = Path("data/staging") / args.doc_id / "uniparser_align"
    gallery_dir = out_dir / "galleries"
    out_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "uniparser_component_relations.jsonl"
    report_path = out_dir / "uniparser_alignment_report.json"
    dry_run_path = out_dir / "dry_run_jobs.json"

    if args.overwrite and not args.dry_run:
        conn.execute("DELETE FROM stg_component_relation WHERE doc_id=?", (args.doc_id,))
        conn.commit()

    candidates = load_uniparser_candidates(
        conn,
        doc_id=args.doc_id,
        only_auto_pass=args.only_auto_pass,
        limit=args.limit,
    )
    groups = group_by_page_key(candidates)

    if args.dry_run:
        jobs = []
        for page_key, rows in groups.items():
            first = rows[0]
            jobs.append({
                "page_key": page_key,
                "page_image": candidate_page_image(first),
                "source_page_no": candidate_source_page_no(first),
                "num_candidates": len(rows),
                "candidate_ids": [row_get(r, "candidate_id") for r in rows],
            })
        dry_run_path.write_text(jdump({
            "doc_id": args.doc_id,
            "num_candidates": len(candidates),
            "num_pages": len(groups),
            "jobs": jobs,
        }), encoding="utf-8")
        print(json.dumps({
            "doc_id": args.doc_id,
            "num_candidates": len(candidates),
            "num_pages": len(groups),
            "dry_run": str(dry_run_path),
        }, ensure_ascii=False, indent=2))
        conn.close()
        return

    stats = Counter()
    outputs = []

    with jsonl_path.open("w", encoding="utf-8") as fw:
        for page_key, page_rows in tqdm(groups.items(), desc="UniParser page alignment"):
            page_image = candidate_page_image(page_rows[0])
            source_page_no = candidate_source_page_no(page_rows[0])
            page_context = collect_page_context(conn, args.doc_id, source_page_no)
            uniparser_objects = compact_uniparser_objects(page_rows)

            page_meta = {
                "page_key": page_key,
                "page_image": page_image,
                "source_page_no": source_page_no,
            }

            for chunk_idx, rows in chunked(page_rows, max(1, args.max_candidates_per_gallery)):
                gallery_path, gallery_meta = build_page_gallery(
                    page_image=page_image,
                    rows=rows,
                    out_dir=gallery_dir,
                    page_key=page_key,
                    chunk_index=chunk_idx,
                )

                page_meta_chunk = dict(page_meta)
                page_meta_chunk["gallery_path"] = gallery_path
                page_meta_chunk["chunk_index"] = chunk_idx

                vlm_obj = call_vlm_alignment(
                    client=client,
                    model=args.vlm_model,
                    gallery_path=gallery_path,
                    candidate_meta=gallery_meta,
                    page_context=page_context,
                    uniparser_objects=uniparser_objects,
                )

                meta_by_candidate = {m["candidate_id"]: m for m in gallery_meta}
                meta_by_bbox = {m["bbox_id"]: m for m in gallery_meta}
                row_by_candidate = {row_get(r, "candidate_id"): r for r in rows}

                normalized = []
                seen_candidates = set()
                for aln in vlm_obj.get("alignments", []):
                    na = normalize_alignment(aln, meta_by_candidate, meta_by_bbox)
                    if not na:
                        continue
                    normalized.append(na)
                    seen_candidates.add(na["candidate_id"])

                # Guarantee one record per candidate even when VLM omits it.
                for m in gallery_meta:
                    if m["candidate_id"] not in seen_candidates:
                        normalized.append(fallback_alignment(m))

                # Deduplicate exact same relation.
                seen = set()
                deduped = []
                for aln in normalized:
                    key = (
                        aln["candidate_id"],
                        aln.get("compound_name", ""),
                        aln.get("component_name", ""),
                        aln.get("component_role", ""),
                        aln.get("relation_type", ""),
                        aln.get("row_label", ""),
                        aln.get("column_label", ""),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(aln)

                for aln in deduped:
                    row = row_by_candidate.get(aln["candidate_id"])
                    if row is None:
                        continue

                    record = insert_component_relation(
                        conn=conn,
                        doc_id=args.doc_id,
                        row=row,
                        alignment=aln,
                        page_meta=page_meta_chunk,
                    )
                    fw.write(jdump(record) + "\n")
                    outputs.append(record)

                    stats["relations"] += 1
                    stats[f"role:{record['component_role']}"] += 1
                    stats[f"relation_type:{record['relation_type']}"] += 1
                    stats["review_required" if record["review_required"] else "auto_aligned"] += 1
                    if record.get("reconstruct_ready"):
                        stats["reconstruct_ready"] += 1

                conn.commit()

    report = {
        "doc_id": args.doc_id,
        "num_candidates": len(candidates),
        "num_pages": len(groups),
        "num_relations": len(outputs),
        "stats": dict(stats),
        "jsonl": str(jsonl_path),
        "gallery_dir": str(gallery_dir),
        "table": "stg_component_relation",
        "note": (
            "UniParser-only page-level alignment. Component relations are preserved for reconstruction. "
            "Only high-confidence full exact structures can be auto-aligned; all components require review but remain reconstruct inputs."
        ),
    }
    report_path.write_text(jdump(report), encoding="utf-8")

    conn.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
