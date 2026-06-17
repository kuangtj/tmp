#!/usr/bin/env python3
import os
import re
import sys
import json
import uuid
import math
import base64
import argparse
from pathlib import Path
from collections import defaultdict, Counter

from tqdm import tqdm
from openai import OpenAI
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import Draw

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
    "shared_scaffold",
    "component_of",
    "variable_series_template",
    "ambiguous",
    "no_relation",
}

IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def uid(prefix, *parts):
    return prefix + "_" + uuid.uuid5(
        uuid.NAMESPACE_URL,
        "|".join(map(str, parts)),
    ).hex[:16]


def jload(s, default=None):
    try:
        return json.loads(s or "")
    except Exception:
        return default if default is not None else {}


def jdump(x):
    return json.dumps(x, ensure_ascii=False, default=str)


def table_exists(conn, table):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def table_cols(conn, table):
    if not table_exists(conn, table):
        return set()
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


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



DIRECT_STRUCTURE_SOURCE_TOOLS = {
    "supplement_table_direct",
    "supplement_direct_smiles",
    "csv_direct",
    "xlsx_direct",
    "sdf_direct",
    "smi_direct",
}

DIRECT_STRUCTURE_SUFFIXES = {
    ".csv", ".tsv", ".tab",
    ".xlsx", ".xlsm",
    ".sdf", ".smi", ".smiles",
}


def row_get(row, key, default=""):
    try:
        v = row[key]
        return default if v is None else v
    except Exception:
        return default


def is_direct_structure_candidate(row):
    source_tool = str(row_get(row, "source_tool", "") or "")
    smiles_source = str(row_get(row, "smiles_source", "") or "")
    asset_file = str(row_get(row, "image_path", "") or row_get(row, "file_path", "") or "")
    suffix = Path(asset_file).suffix.lower() if asset_file else ""

    raw_output = jload(row_get(row, "raw_output", ""), {})
    raw_context = jload(row_get(row, "raw_context_json", ""), {})

    raw_text = json.dumps(
        {
            "source_tool": source_tool,
            "smiles_source": smiles_source,
            "raw_output": raw_output,
            "raw_context": raw_context,
        },
        ensure_ascii=False,
        default=str,
    ).lower()

    if source_tool in DIRECT_STRUCTURE_SOURCE_TOOLS:
        return True

    if suffix in DIRECT_STRUCTURE_SUFFIXES:
        return True

    if "supplement_table_direct" in raw_text:
        return True

    if "direct" in source_tool.lower() and smiles_source.lower() in {
        "smiles", "inchi", "molblock", "sdf", "smi"
    }:
        return True

    return False


def extract_direct_compound_name(row):
    candidates = []

    # 1. candidate 表自身字段
    for k in ["molecule_label", "label", "compound_name", "name"]:
        v = normalize_name(row_get(row, k, ""))
        if v:
            candidates.append(v)

    # 2. raw_output
    raw = jload(row_get(row, "raw_output", ""), {})
    if isinstance(raw, dict):
        for k in ["molecule_label", "label", "compound_name", "name"]:
            v = normalize_name(raw.get(k, ""))
            if v:
                candidates.append(v)

    # 3. raw_context_json from direct supplement parser
    ctx = jload(row_get(row, "raw_context_json", ""), {})
    if isinstance(ctx, dict):
        for k in ["compound_name", "molecule_label", "label", "name"]:
            v = normalize_name(ctx.get(k, ""))
            if v:
                candidates.append(v)

        r = ctx.get("row")
        if isinstance(r, dict):
            for k in [
                "compound", "compound name", "compound_name",
                "compound id", "compound_id",
                "cpd", "cpd id", "id", "name", "no", "no.", "entry"
            ]:
                for rk, rv in r.items():
                    if normalize_name(rk).lower().replace("-", "_") == normalize_name(k).lower().replace("-", "_"):
                        v = normalize_name(rv)
                        if v:
                            candidates.append(v)

            # fallback: first short non-structure field
            for rk, rv in r.items():
                nk = normalize_name(rk).lower()
                if any(x in nk for x in ["smiles", "inchi", "molfile", "molblock", "structure"]):
                    continue
                v = normalize_name(rv)
                if v and len(v) <= 80:
                    candidates.append(v)
                    break

    for v in candidates:
        if v and v.lower() not in {
            "compound", "compounds", "molecule", "molecules",
            "smiles", "structure", "inchi"
        }:
            return v

    return ""


def direct_structure_relation(row, alias_map):
    raw_name = extract_direct_compound_name(row)
    compound_name = canonicalize_compound_name(raw_name, alias_map) or raw_name

    structure_class = structure_class_from_candidate(row)
    auto_decision = row_get(row, "auto_decision", "") or ""
    qc_score = row_get(row, "qc_score", 0) or 0

    try:
        qc_score = float(qc_score)
    except Exception:
        qc_score = 0.0

    if not compound_name:
        return fallback_relation(row, reason="Direct structure candidate has no compound name.")

    if auto_decision.startswith("auto_pass_direct") or (
        auto_decision.startswith("auto_pass") and structure_class == "full_molecule"
    ):
        return {
            "candidate_id": row["candidate_id"],
            "compound_name": compound_name,
            "component_role": "full",
            "relation_type": "exact_full_structure",
            "evidence_text": f"Direct supplementary table/file structure gives full structure for {compound_name}.",
            "confidence": 0.99,
            "review_required": False,
            "source": "direct_table_row_alignment",
            "bbox_id": "",
        }

    if structure_class == "full_molecule" and qc_score >= 90:
        return {
            "candidate_id": row["candidate_id"],
            "compound_name": compound_name,
            "component_role": "full",
            "relation_type": "exact_full_structure",
            "evidence_text": f"Direct supplementary table/file structure gives full structure for {compound_name}.",
            "confidence": 0.95,
            "review_required": False,
            "source": "direct_table_row_alignment",
            "bbox_id": "",
        }

    if structure_class == "markush_or_variable":
        return {
            "candidate_id": row["candidate_id"],
            "compound_name": compound_name,
            "component_role": "scaffold",
            "relation_type": "variable_series_template",
            "evidence_text": f"Direct supplementary table/file gives variable or Markush structure for {compound_name}.",
            "confidence": 0.75,
            "review_required": True,
            "source": "direct_table_row_alignment_review",
            "bbox_id": "",
        }

    if structure_class == "component_or_attachment":
        return {
            "candidate_id": row["candidate_id"],
            "compound_name": compound_name,
            "component_role": "unknown_component",
            "relation_type": "component_of",
            "evidence_text": f"Direct supplementary table/file gives component structure for {compound_name}.",
            "confidence": 0.72,
            "review_required": True,
            "source": "direct_table_row_alignment_review",
            "bbox_id": "",
        }

    if structure_class in {"fragment_or_multicomponent", "multicomponent_or_merged"}:
        return {
            "candidate_id": row["candidate_id"],
            "compound_name": compound_name,
            "component_role": "unknown_component",
            "relation_type": "ambiguous",
            "evidence_text": f"Direct supplementary table/file gives multicomponent structure for {compound_name}; review required.",
            "confidence": 0.65,
            "review_required": True,
            "source": "direct_table_row_alignment_review",
            "bbox_id": "",
        }

    return {
        "candidate_id": row["candidate_id"],
        "compound_name": compound_name,
        "component_role": "unknown_component",
        "relation_type": "ambiguous",
        "evidence_text": f"Direct supplementary table/file structure for {compound_name} requires review.",
        "confidence": 0.50,
        "review_required": True,
        "source": "direct_table_row_alignment_review",
        "bbox_id": "",
    }


def image_to_data_url(path):
    p = Path(path or "")
    if not p.exists() or p.suffix.lower() not in IMAGE_SUFFIX:
        return None

    mime = "image/png"
    if p.suffix.lower() in [".jpg", ".jpeg"]:
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

    return {}


def normalize_name(x):
    x = str(x or "").strip()
    x = re.sub(r"\s+", " ", x)
    x = x.strip(" ,;:.()[]{}")
    return x


def compact_name_key(x):
    x = normalize_name(x).lower()
    x = re.sub(r"\b(compound|compounds|compd\.?|cmpd\.?|molecule|protac|degrader)\b", "", x)
    x = re.sub(r"[^a-z0-9]+", "", x)
    return x


def add_name(names, name, source):
    name = normalize_name(name)
    if not name:
        return

    bad = {
        "compound", "compounds", "molecule", "molecules",
        "protac", "degrader", "linker", "warhead", "scaffold",
    }
    if name.lower() in bad:
        return
    if len(name) > 80:
        return

    names[name].add(source)


def extract_names_from_label(label, names):
    label = label or ""

    for part in re.split(r"\|\||[,;/]", label):
        p = normalize_name(part)
        if not p:
            continue

        # e.g. 27a: n = 2 -> 27a
        if ":" in p:
            left = normalize_name(p.split(":", 1)[0])
            if left:
                add_name(names, left, "structure_label")

        for m in re.finditer(
            r"\b(?:compound|compd\.?|cmpd\.?|PROTAC|degrader|molecule)\s*[-#:]?\s*([A-Za-z]*\d+[A-Za-z]?(?:[-–][A-Za-z0-9]+)?)",
            p,
            re.I,
        ):
            add_name(names, m.group(1), "structure_label")
            add_name(names, m.group(0), "structure_label")

        if re.fullmatch(r"[A-Za-z]?\d+[A-Za-z]?", p):
            add_name(names, p, "structure_label")

        for m in re.finditer(r"\b[A-Za-z]{1,8}-?\d+[A-Za-z]?\b", p):
            add_name(names, m.group(0), "structure_label")


def extract_names_regex(text, names, source):
    text = text or ""

    patterns = [
        r"\b(?:compound|compounds|compd\.?|cmpd\.?)\s*[-#:]?\s*([A-Za-z]*\d+[A-Za-z]?(?:[-–][A-Za-z0-9]+)?)",
        r"\b(?:PROTAC|degrader|molecule)\s*[-#:]?\s*([A-Za-z]*\d+[A-Za-z]?(?:[-–][A-Za-z0-9]+)?)",
        r"\b(?:ARV-\d+[A-Za-z]?|dBET\d*|MZ\d+|MT-\d+|AT\d+|PROTAC\s*\d+[A-Za-z]?)\b",
        r"\b\d+[a-z]\b",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            if len(m.groups()) >= 1:
                add_name(names, m.group(1), source)
                add_name(names, m.group(0), source)
            else:
                add_name(names, m.group(0), source)


def get_text_column(cols):
    for c in ["text", "content", "block_text", "text_content", "markdown"]:
        if c in cols:
            return c
    return None


def build_global_context(conn, doc_id, max_chars=30000):
    chunks = []

    if table_exists(conn, "raw_text_block"):
        cols = table_cols(conn, "raw_text_block")
        text_col = get_text_column(cols)
        if text_col and "doc_id" in cols:
            rows = conn.execute(
                f"""
                SELECT {text_col} AS text
                FROM raw_text_block
                WHERE doc_id=?
                LIMIT 300
                """,
                (doc_id,),
            ).fetchall()
            for r in rows:
                txt = r["text"] or ""
                if txt.strip():
                    chunks.append(txt.strip())

    if table_exists(conn, "raw_figure"):
        cols = table_cols(conn, "raw_figure")
        if "caption" in cols and "doc_id" in cols:
            rows = conn.execute(
                "SELECT caption FROM raw_figure WHERE doc_id=? LIMIT 200",
                (doc_id,),
            ).fetchall()
            for r in rows:
                if r["caption"]:
                    chunks.append(str(r["caption"]))

    if table_exists(conn, "raw_table"):
        cols = table_cols(conn, "raw_table")
        if "table_json" in cols and "doc_id" in cols:
            rows = conn.execute(
                "SELECT table_json FROM raw_table WHERE doc_id=? LIMIT 100",
                (doc_id,),
            ).fetchall()
            for r in rows:
                if r["table_json"]:
                    chunks.append(str(r["table_json"])[:3000])

    return "\n\n".join(chunks)[:max_chars]


def extract_names_llm(client, model, context):
    if not context.strip():
        return []

    prompt = f"""
Extract paper-native compound names or labels from the text.

Allowed examples:
- compound 1
- 1
- 2a
- dBET1
- ARV-825
- PROTAC 12
- degrader 5

Rules:
- Return exact strings that appear in the text.
- Do not invent names.
- Do not include generic words alone, such as "compound", "PROTAC", "degrader", "linker", "warhead".
- Return compact JSON only.

Schema:
{{
  "compound_names": ["..."]
}}

Text:
{context}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    obj = extract_json(resp.choices[0].message.content)
    out = obj.get("compound_names", [])
    if not isinstance(out, list):
        return []

    return [normalize_name(x) for x in out if normalize_name(x)]


def load_asset_contexts(conn, doc_id):
    out = {}

    if not table_exists(conn, "raw_asset"):
        return out

    asset_cols = table_cols(conn, "raw_asset")
    select_cols = [
        c for c in [
            "asset_id", "asset_type", "file_path", "page_no",
            "figure_ref", "table_ref", "metadata_json",
        ]
        if c in asset_cols
    ]

    rows = conn.execute(
        f"SELECT {','.join(select_cols)} FROM raw_asset WHERE doc_id=?",
        (doc_id,),
    ).fetchall()

    for r in rows:
        d = dict(r)
        asset_id = d.get("asset_id")
        if asset_id:
            out[asset_id] = {"asset": d, "caption": "", "table_json": ""}

    if table_exists(conn, "raw_figure"):
        cols = table_cols(conn, "raw_figure")
        if "asset_id" in cols:
            select = [c for c in ["asset_id", "caption", "figure_ref"] if c in cols]
            rows = conn.execute(f"SELECT {','.join(select)} FROM raw_figure").fetchall()
            for r in rows:
                d = dict(r)
                aid = d.get("asset_id")
                if aid in out:
                    if d.get("caption"):
                        out[aid]["caption"] = d.get("caption") or ""
                    if d.get("figure_ref"):
                        out[aid]["asset"]["figure_ref"] = d.get("figure_ref")

    if table_exists(conn, "raw_table"):
        cols = table_cols(conn, "raw_table")
        if "asset_id" in cols:
            select = [c for c in ["asset_id", "table_json", "table_ref"] if c in cols]
            rows = conn.execute(f"SELECT {','.join(select)} FROM raw_table").fetchall()
            for r in rows:
                d = dict(r)
                aid = d.get("asset_id")
                if aid in out:
                    if d.get("table_json"):
                        out[aid]["table_json"] = d.get("table_json") or ""
                    if d.get("table_ref"):
                        out[aid]["asset"]["table_ref"] = d.get("table_ref")

    return out


def load_candidates(conn, doc_id, source_tool="", only_auto_pass=False, limit=None):
    where = ["c.doc_id=?"]
    params = [doc_id]

    if source_tool:
        where.append("c.source_tool=?")
        params.append(source_tool)

    if only_auto_pass:
        where.append("q.auto_decision LIKE 'auto_pass%'")

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
    ORDER BY c.asset_id, c.candidate_index
    """

    if limit:
        sql += f" LIMIT {int(limit)}"

    return conn.execute(sql, params).fetchall()


def candidate_label(raw):
    if not isinstance(raw, dict):
        return ""

    label = (
        raw.get("molecule_label")
        or raw.get("label")
        or "||".join(raw.get("molecule_texts", []) or [])
        or ""
    )
    return str(label or "")


def structure_class_from_candidate(row):
    vlm = jload(row["vlm_qc_json"], {})
    if isinstance(vlm, dict) and vlm.get("structure_class"):
        return vlm.get("structure_class")

    flags = jload(row["qc_flags_json"], [])
    if not isinstance(flags, list):
        flags = []

    for f in flags:
        if isinstance(f, str) and f.startswith("class:"):
            return f.split(":", 1)[1]

    return "uncertain"


def build_compound_name_list(conn, doc_id, candidates, client, model, use_llm=True):
    names = defaultdict(set)
    global_context = build_global_context(conn, doc_id)

    extract_names_regex(global_context, names, "paper_text")

    for r in candidates:
        raw = jload(r["raw_output"], {})
        label = candidate_label(raw)
        extract_names_from_label(label, names)
        if isinstance(raw, dict):
            extract_names_regex(jdump(raw)[:5000], names, "structure_raw_output")

    if use_llm and global_context.strip():
        try:
            for n in extract_names_llm(client, model, global_context):
                add_name(names, n, "llm_text_extraction")
        except Exception as e:
            print(f"[WARN] LLM compound extraction failed: {e}")

    out = []
    for name, sources in names.items():
        out.append({"compound_name": name, "sources": sorted(sources)})

    out.sort(key=lambda x: (len(x["compound_name"]), x["compound_name"]))
    return out


def build_name_alias_map(compound_names):
    alias = {}

    for item in compound_names:
        name = item["compound_name"] if isinstance(item, dict) else str(item)
        name = normalize_name(name)
        if not name:
            continue

        forms = {name, name.lower(), compact_name_key(name)}

        m = re.search(r"\b(?:compound|compounds|compd\.?|cmpd\.?|PROTAC|degrader|molecule)\s*[-#:]?\s*([A-Za-z]*\d+[A-Za-z]?)\b", name, re.I)
        if m:
            forms.add(m.group(1))
            forms.add(m.group(1).lower())
            forms.add(compact_name_key(m.group(1)))

        if re.fullmatch(r"[A-Za-z]?\d+[A-Za-z]?", name):
            forms.add(f"compound {name}")
            forms.add(f"PROTAC {name}")
            forms.add(compact_name_key(f"compound {name}"))
            forms.add(compact_name_key(f"PROTAC {name}"))

        for f in forms:
            if f:
                alias[f] = name

    return alias


def canonicalize_compound_name(name, alias_map):
    name = normalize_name(name)
    if not name:
        return ""

    for key in [name, name.lower(), compact_name_key(name)]:
        if key in alias_map:
            return alias_map[key]

    return ""


def find_names_in_label(label, alias_map):
    found = []
    label = label or ""

    candidates = []
    for part in re.split(r"\|\||[,;/]", label):
        p = normalize_name(part)
        if not p:
            continue
        candidates.append(p)
        if ":" in p:
            candidates.append(normalize_name(p.split(":", 1)[0]))
        for m in re.finditer(r"\b(?:compound|compd\.?|cmpd\.?|PROTAC|degrader|molecule)\s*[-#:]?\s*([A-Za-z]*\d+[A-Za-z]?)", p, re.I):
            candidates.append(m.group(1))
            candidates.append(m.group(0))
        for m in re.finditer(r"\b[A-Za-z]?\d+[A-Za-z]?\b", p):
            candidates.append(m.group(0))

    for c in candidates:
        cn = canonicalize_compound_name(c, alias_map)
        if cn and cn not in found:
            found.append(cn)

    return found


def compact_asset_context(asset_ctx):
    if not asset_ctx:
        return {}

    asset = asset_ctx.get("asset", {})
    return {
        "asset_id": asset.get("asset_id", ""),
        "asset_type": asset.get("asset_type", ""),
        "figure_ref": asset.get("figure_ref", ""),
        "table_ref": asset.get("table_ref", ""),
        "page_no": asset.get("page_no", ""),
        "caption": (asset_ctx.get("caption") or "")[:3000],
        "table_json": (asset_ctx.get("table_json") or "")[:5000],
    }


def find_bbox(obj):
    if not isinstance(obj, dict):
        return None

    for k in ["bbox", "box", "xyxy", "position"]:
        if k in obj and obj[k]:
            return obj[k]

    for v in obj.values():
        if isinstance(v, dict):
            b = find_bbox(v)
            if b:
                return b
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, dict):
                    b = find_bbox(x)
                    if b:
                        return b

    return None


def find_bbox_id(obj):
    if not isinstance(obj, dict):
        return ""

    for k in ["bbox_id", "box_id", "bbox_index", "crop_id", "id"]:
        v = obj.get(k)
        if v not in [None, ""]:
            return str(v)

    for v in obj.values():
        if isinstance(v, dict):
            got = find_bbox_id(v)
            if got:
                return got
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, dict):
                    got = find_bbox_id(x)
                    if got:
                        return got
    return ""


def bbox_to_xyxy(bbox, w, h):
    if bbox is None:
        return None

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
        vals = [min(xs), min(ys), max(xs), max(ys)] if xs and ys else None
    else:
        vals = None

    if not vals or any(v is None for v in vals):
        return None

    x0, y0, x1, y1 = map(float, vals)

    if max(x0, y0, x1, y1) <= 1.5:
        x0, x1 = x0 * w, x1 * w
        y0, y1 = y0 * h, y1 * h

    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(0, min(w, int(x1)))
    y1 = max(0, min(h, int(y1)))

    if x1 <= x0 or y1 <= y0:
        return None

    return [x0, y0, x1, y1]


def draw_label(draw, xy, text, fill=(220, 0, 0)):
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


def resize_keep(im, max_w, max_h, bg=(255, 255, 255)):
    im = im.convert("RGB")
    scale = min(max_w / im.width, max_h / im.height, 1.0)
    nw = max(1, int(im.width * scale))
    nh = max(1, int(im.height * scale))
    resized = im.resize((nw, nh))
    canvas = Image.new("RGB", (max_w, max_h), bg)
    canvas.paste(resized, ((max_w - nw) // 2, (max_h - nh) // 2))
    return canvas


def crop_with_padding(im, xyxy, pad_ratio=0.18, min_pad=20):
    if not xyxy:
        return im.copy()
    x0, y0, x1, y1 = xyxy
    bw = x1 - x0
    bh = y1 - y0
    pad = max(min_pad, int(max(bw, bh) * pad_ratio))
    x0p = max(0, x0 - pad)
    y0p = max(0, y0 - pad)
    x1p = min(im.width, x1 + pad)
    y1p = min(im.height, y1 + pad)
    crop = im.crop([x0p, y0p, x1p, y1p]).convert("RGB")
    draw = ImageDraw.Draw(crop)
    draw.rectangle([x0 - x0p, y0 - y0p, x1 - x0p, y1 - y0p], outline=(220, 0, 0), width=4)
    return crop


def render_smiles_image(smiles, size=(300, 180)):
    smiles = smiles or ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=size).convert("RGB")
    except Exception:
        return None


def candidate_bbox_label(row, raw, fallback_index):
    bid = find_bbox_id(raw)
    if bid:
        return str(bid)
    idx = row["candidate_index"] if "candidate_index" in row.keys() else fallback_index
    try:
        return f"B{int(idx):03d}"
    except Exception:
        return f"B{fallback_index:03d}"


def make_asset_bbox_gallery(asset_image_path, rows, out_dir, asset_id, page_index=0):
    """
    Create one composite image for an asset chunk:
    - Left: original asset image with all candidate bboxes highlighted and labelled.
    - Right: per-candidate thumbnail rows, each labelled by bbox_id/candidate_id and paired with RDKit render.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{asset_id}_bbox_gallery_{page_index:03d}.png"

    if not asset_image_path or not Path(asset_image_path).exists():
        blank = Image.new("RGB", (1200, 800), (255, 255, 255))
        draw = ImageDraw.Draw(blank)
        draw.text((30, 30), f"Missing asset image: {asset_image_path}", fill=(0, 0, 0))
        blank.save(out_path)
        return str(out_path), []

    im = Image.open(asset_image_path).convert("RGB")
    overlay = im.copy()
    od = ImageDraw.Draw(overlay)

    meta = []
    for i, r in enumerate(rows, start=1):
        raw = jload(r["raw_output"], {})
        bbox = find_bbox(raw)
        xyxy = bbox_to_xyxy(bbox, im.width, im.height)
        bbox_id = candidate_bbox_label(r, raw, i)
        label = candidate_label(raw)
        if xyxy:
            od.rectangle(xyxy, outline=(220, 0, 0), width=5)
            draw_label(od, (xyxy[0], max(0, xyxy[1] - 24)), bbox_id)
        meta.append({
            "bbox_id": bbox_id,
            "candidate_id": r["candidate_id"],
            "candidate_index": r["candidate_index"],
            "bbox": bbox,
            "xyxy": xyxy,
            "molecule_label": label,
            "smiles": r["smiles"] or "",
            "canonical_smiles": r["canonical_smiles"] or "",
            "stage7_auto_decision": r["auto_decision"] or "",
            "stage7_qc_score": r["qc_score"],
            "stage7_structure_class": structure_class_from_candidate(r),
        })

    left_max_w = 1200
    left_max_h = 1600
    scale = min(left_max_w / overlay.width, left_max_h / overlay.height, 1.0)
    left_w = max(1, int(overlay.width * scale))
    left_h = max(1, int(overlay.height * scale))
    left_img = overlay.resize((left_w, left_h))

    row_h = 240
    right_w = 900
    header_h = 70
    gallery_h = header_h + max(row_h * len(rows), 1)
    H = max(left_h, gallery_h)
    W = left_w + right_w + 30

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(left_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line([left_w + 10, 0, left_w + 10, H], fill=(180, 180, 180), width=2)
    draw.text((left_w + 25, 18), f"BBox candidate gallery: {asset_id} | page {page_index}", fill=(0, 0, 0))
    draw.text((left_w + 25, 42), "Each row: highlighted crop + RDKit render + candidate metadata", fill=(80, 80, 80))

    for i, (r, m) in enumerate(zip(rows, meta)):
        y = header_h + i * row_h
        raw = jload(r["raw_output"], {})
        xyxy = m["xyxy"]
        bbox_id = m["bbox_id"]
        label = m["molecule_label"]

        crop = crop_with_padding(im, xyxy) if xyxy else im.copy()
        crop_thumb = resize_keep(crop, 270, 180)
        canvas.paste(crop_thumb, (left_w + 25, y + 25))
        draw_label(draw, (left_w + 28, y + 28), bbox_id)

        smiles = r["canonical_smiles"] or r["smiles"] or ""
        rdkit_img = render_smiles_image(smiles, size=(270, 180))
        if rdkit_img:
            rdkit_thumb = resize_keep(rdkit_img, 270, 180)
        else:
            rdkit_thumb = Image.new("RGB", (270, 180), (245, 245, 245))
            dd = ImageDraw.Draw(rdkit_thumb)
            dd.text((20, 80), "No RDKit render", fill=(80, 80, 80))
        canvas.paste(rdkit_thumb, (left_w + 315, y + 25))

        tx = left_w + 610
        ty = y + 24
        lines = [
            f"bbox_id: {bbox_id}",
            f"candidate_index: {r['candidate_index']}",
            f"candidate_id: {r['candidate_id'][:28]}",
            f"label: {label[:58]}",
            f"class: {m['stage7_structure_class']}",
            f"decision: {(r['auto_decision'] or '')[:42]}",
            f"smiles: {(r['smiles'] or '')[:58]}",
        ]
        for j, line in enumerate(lines):
            draw.text((tx, ty + j * 22), line, fill=(0, 0, 0))

        draw.line([left_w + 20, y + row_h - 5, W - 15, y + row_h - 5], fill=(220, 220, 220), width=1)

    canvas.save(out_path)
    return str(out_path), meta


def infer_role_relation_from_structure_class(structure_class):
    if structure_class == "full_molecule":
        return "full", "exact_full_structure"
    if structure_class == "markush_or_variable":
        return "scaffold", "variable_series_template"
    if structure_class == "component_or_attachment":
        return "unknown_component", "component_of"
    if structure_class in {"fragment_or_multicomponent", "multicomponent_or_merged"}:
        return "unknown_component", "ambiguous"
    return "unknown_component", "ambiguous"


def rule_based_relations(row, alias_map):
    raw = jload(row["raw_output"], {})
    label = candidate_label(raw)
    names = find_names_in_label(label, alias_map)
    if not names:
        return []

    structure_class = structure_class_from_candidate(row)
    role, relation_type = infer_role_relation_from_structure_class(structure_class)

    auto_decision = row["auto_decision"] or ""
    is_auto_pass = auto_decision.startswith("auto_pass")
    confidence = 0.84 if is_auto_pass else 0.72

    out = []
    for name in names:
        out.append({
            "candidate_id": row["candidate_id"],
            "compound_name": name,
            "component_role": role,
            "relation_type": relation_type,
            "evidence_text": f"Structure label indicates {name}: {label}"[:500],
            "confidence": confidence,
            "review_required": not is_auto_pass or relation_type != "exact_full_structure",
            "source": "rule_label_match",
        })

    return out


def filter_allowed_names_for_asset(allowed_names, rows, asset_ctx, alias_map, max_names=250):
    prioritized = []

    def add(x):
        if x and x not in prioritized:
            prioritized.append(x)

    for r in rows:
        raw = jload(r["raw_output"], {})
        for x in find_names_in_label(candidate_label(raw), alias_map):
            add(x)

    ctx = jdump(compact_asset_context(asset_ctx)).lower()
    for name in allowed_names:
        if name.lower() in ctx or compact_name_key(name) in re.sub(r"[^a-z0-9]+", "", ctx):
            add(name)

    for name in allowed_names:
        add(name)
        if len(prioritized) >= max_names:
            break

    return prioritized[:max_names]


def align_asset_gallery_with_vlm(
    client,
    model,
    asset_ctx,
    rows,
    allowed_names,
    alias_map,
    gallery_path,
    gallery_meta,
):
    allowed_subset = filter_allowed_names_for_asset(allowed_names, rows, asset_ctx, alias_map)

    candidate_payload = []
    for m in gallery_meta:
        candidate_payload.append({
            "bbox_id": m["bbox_id"],
            "candidate_id": m["candidate_id"],
            "candidate_index": m["candidate_index"],
            "molecule_label": m["molecule_label"],
            "smiles": m["smiles"],
            "canonical_smiles": m["canonical_smiles"],
            "stage7_auto_decision": m["stage7_auto_decision"],
            "stage7_qc_score": m["stage7_qc_score"],
            "stage7_structure_class": m["stage7_structure_class"],
            "has_bbox": bool(m.get("xyxy")),
        })

    prompt = f"""
You are aligning chemical structure candidates to compound names in a real medicinal chemistry / IPM paper.

You are given ONE composite image.
Composite image layout:
- Left side: the original asset image with candidate bboxes highlighted and labelled by bbox_id.
- Right side: a bbox gallery. Each row contains a highlighted candidate crop, the RDKit rendering of the candidate SMILES, and metadata including bbox_id, candidate_index, candidate_id, molecule_label, and stage7 class.

Task:
For each candidate_id in Candidate structures, decide which compound name(s) it corresponds to and whether the structure is a full molecule or a component.

Important rules:
1. Use the bbox_id labels in the composite image to locate the candidate structure.
2. One shared scaffold may map to multiple compounds.
3. A full molecule relation has priority over component-based relations.
4. If the candidate crop/overlay does not clearly bind to a compound label, set review_required=true.
5. If the figure/caption only shows a component, use component_role such as scaffold, warhead, E3_ligand, linker, R_group, unknown_component.
6. If the same candidate is a shared scaffold/template for multiple compounds, output multiple relation objects.
7. Prefer evidence from visible label near the bbox, caption, SAR table row/column, and candidate molecule_label.
8. Do not reject just because SMILES has dummy atoms, R-group notation, or disconnected components; classify the component role.

Allowed component_role:
{sorted(COMPONENT_ROLES)}

Allowed relation_type:
{sorted(RELATION_TYPES)}

Allowed compound names:
{json.dumps(allowed_subset, ensure_ascii=False)}

Asset context:
{json.dumps(compact_asset_context(asset_ctx), ensure_ascii=False)}

Candidate structures:
{json.dumps(candidate_payload, ensure_ascii=False)}

Return compact JSON only.

Schema:
{{
  "relations": [
    {{
      "bbox_id": "...",
      "candidate_id": "...",
      "compound_name": "...",
      "component_role": "full|scaffold|warhead|E3_ligand|linker|R_group|unknown_component",
      "relation_type": "exact_full_structure|shared_scaffold|component_of|variable_series_template|ambiguous|no_relation",
      "evidence_text": "short evidence from label/caption/table/image",
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
    rels = obj.get("relations", [])
    if not isinstance(rels, list):
        rels = []

    return rels, obj


def normalize_relation(rel, candidate_id_set, bbox_to_candidate, alias_map):
    cid = str(rel.get("candidate_id", "")).strip()
    bbox_id = str(rel.get("bbox_id", "")).strip()

    if cid not in candidate_id_set and bbox_id in bbox_to_candidate:
        cid = bbox_to_candidate[bbox_id]

    if cid not in candidate_id_set:
        return None

    compound_name = canonicalize_compound_name(rel.get("compound_name", ""), alias_map)

    component_role = rel.get("component_role", "unknown_component")
    if component_role not in COMPONENT_ROLES:
        component_role = "unknown_component"

    relation_type = rel.get("relation_type", "ambiguous")
    if relation_type not in RELATION_TYPES:
        relation_type = "ambiguous"

    try:
        confidence = float(rel.get("confidence", 0) or 0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence_text = str(rel.get("evidence_text", ""))[:500]
    review_required = bool(rel.get("review_required", False))

    if not compound_name:
        review_required = True
    if confidence < 0.70:
        review_required = True
    if relation_type in {"ambiguous", "no_relation"}:
        review_required = True

    return {
        "candidate_id": cid,
        "compound_name": compound_name,
        "component_role": component_role,
        "relation_type": relation_type,
        "evidence_text": evidence_text,
        "confidence": confidence,
        "review_required": review_required,
        "source": rel.get("source", "vlm_asset_bbox_gallery_alignment"),
        "bbox_id": bbox_id,
    }


def fallback_relation(row, reason="No confident compound-structure relation returned."):
    raw = jload(row["raw_output"], {})
    return {
        "candidate_id": row["candidate_id"],
        "compound_name": "",
        "component_role": "unknown_component",
        "relation_type": "ambiguous",
        "evidence_text": reason,
        "confidence": 0.0,
        "review_required": True,
        "source": "fallback_review",
        "bbox_id": find_bbox_id(raw),
    }


def insert_relation(conn, doc_id, asset_id, rel, figure_ref, raw_output):
    relation_id = uid(
        "rel",
        doc_id,
        asset_id,
        rel["candidate_id"],
        rel.get("compound_name", ""),
        rel["component_role"],
        rel["relation_type"],
        rel.get("source", ""),
    )

    conn.execute(
        """
        INSERT OR REPLACE INTO stg_component_relation
        (
            relation_id,
            doc_id,
            asset_id,
            candidate_id,
            compound_name,
            component_role,
            relation_type,
            evidence_text,
            figure_ref,
            confidence,
            review_required,
            raw_output
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            relation_id,
            doc_id,
            asset_id,
            rel["candidate_id"],
            rel.get("compound_name", ""),
            rel["component_role"],
            rel["relation_type"],
            rel["evidence_text"],
            figure_ref or "",
            rel["confidence"],
            int(rel["review_required"]),
            jdump(raw_output),
        ),
    )

    out = dict(rel)
    out.update({
        "relation_id": relation_id,
        "doc_id": doc_id,
        "asset_id": asset_id,
        "figure_ref": figure_ref or "",
    })
    return out


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield i // n, seq[i:i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--vlm-base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--vlm-model", default=os.getenv("VLLM_MODEL", "ipm-vlm"))
    ap.add_argument("--vlm-api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--source-tool", default="")
    ap.add_argument("--only-auto-pass", action="store_true")
    ap.add_argument("--no-llm-name-extract", action="store_true")
    ap.add_argument("--skip-vlm", action="store_true", help="Use label/rule-based alignment only.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-candidates-per-gallery", type=int, default=12)
    args = ap.parse_args()

    client = OpenAI(base_url=args.vlm_base_url, api_key=args.vlm_api_key)

    conn = get_conn()
    ensure_table(conn)

    if args.overwrite:
        conn.execute("DELETE FROM stg_component_relation WHERE doc_id=?", (args.doc_id,))
        conn.commit()

    candidates = load_candidates(
        conn,
        doc_id=args.doc_id,
        source_tool=args.source_tool,
        only_auto_pass=args.only_auto_pass,
        limit=args.limit or None,
    )

    asset_contexts = load_asset_contexts(conn, args.doc_id)

    compound_names = build_compound_name_list(
        conn=conn,
        doc_id=args.doc_id,
        candidates=candidates,
        client=client,
        model=args.vlm_model,
        use_llm=not args.no_llm_name_extract,
    )

    allowed_names = [x["compound_name"] for x in compound_names]
    alias_map = build_name_alias_map(compound_names)

    direct_candidates = []
    image_candidates = []

    for r in candidates:
        if is_direct_structure_candidate(r):
            direct_candidates.append(r)
        else:
            image_candidates.append(r)

    grouped = defaultdict(list)
    for r in image_candidates:
        grouped[r["asset_id"]].append(r)

    out_dir = Path("data/staging") / args.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    gallery_dir = out_dir / "compound_structure_bbox_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "compound_structure_relation.jsonl"
    compound_name_path = out_dir / "compound_name_list.json"
    report_path = out_dir / "compound_structure_relation_report.json"

    compound_name_path.write_text(
        jdump({"doc_id": args.doc_id, "compound_names": compound_names}),
        encoding="utf-8",
    )

    all_outputs = []
    stats = Counter()

    with open(jsonl_path, "w", encoding="utf-8") as fw:
        # -----------------------------------------------------------------
        # Direct CSV / XLSX / SDF / SMI / supplement-table structures:
        # no image gallery, no VLM. Directly align table row compound name
        # to candidate full structure.
        # -----------------------------------------------------------------
        for r in tqdm(direct_candidates, desc="Direct table/file compound alignment"):
            asset_id = r["asset_id"]
            asset_ctx = asset_contexts.get(asset_id, {"asset": {"asset_id": asset_id}})
            asset = asset_ctx.get("asset", {})
            figure_ref = asset.get("figure_ref") or asset.get("table_ref") or ""

            rel = direct_structure_relation(r, alias_map)

            record = insert_relation(
                conn=conn,
                doc_id=args.doc_id,
                asset_id=asset_id,
                rel=rel,
                figure_ref=figure_ref,
                raw_output={
                    "asset_id": asset_id,
                    "alignment_method": "direct_table_or_file",
                    "source": rel.get("source", ""),
                    "candidate": {
                        "candidate_id": r["candidate_id"],
                        "smiles": r["smiles"] if "smiles" in r.keys() else "",
                        "canonical_smiles": r["canonical_smiles"] if "canonical_smiles" in r.keys() else "",
                        "molecule_label": row_get(r, "molecule_label", ""),
                        "source_tool": row_get(r, "source_tool", ""),
                        "smiles_source": row_get(r, "smiles_source", ""),
                        "stage7_auto_decision": row_get(r, "auto_decision", ""),
                        "stage7_qc_score": row_get(r, "qc_score", None),
                        "stage7_structure_class": structure_class_from_candidate(r),
                    },
                    "raw_context_json": jload(row_get(r, "raw_context_json", ""), {}),
                    "raw_output": jload(row_get(r, "raw_output", ""), {}),
                },
            )

            fw.write(jdump(record) + "\n")
            all_outputs.append(record)

            stats["relations"] += 1
            stats["direct_candidates"] += 1
            stats[f"role:{record['component_role']}"] += 1
            stats[f"relation_type:{record['relation_type']}"] += 1
            stats[f"source:{record.get('source', '')}"] += 1

            if record["review_required"]:
                stats["review_required"] += 1
            else:
                stats["auto_aligned"] += 1

            conn.commit()

        for asset_id, asset_rows in tqdm(grouped.items(), desc="Asset bbox-gallery compound alignment"):
            asset_ctx = asset_contexts.get(asset_id, {"asset": {"asset_id": asset_id}})
            asset = asset_ctx.get("asset", {})
            figure_ref = asset.get("figure_ref") or asset.get("table_ref") or ""
            asset_image_path = asset.get("file_path", "")

            for page_idx, rows in chunked(asset_rows, max(1, args.max_candidates_per_gallery)):
                gallery_path, gallery_meta = make_asset_bbox_gallery(
                    asset_image_path=asset_image_path,
                    rows=rows,
                    out_dir=gallery_dir,
                    asset_id=asset_id,
                    page_index=page_idx,
                )

                raw_model_output = {}
                raw_rels = []

                if not args.skip_vlm:
                    try:
                        raw_rels, raw_model_output = align_asset_gallery_with_vlm(
                            client=client,
                            model=args.vlm_model,
                            asset_ctx=asset_ctx,
                            rows=rows,
                            allowed_names=allowed_names,
                            alias_map=alias_map,
                            gallery_path=gallery_path,
                            gallery_meta=gallery_meta,
                        )
                    except Exception as e:
                        raw_model_output = {
                            "error": str(e),
                            "asset_id": asset_id,
                            "gallery_path": gallery_path,
                        }
                        raw_rels = []

                candidate_id_set = {r["candidate_id"] for r in rows}
                bbox_to_candidate = {m["bbox_id"]: m["candidate_id"] for m in gallery_meta}

                normalized = []
                for rel in raw_rels:
                    nr = normalize_relation(
                        rel,
                        candidate_id_set=candidate_id_set,
                        bbox_to_candidate=bbox_to_candidate,
                        alias_map=alias_map,
                    )
                    if nr:
                        normalized.append(nr)

                by_candidate = defaultdict(list)
                for rel in normalized:
                    by_candidate[rel["candidate_id"]].append(rel)

                # Rule fallback per candidate, essential for ChemEAGLE outputs with labels but weak/no bbox.
                for row in rows:
                    cid = row["candidate_id"]
                    if not by_candidate[cid] or all(not x.get("compound_name") for x in by_candidate[cid]):
                        for rel in rule_based_relations(row, alias_map):
                            nr = normalize_relation(
                                rel,
                                candidate_id_set=candidate_id_set,
                                bbox_to_candidate=bbox_to_candidate,
                                alias_map=alias_map,
                            )
                            if nr:
                                normalized.append(nr)
                                by_candidate[cid].append(nr)

                for row in rows:
                    cid = row["candidate_id"]
                    if not by_candidate[cid]:
                        normalized.append(fallback_relation(row))

                seen = set()
                deduped = []
                for rel in normalized:
                    key = (
                        rel.get("candidate_id"),
                        rel.get("compound_name"),
                        rel.get("component_role"),
                        rel.get("relation_type"),
                    )
                    if key not in seen:
                        seen.add(key)
                        deduped.append(rel)

                for rel in deduped:
                    row_lookup = {r["candidate_id"]: r for r in rows}
                    r = row_lookup.get(rel["candidate_id"])
                    raw = jload(r["raw_output"], {}) if r is not None else {}

                    record = insert_relation(
                        conn=conn,
                        doc_id=args.doc_id,
                        asset_id=asset_id,
                        rel=rel,
                        figure_ref=figure_ref,
                        raw_output={
                            "asset_id": asset_id,
                            "gallery_path": gallery_path,
                            "gallery_meta": gallery_meta,
                            "vlm_output": raw_model_output,
                            "allowed_compound_names": allowed_names,
                            "candidate": {
                                "candidate_id": rel["candidate_id"],
                                "bbox_id": rel.get("bbox_id", ""),
                                "smiles": r["smiles"] if r is not None else "",
                                "canonical_smiles": r["canonical_smiles"] if r is not None else "",
                                "molecule_label": candidate_label(raw),
                                "stage7_auto_decision": r["auto_decision"] if r is not None else "",
                                "stage7_qc_score": r["qc_score"] if r is not None else None,
                                "stage7_structure_class": structure_class_from_candidate(r) if r is not None else "",
                            },
                        },
                    )

                    fw.write(jdump(record) + "\n")
                    all_outputs.append(record)

                    stats["relations"] += 1
                    stats[f"role:{record['component_role']}"] += 1
                    stats[f"relation_type:{record['relation_type']}"] += 1
                    stats[f"source:{record.get('source', '')}"] += 1
                    if record["review_required"]:
                        stats["review_required"] += 1
                    else:
                        stats["auto_aligned"] += 1

                conn.commit()

    conn.close()

    report = {
        "doc_id": args.doc_id,
        "num_candidates": len(candidates),
        "num_direct_candidates": len(direct_candidates),
        "num_image_candidates": len(image_candidates),
        "num_assets": len(grouped),
        "num_compound_names": len(allowed_names),
        "num_relations": len(all_outputs),
        "max_candidates_per_gallery": args.max_candidates_per_gallery,
        "stats": dict(stats),
        "compound_name_list": str(compound_name_path),
        "jsonl": str(jsonl_path),
        "gallery_dir": str(gallery_dir),
        "table": "stg_component_relation",
        "note": (
            "Direct CSV/XLSX/SDF/SMI structures are aligned without VLM. Image-derived candidates use asset-level bbox-gallery alignment. Each VLM call receives one composite image: original asset with highlighted bboxes "
            "plus a right-side candidate gallery containing bbox_id, candidate_id, source crop, RDKit render, and labels. "
            "Compound names are restricted to extracted paper-native names; label-based fallback is used for ChemEAGLE outputs."
        ),
    }

    report_path.write_text(jdump(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
