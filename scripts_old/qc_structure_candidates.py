#!/usr/bin/env python3
import os
import re
import sys
import json
import base64
import argparse
from pathlib import Path
from collections import Counter

from tqdm import tqdm
from PIL import Image, ImageDraw
import xlsxwriter
from rdkit import Chem
from rdkit.Chem import Draw
from openai import OpenAI
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn


def jload(s, default=None):
    try:
        return json.loads(s or "")
    except Exception:
        return default if default is not None else {}


def clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def jdump(x):
    return json.dumps(x, ensure_ascii=False, default=str)


def ensure_table(conn):
    conn.execute("""
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
    )
    """)
    conn.commit()


def has_variable_label(label):
    label = label or ""

    if re.search(r"\b[RXYZ]\d*\s*=", label, re.I):
        return True

    if re.search(r"\bn\s*=", label, re.I):
        return True

    if re.search(r"\bm\s*=", label, re.I):
        return True

    if re.search(r"\b\d+[a-z]\s*:", label, re.I):
        return True

    if "||" in label:
        return True

    return False


def has_r_placeholder(smiles):
    s = smiles or ""

    if re.search(r"\[[^\]]*R[^\]]*\]", s):
        return True

    if re.search(r"(^|[^A-Za-z])R\d*([^A-Za-z]|$)", s):
        return True

    return False


def classify_fragments(mol, flags):
    info = {
        "fragment_count": 1,
        "fragment_heavy_atom_counts": [],
        "fragment_classes": [],
    }

    if mol is None:
        return info

    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception:
        return info

    info["fragment_count"] = len(frags)

    if len(frags) <= 1:
        return info

    salt_like_atoms = {
        3,   # Li
        4,   # Be
        9,   # F
        11,  # Na
        12,  # Mg
        17,  # Cl
        19,  # K
        20,  # Ca
        35,  # Br
        53,  # I
    }

    small_counterion_count = 0
    large_fragment_count = 0
    dummy_fragment_count = 0

    for frag in frags:
        atoms = list(frag.GetAtoms())
        heavy = sum(1 for a in atoms if a.GetAtomicNum() > 1)
        atomic_nums = {a.GetAtomicNum() for a in atoms}
        formal_charge = sum(a.GetFormalCharge() for a in atoms)

        info["fragment_heavy_atom_counts"].append(heavy)

        has_dummy = any(a.GetAtomicNum() == 0 for a in atoms)
        if has_dummy:
            dummy_fragment_count += 1

        is_small_counterion = (
            heavy <= 3
            and (
                atomic_nums.issubset(salt_like_atoms)
                or formal_charge != 0
            )
        )

        if is_small_counterion:
            small_counterion_count += 1

        if heavy >= 6:
            large_fragment_count += 1

    if small_counterion_count > 0:
        flags.append("salt_or_counterion_like")
        info["fragment_classes"].append("salt_or_counterion_like")

    if dummy_fragment_count > 0:
        flags.append("component_or_attachment_fragment")
        info["fragment_classes"].append("component_or_attachment_fragment")

    if large_fragment_count >= 2:
        flags.append("multi_large_fragments")
        info["fragment_classes"].append("multi_large_fragments")

    if not info["fragment_classes"]:
        flags.append("multicomponent_or_fragmented")
        info["fragment_classes"].append("multicomponent_or_fragmented")

    return info


def rdkit_qc(smiles, label=""):
    flags = []
    s = smiles or ""

    out = {
        "rdkit_valid": False,
        "canonical_smiles": "",
        "rdkit_error": "",
        "flags": flags,
        "fragment_info": {
            "fragment_count": 0,
            "fragment_heavy_atom_counts": [],
            "fragment_classes": [],
        },
    }

    if not s.strip():
        flags.append("empty_smiles")
        out["rdkit_error"] = "empty smiles"
        return out

    if "*" in s:
        flags.append("dummy_atom")

    if has_r_placeholder(s):
        flags.append("unresolved_placeholder")

    if has_variable_label(label):
        flags.append("variable_or_series_label")

    if "." in s:
        flags.append("fragment_separator")

    try:
        mol = Chem.MolFromSmiles(s, sanitize=False)

        if mol is None:
            if "unresolved_placeholder" in flags or "variable_or_series_label" in flags:
                flags.append("non_rdkit_markush_notation")
                out["rdkit_error"] = "non-RDKit Markush/R placeholder notation"
            else:
                flags.append("invalid_smiles")
                out["rdkit_error"] = "MolFromSmiles returned None"
            return out

        if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            flags.append("dummy_atom")

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            if "dummy_atom" in flags or "unresolved_placeholder" in flags or "variable_or_series_label" in flags:
                flags.append("nonstandard_component_or_markush")
                out["rdkit_error"] = str(e)
            else:
                flags.append("sanitize_or_valence_error")
                out["rdkit_error"] = str(e)
            return out

        out["rdkit_valid"] = True
        out["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)
        out["fragment_info"] = classify_fragments(mol, flags)

    except Exception as e:
        flags.append("rdkit_exception")
        out["rdkit_error"] = str(e)

    out["flags"] = sorted(set(flags))
    return out


def structure_class_from_qc(qc, label=""):
    flags = set(qc.get("flags", []))

    if "empty_smiles" in flags:
        return "empty"

    if (
        "invalid_smiles" in flags
        or "sanitize_or_valence_error" in flags
        or "rdkit_exception" in flags
    ):
        return "invalid_structure"

    if (
        "variable_or_series_label" in flags
        or "unresolved_placeholder" in flags
        or "non_rdkit_markush_notation" in flags
    ):
        return "markush_or_variable"

    if "dummy_atom" in flags or "component_or_attachment_fragment" in flags:
        return "component_or_attachment"

    if "fragment_separator" in flags:
        if "salt_or_counterion_like" in flags:
            return "salt_or_counterion"
        if "multi_large_fragments" in flags:
            return "multicomponent_or_merged"
        return "fragment_or_multicomponent"

    return "full_molecule"


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


def expand_xyxy(xyxy, w, h, pad_ratio=0.30, min_pad=80):
    """
    Expand bbox to include nearby compound labels / component names.
    This expanded crop is used for VLM and downstream alignment.
    """
    if not xyxy:
        return None

    x0, y0, x1, y1 = xyxy
    bw = x1 - x0
    bh = y1 - y0

    pad = max(min_pad, int(max(bw, bh) * pad_ratio))

    return [
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(w, x1 + pad),
        min(h, y1 + pad),
    ]


def draw_corner_marker(draw, xyxy, color=(255, 0, 0), width=3, corner_len=35):
    """
    Draw only corner markers instead of a full rectangle.
    This avoids covering compound labels near the structure.
    """
    x0, y0, x1, y1 = xyxy

    corner_len = min(
        corner_len,
        max(8, int((x1 - x0) * 0.20)),
        max(8, int((y1 - y0) * 0.20)),
    )

    # top-left
    draw.line([(x0, y0), (x0 + corner_len, y0)], fill=color, width=width)
    draw.line([(x0, y0), (x0, y0 + corner_len)], fill=color, width=width)

    # top-right
    draw.line([(x1, y0), (x1 - corner_len, y0)], fill=color, width=width)
    draw.line([(x1, y0), (x1, y0 + corner_len)], fill=color, width=width)

    # bottom-left
    draw.line([(x0, y1), (x0 + corner_len, y1)], fill=color, width=width)
    draw.line([(x0, y1), (x0, y1 - corner_len)], fill=color, width=width)

    # bottom-right
    draw.line([(x1, y1), (x1 - corner_len, y1)], fill=color, width=width)
    draw.line([(x1, y1), (x1, y1 - corner_len)], fill=color, width=width)


def make_crop_and_overlay(image_path, bbox, out_dir, candidate_id):
    """
    Outputs:
    - crop_path: expanded context crop WITHOUT any bbox drawing.
                 Use this for VLM / downstream compound alignment.
    - overlay_path: full image with non-blocking corner markers.
                    Use this only for human review.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path or not Path(image_path).exists():
        return "", ""

    im = Image.open(image_path).convert("RGB")
    xyxy = bbox_to_xyxy(bbox, im.width, im.height)

    overlay_path = out_dir / f"{candidate_id}_overlay_corner.png"
    crop_path = out_dir / f"{candidate_id}_context_crop.png"
    focus_crop_path = out_dir / f"{candidate_id}_focus_crop.png"

    if xyxy:
        # 1. Expanded no-box crop for VLM and stage 8.
        context_xyxy = expand_xyxy(
            xyxy,
            im.width,
            im.height,
            pad_ratio=0.30,
            min_pad=80,
        )
        context_crop = im.crop(context_xyxy)
        context_crop.save(crop_path)

        # 2. Exact crop without any drawing, useful for debugging.
        focus_crop = im.crop(xyxy)
        focus_crop.save(focus_crop_path)

        # 3. Human review overlay: corner markers only.
        overlay = im.copy()
        draw = ImageDraw.Draw(overlay)
        draw_corner_marker(
            draw,
            xyxy,
            color=(255, 0, 0),
            width=3,
            corner_len=35,
        )
        overlay.save(overlay_path)

    else:
        # No bbox: use original image, no drawing.
        im.save(crop_path)
        im.save(overlay_path)

    return str(crop_path), str(overlay_path)

def render_smiles(smiles, out_path):
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Draw.MolToImage(mol, size=(520, 360))
    img.save(out_path)
    return str(out_path)


def image_to_data_url(path):
    p = Path(path)
    if not p.exists():
        return None

    mime = "image/png"
    if p.suffix.lower() in [".jpg", ".jpeg"]:
        mime = "image/jpeg"

    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"



IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"
}

DIRECT_STRUCTURE_SOURCE_TOOLS = {
    "supplement_table_direct",
    "supplement_direct_smiles",
    "sdf_direct",
    "smi_direct",
    "csv_direct",
    "xlsx_direct",
}


def row_get(row, key, default=""):
    try:
        v = row[key]
        return default if v is None else v
    except Exception:
        return default


def is_image_file(path):
    if not path:
        return False

    p = Path(str(path))
    return p.exists() and p.suffix.lower() in IMAGE_SUFFIXES


def is_direct_structure_candidate(row, raw):
    source_tool = str(row_get(row, "source_tool", "") or "")
    smiles_source = str(row_get(row, "smiles_source", "") or "")
    asset_type = str(row_get(row, "asset_type", "") or "")
    image_path = str(row_get(row, "image_path", "") or row_get(row, "asset_file_path", "") or "")
    suffix = Path(image_path).suffix.lower() if image_path else ""

    raw_text = json.dumps(raw or {}, ensure_ascii=False, default=str).lower()

    if source_tool in DIRECT_STRUCTURE_SOURCE_TOOLS:
        return True

    if "supplement_table_direct" in raw_text:
        return True

    if "direct" in source_tool.lower() and smiles_source.lower() in {"smiles", "inchi", "molblock", "sdf", "smi"}:
        return True

    if suffix in {".csv", ".tsv", ".tab", ".xlsx", ".xlsm", ".sdf", ".smi", ".smiles"}:
        return True

    if asset_type in {"supplementary_table"} and smiles_source.lower() in {"smiles", "inchi", "molblock"}:
        return True

    return False


def decide_without_vlm(qc, label="", direct_structure=False):
    flags = list(qc["flags"])
    structure_class = structure_class_from_qc(qc, label=label)

    vlm_qc = {
        "visual_match": "not_applicable",
        "structure_class": structure_class,
        "confidence": 0,
        "reason": "VLM skipped for direct table/file structure",
    }

    if not qc["rdkit_valid"]:
        if structure_class == "markush_or_variable":
            return (
                45,
                sorted(set(flags + ["direct_structure_no_vlm", "no_rdkit_render"])),
                "review_direct_markush_or_variable",
                structure_class,
                vlm_qc,
            )

        if structure_class == "empty":
            return (
                0,
                sorted(set(flags + ["direct_structure_no_vlm"])),
                "review_empty_smiles",
                structure_class,
                vlm_qc,
            )

        return (
            0,
            sorted(set(flags + ["direct_structure_no_vlm"])),
            "review_invalid_direct_structure",
            structure_class,
            vlm_qc,
        )

    if direct_structure and structure_class == "full_molecule":
        return (
            98,
            sorted(set(flags + ["direct_table_structure", "rdkit_valid", "vlm_skipped"])),
            "auto_pass_direct_table_full_molecule",
            structure_class,
            vlm_qc,
        )

    if direct_structure:
        return (
            75,
            sorted(set(flags + ["direct_table_structure", "rdkit_valid", "direct_structure_not_full_molecule", "vlm_skipped"])),
            f"review_direct_table_{structure_class}",
            structure_class,
            vlm_qc,
        )

    return (
        60,
        sorted(set(flags + ["non_image_asset", "rdkit_valid", "visual_qc_skipped"])),
        "review_non_image_asset_no_visual_qc",
        structure_class,
        vlm_qc,
    )

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

    return {
        "visual_match": "uncertain",
        "structure_class": "uncertain",
        "confidence": 0,
        "reason": "invalid json from vlm",
    }


def normalize_vlm_result(obj):
    if obj.get("visual_match") not in ["yes", "no", "uncertain"]:
        obj["visual_match"] = "uncertain"

    valid_classes = {
        "full_molecule",
        "component_or_attachment",
        "markush_or_variable",
        "salt_or_counterion",
        "fragment_or_multicomponent",
        "multicomponent_or_merged",
        "invalid_or_not_match",
        "uncertain",
    }

    if obj.get("structure_class") not in valid_classes:
        obj["structure_class"] = "uncertain"

    try:
        obj["confidence"] = float(obj.get("confidence", 0) or 0)
    except Exception:
        obj["confidence"] = 0

    obj["reason"] = str(obj.get("reason", ""))[:300]
    return obj



def normalize_numeric_repeat_result(obj):
    """
    Normalize targeted VLM output for UniParser's known failure mode:
    fixed numeric repeat abbreviations such as (OCH2CH2)4, [CH2CH2O]3, PEG4.
    """
    if not isinstance(obj, dict):
        obj = {}

    has_repeat = obj.get("has_fixed_numeric_repeat")
    if isinstance(has_repeat, str):
        has_repeat = has_repeat.strip().lower() in {"true", "yes", "1", "y"}
    else:
        has_repeat = bool(has_repeat)

    should_fail = obj.get("should_fail_uniparser")
    if isinstance(should_fail, str):
        should_fail = should_fail.strip().lower() in {"true", "yes", "1", "y"}
    else:
        should_fail = bool(should_fail)

    notation = clean(obj.get("repeat_notation"))
    valid_notations = {
        "bracket_subscript",
        "parenthesis_subscript",
        "n_equals_fixed",
        "m_equals_fixed",
        "peg_number",
        "x_number",
        "other",
        "none",
    }
    if notation not in valid_notations:
        notation = "other" if has_repeat else "none"

    try:
        conf = float(obj.get("confidence", 0) or 0)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    return {
        "has_fixed_numeric_repeat": has_repeat,
        "repeat_count": clean(obj.get("repeat_count"))[:40],
        "repeat_notation": notation,
        "repeat_region_description": clean(obj.get("repeat_region_description"))[:300],
        "should_fail_uniparser": should_fail or has_repeat,
        "confidence": conf,
        "reason": clean(obj.get("reason"))[:300],
    }


def vlm_check_uniparser_fixed_numeric_repeat(client, model, paper_img, smiles="", label=""):
    """
    Targeted UniParser QC.

    This intentionally does NOT compare the full structure with RDKit rendering.
    It only asks whether the original crop contains a fixed numeric repeat abbreviation,
    which is a known UniParser failure mode.
    """
    u1 = image_to_data_url(paper_img)
    if not u1:
        return normalize_numeric_repeat_result({
            "has_fixed_numeric_repeat": False,
            "repeat_count": "",
            "repeat_notation": "none",
            "repeat_region_description": "",
            "should_fail_uniparser": False,
            "confidence": 0,
            "reason": "missing candidate crop image",
        })

    prompt = f"""
You are checking ONE known UniParser molecule-recognition failure mode.

Image 1 is a crop of a molecule from the original medicinal chemistry paper.

Question:
Does this molecule drawing contain an abbreviated repeating unit with an explicit fixed numeric repeat count?

Positive examples:
- a bracketed or parenthesized linker segment with subscript 2, 3, 4, 5, etc.
- [CH2CH2O]4, (OCH2CH2)3, (CH2)5, -[OCH2CH2]4-
- PEG2, PEG3, PEG4, PEG5 when used as a structural repeated linker/unit
- n=4, m=3, x=2 when it defines the fixed repeat count of a drawn repeated unit
- x4 or similar numeric repeat count next to a bracketed structural segment

Negative examples:
- compound number 4 or compound label 4a
- atom numbering, ring numbering, figure numbering, reaction step numbering
- R1/R2/R3 substituent labels
- variable n or m without a fixed numeric value
- a fully explicitly drawn chain with no bracket/parenthesis/PEG/repeat abbreviation
- generic labels like PROTACs, HyT molecules, compounds, analogs

Candidate SMILES from UniParser:
{smiles}

Nearby molecule label:
{label}

Return compact JSON only.
Schema:
{{
  "has_fixed_numeric_repeat": true|false,
  "repeat_count": "",
  "repeat_notation": "bracket_subscript|parenthesis_subscript|n_equals_fixed|m_equals_fixed|peg_number|x_number|other|none",
  "repeat_region_description": "brief visible region description",
  "should_fail_uniparser": true|false,
  "confidence": 0.0,
  "reason": "max 30 words"
}}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": u1}},
                ],
            }],
        )
        return normalize_numeric_repeat_result(extract_json(resp.choices[0].message.content))
    except Exception as e:
        return normalize_numeric_repeat_result({
            "has_fixed_numeric_repeat": False,
            "repeat_count": "",
            "repeat_notation": "none",
            "repeat_region_description": "",
            "should_fail_uniparser": False,
            "confidence": 0,
            "reason": f"VLM numeric repeat check failed: {e}",
        })


def decide_uniparser_repeat_qc(
    qc,
    repeat_qc,
    label="",
    fail_threshold=0.80,
    possible_threshold=0.50,
):
    """
    UniParser-specific QC decision.

    Only targeted VLM failure mode:
    - fixed numeric repeat abbreviation visible in the original crop.

    Other visual-consistency issues are deliberately not checked here.
    """
    flags = list(qc.get("flags", []))
    structure_class = structure_class_from_qc(qc, label=label)

    if not qc.get("rdkit_valid"):
        if structure_class == "markush_or_variable":
            return (
                45,
                sorted(set(flags + ["uniparser", "no_rdkit_render"])),
                "review_uniparser_markush_or_variable",
                structure_class,
            )

        if structure_class == "empty":
            return (
                0,
                sorted(set(flags + ["uniparser"])),
                "review_uniparser_empty_smiles",
                structure_class,
            )

        return (
            0,
            sorted(set(flags + ["uniparser"])),
            "review_uniparser_invalid_structure",
            structure_class,
        )

    has_repeat = bool(repeat_qc.get("has_fixed_numeric_repeat"))
    should_fail = bool(repeat_qc.get("should_fail_uniparser"))
    conf = float(repeat_qc.get("confidence", 0) or 0)

    if has_repeat and should_fail and conf >= fail_threshold:
        return (
            20,
            sorted(set(flags + [
                "uniparser",
                "uniparser_fixed_numeric_repeat_abbreviation",
                "known_uniparser_repeat_failure",
                f"repeat_notation:{repeat_qc.get('repeat_notation', 'unknown')}",
            ])),
            "review_uniparser_fixed_numeric_repeat_error",
            structure_class,
        )

    if has_repeat and conf >= possible_threshold:
        return (
            45,
            sorted(set(flags + [
                "uniparser",
                "possible_fixed_numeric_repeat_abbreviation",
                f"repeat_notation:{repeat_qc.get('repeat_notation', 'unknown')}",
            ])),
            "review_uniparser_possible_fixed_numeric_repeat",
            structure_class,
        )

    return (
        98,
        sorted(set(flags + [
            "uniparser",
            "rdkit_valid",
            "targeted_repeat_qc_pass",
            "no_fixed_numeric_repeat_detected",
            f"class:{structure_class}",
        ])),
        "auto_pass_uniparser_valid_no_fixed_numeric_repeat",
        structure_class,
    )


def vlm_check(client, model, paper_img, rendered_img, smiles, label, qc_flags, structure_class):
    u1 = image_to_data_url(paper_img)
    u2 = image_to_data_url(rendered_img)

    if not u1 or not u2:
        return {
            "visual_match": "uncertain",
            "structure_class": structure_class or "uncertain",
            "confidence": 0,
            "reason": "missing image",
        }

    prompt = f"""
You are checking molecule structure recognition for real medicinal chemistry / IPM papers.

Image 1 is a crop from the original paper figure. It may contain:
- one full molecule
- several molecules in the same figure
- a component such as warhead, linker, E3 ligand, binder, handle, or attachment fragment
- a Markush/R-group/variable template
- salt/counterion or disconnected components

Image 2 is the RDKit rendering of the candidate SMILES.

Candidate SMILES:
{smiles}

Candidate label or nearby text:
{label}

QC flags:
{json.dumps(qc_flags, ensure_ascii=False)}

Rule:
- Do NOT reject only because the candidate has "*", dummy atoms, R-group-like notation, or disconnected fragments.
- Do NOT reject only because the original figure contains multiple molecules.
- Answer yes if Image 2 visually matches one corresponding molecule/component/template in Image 1.
- Focus on atom connectivity, ring systems, scaffold, linker length, major substituents, and attachment points.
- If Image 2 is only a component or Markush template, classify it correctly instead of rejecting it.
- Ignore drawing style, orientation, font, and minor layout differences.

Return compact JSON only.

Schema:
{{
  "visual_match": "yes|no|uncertain",
  "structure_class": "full_molecule|component_or_attachment|markush_or_variable|salt_or_counterion|fragment_or_multicomponent|multicomponent_or_merged|invalid_or_not_match|uncertain",
  "confidence": 0.0,
  "reason": "max 30 words"
}}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": u1}},
                {"type": "image_url", "image_url": {"url": u2}},
            ],
        }],
    )

    return normalize_vlm_result(extract_json(resp.choices[0].message.content))


def decision_name_for_class(structure_class, conf_level):
    if structure_class == "full_molecule":
        return f"auto_pass_full_molecule_{conf_level}"

    if structure_class == "component_or_attachment":
        return f"auto_pass_component_candidate_{conf_level}"

    if structure_class == "markush_or_variable":
        return f"auto_pass_markush_candidate_{conf_level}"

    if structure_class == "salt_or_counterion":
        return f"auto_pass_salt_candidate_{conf_level}"

    if structure_class in ["fragment_or_multicomponent", "multicomponent_or_merged"]:
        return f"auto_pass_multicomponent_candidate_{conf_level}"

    return f"auto_pass_structure_candidate_{conf_level}"


def review_name_for_class(structure_class):
    if structure_class == "empty":
        return "review_empty_smiles"

    if structure_class == "invalid_structure":
        return "review_invalid_structure"

    if structure_class == "markush_or_variable":
        return "review_markush_or_variable"

    if structure_class == "component_or_attachment":
        return "review_component_or_attachment"

    if structure_class == "salt_or_counterion":
        return "review_salt_or_counterion"

    if structure_class in ["fragment_or_multicomponent", "multicomponent_or_merged"]:
        return "review_fragment_or_multicomponent"

    return "review_required"


def decide(qc, vlm, label=""):
    flags = list(qc["flags"])
    structure_class = structure_class_from_qc(qc, label=label)

    if not qc["rdkit_valid"]:
        if structure_class == "markush_or_variable":
            return 45, sorted(set(flags + ["no_rdkit_render"])), "review_markush_or_variable", structure_class

        if structure_class == "empty":
            return 0, sorted(set(flags)), "review_empty_smiles", structure_class

        return 0, sorted(set(flags)), "review_invalid_structure", structure_class

    visual_match = vlm.get("visual_match", "uncertain")
    confidence = float(vlm.get("confidence", 0) or 0)

    vlm_class = vlm.get("structure_class") or "uncertain"
    if vlm_class not in ["uncertain", "invalid_or_not_match"]:
        structure_class = vlm_class

    if visual_match == "yes" and confidence >= 0.80:
        return (
            95,
            sorted(set(flags + ["vlm_visual_match", f"class:{structure_class}"])),
            decision_name_for_class(structure_class, "high_conf"),
            structure_class,
        )

    if visual_match == "yes" and confidence >= 0.60:
        return (
            80,
            sorted(set(flags + ["vlm_visual_match_low_conf", f"class:{structure_class}"])),
            decision_name_for_class(structure_class, "medium_conf"),
            structure_class,
        )

    if visual_match == "uncertain":
        return (
            55,
            sorted(set(flags + ["vlm_uncertain", f"class:{structure_class}"])),
            review_name_for_class(structure_class),
            structure_class,
        )

    return (
        30,
        sorted(set(flags + ["vlm_visual_mismatch", f"class:{structure_class}"])),
        "review_visual_mismatch",
        structure_class,
    )


def load_rows(conn, doc_id):
    return conn.execute(
        """
        SELECT
            c.*,
            a.asset_type,
            a.file_path AS asset_file_path,
            a.page_no,
            a.figure_ref,
            a.table_ref
        FROM stg_structure_candidate c
        LEFT JOIN raw_asset a ON c.asset_id = a.asset_id
        WHERE c.doc_id=?
        ORDER BY c.asset_id, c.candidate_index
        """,
        (doc_id,),
    ).fetchall()


def write_xlsx(rows, xlsx_path):
    wb = xlsxwriter.Workbook(str(xlsx_path))
    ws = wb.add_worksheet("structure_candidates")
    sm = wb.add_worksheet("summary")

    header_fmt = wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    wrap_fmt = wb.add_format({"text_wrap": True, "valign": "top"})
    pass_fmt = wb.add_format({"bg_color": "#E2F0D9", "text_wrap": True, "valign": "top"})
    review_fmt = wb.add_format({"bg_color": "#FFF2CC", "text_wrap": True, "valign": "top"})
    bad_fmt = wb.add_format({"bg_color": "#F4CCCC", "text_wrap": True, "valign": "top"})

    headers = [
        "paper_crop",
        "rdkit_render",
        "bbox_overlay",
        "candidate_id",
        "asset_id",
        "candidate_index",
        "molecule_label",
        "structure_class",
        "smiles",
        "canonical_smiles",
        "qc_score",
        "auto_decision",
        "qc_flags",
        "vlm_visual_match",
        "vlm_structure_class",
        "vlm_confidence",
        "vlm_reason",
        "repeat_has_fixed_numeric_repeat",
        "repeat_count",
        "repeat_notation",
        "repeat_confidence",
        "repeat_reason",
        "human_decision",
        "human_corrected_smiles",
        "human_note",
    ]

    for i, h in enumerate(headers):
        ws.write(0, i, h, header_fmt)

    widths = [18, 18, 18, 24, 24, 10, 28, 24, 45, 45, 10, 34, 42, 16, 24, 12, 35, 18, 12, 18, 12, 35, 18, 35, 35]
    for i, w in enumerate(widths):
        ws.set_column(i, i, w)

    ws.freeze_panes(1, 3)
    ws.autofilter(0, 0, max(1, len(rows)), len(headers) - 1)

    for r, row in enumerate(rows, start=1):
        ws.set_row(r, 110)

        for key, col in [
            ("crop_path", 0),
            ("render_path", 1),
            ("overlay_path", 2),
        ]:
            p = row.get(key)
            if p and Path(p).exists():
                ws.insert_image(r, col, p, {"x_scale": 0.25, "y_scale": 0.25})
            else:
                ws.write(r, col, "")

        decision = row["auto_decision"]
        if decision.startswith("auto_pass"):
            fmt = pass_fmt
        elif "invalid" in decision or "mismatch" in decision or "empty" in decision:
            fmt = bad_fmt
        else:
            fmt = review_fmt

        repeat_qc = (row.get("vlm_qc") or {}).get("numeric_repeat_qc") or {}

        values = [
            row["candidate_id"],
            row["asset_id"],
            row["candidate_index"],
            row["molecule_label"],
            row["structure_class"],
            row["smiles"],
            row["canonical_smiles"],
            row["qc_score"],
            row["auto_decision"],
            "||".join(row["qc_flags"]),
            row["vlm_qc"].get("visual_match", ""),
            row["vlm_qc"].get("structure_class", ""),
            row["vlm_qc"].get("confidence", ""),
            row["vlm_qc"].get("reason", ""),
            repeat_qc.get("has_fixed_numeric_repeat", ""),
            repeat_qc.get("repeat_count", ""),
            repeat_qc.get("repeat_notation", ""),
            repeat_qc.get("confidence", ""),
            repeat_qc.get("reason", ""),
            "",
            "",
            "",
        ]

        for i, v in enumerate(values, start=3):
            ws.write(r, i, v, fmt)

    ws.data_validation(
        1, 22, max(1, len(rows)), 22,
        {
            "validate": "list",
            "source": ["accept", "reject", "corrected", "uncertain"],
        },
    )

    sm.write(0, 0, "auto_decision", header_fmt)
    sm.write(0, 1, "count", header_fmt)

    counts = Counter(x["auto_decision"] for x in rows)
    for i, (k, v) in enumerate(counts.items(), start=1):
        sm.write(i, 0, k)
        sm.write(i, 1, v)

    sm.write(0, 3, "structure_class", header_fmt)
    sm.write(0, 4, "count", header_fmt)

    class_counts = Counter(x["structure_class"] for x in rows)
    for i, (k, v) in enumerate(class_counts.items(), start=1):
        sm.write(i, 3, k)
        sm.write(i, 4, v)

    sm.write(0, 6, "qc_flag", header_fmt)
    sm.write(0, 7, "count", header_fmt)

    flags = Counter(f for x in rows for f in x["qc_flags"])
    for i, (k, v) in enumerate(flags.items(), start=1):
        sm.write(i, 6, k)
        sm.write(i, 7, v)

    wb.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--vlm-base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--vlm-model", default=os.getenv("VLLM_MODEL", "ipm-vlm"))
    ap.add_argument("--vlm-api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--uniparser-repeat-fail-threshold", type=float, default=0.80)
    ap.add_argument("--uniparser-repeat-possible-threshold", type=float, default=0.50)
    args = ap.parse_args()

    client = OpenAI(base_url=args.vlm_base_url, api_key=args.vlm_api_key)

    conn = get_conn()
    (conn)

    review_dir = Path("data/review") / args.doc_id / "structure_vlm"
    img_dir = review_dir / "structure_candidate_images"
    raw_dir = review_dir / "raw_outputs"
    img_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(conn, args.doc_id)
    if args.limit:
        rows = rows[:args.limit]

    output = []

    for r in tqdm(rows, desc="VLM structure QC"):
        cid = r["candidate_id"]
        asset_id = r["asset_id"]
        smiles = r["smiles"] or ""

        raw = jload(r["raw_output"], {})
        raw_path = raw_dir / f"{cid}.json"
        raw_path.write_text(jdump(raw), encoding="utf-8")

        label = (
            raw.get("molecule_label") 
            or raw.get("label")
            or "||".join(raw.get("molecule_texts", []) or [])
            or ""
        ) if isinstance(raw, dict) else ""

        bbox = find_bbox(raw)
        image_path = r["image_path"] or r["asset_file_path"] or ""

        cand_dir = img_dir / cid
        cand_dir.mkdir(parents=True, exist_ok=True)

        qc = rdkit_qc(smiles.split("<sep>")[0], label=label)
        structure_class = structure_class_from_qc(qc, label=label)

        crop_path = ""
        overlay_path = ""
        render_path = ""

        direct_structure = is_direct_structure_candidate(r, raw)
        image_asset = is_image_file(image_path)

        source_tool = str(row_get(r, "source_tool", "") or "").lower()
        is_uniparser = source_tool == "uniparser"

        # UniParser-specific targeted QC:
        # Only check the known failure mode: fixed numeric repeat abbreviations.
        # Do not run general paper-vs-RDKit visual matching for UniParser.
        if is_uniparser and image_asset:
            crop_path = str(image_path)
            overlay_path = str(image_path)

            if qc["rdkit_valid"]:
                render_path = render_smiles(
                    qc["canonical_smiles"] or smiles,
                    cand_dir / f"{cid}_rdkit.png",
                )

            repeat_qc = vlm_check_uniparser_fixed_numeric_repeat(
                client=client,
                model=args.vlm_model,
                paper_img=crop_path,
                smiles=qc["canonical_smiles"] or smiles,
                label=label,
            )

            qc_score, qc_flags, decision, structure_class = decide_uniparser_repeat_qc(
                qc=qc,
                repeat_qc=repeat_qc,
                label=label,
                fail_threshold=args.uniparser_repeat_fail_threshold,
                possible_threshold=args.uniparser_repeat_possible_threshold,
            )

            vlm_qc = {
                "visual_match": "not_checked",
                "structure_class": structure_class,
                "confidence": repeat_qc.get("confidence", 0),
                "reason": "UniParser targeted fixed numeric repeat QC only.",
                "numeric_repeat_qc": repeat_qc,
                "general_visual_qc": None,
            }

        # CSV / XLSX / SDF / SMI / direct supplementary structures:
        # RDKit-only QC, no Image.open, no VLM.
        elif direct_structure or not image_asset:
            if qc["rdkit_valid"]:
                render_path = render_smiles(
                    qc["canonical_smiles"] or smiles,
                    cand_dir / f"{cid}_rdkit.png",
                )

            qc_score, qc_flags, decision, structure_class, vlm_qc = decide_without_vlm(
                qc=qc,
                label=label,
                direct_structure=direct_structure,
            )

        else:
            crop_path, overlay_path = make_crop_and_overlay(
                image_path=image_path,
                bbox=bbox,
                out_dir=cand_dir,
                candidate_id=cid,
            )

            vlm_qc = {
                "visual_match": "uncertain",
                "structure_class": structure_class,
                "confidence": 0,
                "reason": "not checked",
            }

            if qc["rdkit_valid"]:
                render_path = render_smiles(
                    qc["canonical_smiles"] or smiles,
                    cand_dir / f"{cid}_rdkit.png",
                )

                if render_path:
                    vlm_qc = vlm_check(
                        client=client,
                        model=args.vlm_model,
                        paper_img=crop_path,
                        rendered_img=render_path,
                        smiles=qc["canonical_smiles"] or smiles,
                        label=label,
                        qc_flags=qc["flags"],
                        structure_class=structure_class,
                    )

            qc_score, qc_flags, decision, structure_class = decide(qc, vlm_qc, label=label)

        conn.execute(
            """
            INSERT OR REPLACE INTO structure_qc_result
            (candidate_id, doc_id, asset_id, qc_score, auto_decision, qc_flags_json, vlm_qc_json, review_xlsx_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cid,
                args.doc_id,
                asset_id,
                qc_score,
                decision,
                jdump(qc_flags),
                jdump(vlm_qc),
                str(review_dir / "structure_review.xlsx"),
            ),
        )

        output.append({
            "candidate_id": cid,
            "asset_id": asset_id,
            "source_tool": str(row_get(r, "source_tool", "")),
            "candidate_index": r["candidate_index"],
            "molecule_label": label,
            "structure_class": structure_class,
            "smiles": smiles,
            "canonical_smiles": qc["canonical_smiles"],
            "qc_score": qc_score,
            "auto_decision": decision,
            "qc_flags": qc_flags,
            "vlm_qc": vlm_qc,
            "crop_path": crop_path,
            "render_path": render_path,
            "overlay_path": overlay_path,
        })

    conn.commit()
    conn.close()

    xlsx_path = review_dir / "structure_review.xlsx"
    report_path = review_dir / "structure_qc_report.json"

    write_xlsx(output, xlsx_path)

    report = {
        "doc_id": args.doc_id,
        "num_candidates": len(output),
        "decision_counts": dict(Counter(x["auto_decision"] for x in output)),
        "structure_class_counts": dict(Counter(x["structure_class"] for x in output)),
        "flag_counts": dict(Counter(f for x in output for f in x["qc_flags"])),
        "xlsx": str(xlsx_path),
        "image_dir": str(img_dir),
        "raw_output_dir": str(raw_dir),
        "note": (
            "UniParser image candidates use targeted VLM QC only for fixed numeric repeat abbreviations. "
            "Other image candidates use general VLM visual consistency checks. "
            "Direct CSV/XLSX/SDF/SMI structures use RDKit-only QC."
        ),
    }

    report_path.write_text(jdump(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()