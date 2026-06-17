#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import uuid
import base64
import argparse
from pathlib import Path
from typing import Any, Dict, List

import requests
from tqdm import tqdm
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn


IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def uid(prefix: str, *parts: Any) -> str:
    return prefix + "_" + uuid.uuid5(
        uuid.NAMESPACE_URL,
        "|".join(map(str, parts)),
    ).hex[:16]


def clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


def jload(x: Any, default: Any = None) -> Any:
    if x in ("", None):
        return default if default is not None else {}
    try:
        return json.loads(x)
    except Exception:
        return default if default is not None else {}


def table_cols(conn, table: str) -> set:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def ensure_candidate_table(conn) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS stg_structure_candidate (
        candidate_id TEXT PRIMARY KEY,
        doc_id TEXT,
        asset_id TEXT,
        image_path TEXT,
        candidate_index INTEGER,
        smiles TEXT,
        canonical_smiles TEXT,
        rdkit_valid INTEGER,
        rdkit_error TEXT,
        source_tool TEXT,
        raw_output TEXT,
        status TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    wanted = {
        "candidate_id": "TEXT PRIMARY KEY",
        "doc_id": "TEXT",
        "asset_id": "TEXT",
        "image_path": "TEXT",
        "image_name": "TEXT",
        "candidate_index": "INTEGER",
        "smiles": "TEXT",
        "canonical_smiles": "TEXT",
        "rdkit_valid": "INTEGER",
        "rdkit_error": "TEXT",
        "smiles_source": "TEXT",
        "raw_json": "TEXT",
        "status": "TEXT",
        "source_tool": "TEXT",
        "raw_output": "TEXT",
        "molecule_label": "TEXT",
        "bbox_json": "TEXT",
        "raw_context_json": "TEXT",
        "created_at": "TEXT DEFAULT CURRENT_TIMESTAMP",
    }

    cols = table_cols(conn, "stg_structure_candidate")
    for col, typ in wanted.items():
        if col not in cols:
            try:
                conn.execute(f"ALTER TABLE stg_structure_candidate ADD COLUMN {col} {typ}")
            except Exception:
                pass

    conn.commit()


def resolve_path(path: str) -> Path:
    p = Path(path or "")
    if not p.is_absolute():
        p = ROOT / p
    return p


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def image_to_data_url(path: Path) -> str:
    mime = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    return f"data:{mime};base64,{image_to_base64(path)}"


def extract_json_maybe(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return {}
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

    return {"text": text}


def find_smiles_candidates(obj: Any) -> List[Dict[str, Any]]:
    hits = []

    smiles_keys = {
        "smiles",
        "SMILES",
        "canonical_smiles",
        "Canonical_SMILES",
        "canonicalSMILES",
        "mol_smiles",
        "extended_smiles",
        "cxsmiles",
        "cx_smiles",
    }

    def walk(x: Any, path: str = "") -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                p = f"{path}.{k}" if path else str(k)
                if k in smiles_keys and isinstance(v, str) and v.strip():
                    hits.append({
                        "smiles": v.strip(),
                        "source_path": p,
                        "raw_context": x,
                    })
                walk(v, p)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(obj)

    seen = set()
    out = []
    for h in hits:
        s = h["smiles"]
        if s not in seen:
            seen.add(s)
            out.append(h)

    return out


def rdkit_qc(smiles: str) -> Dict[str, Any]:
    s = clean(smiles)

    out = {
        "rdkit_valid": 0,
        "canonical_smiles": "",
        "rdkit_error": "",
        "has_dummy_atom": False,
        "has_r_group": False,
        "has_fragment_separator": "." in s,
    }

    if not s:
        out["rdkit_error"] = "empty smiles"
        return out

    if "*" in s:
        out["has_dummy_atom"] = True

    if re.search(r"(^|[^A-Za-z])R\d*([^A-Za-z]|$)", s) or re.search(r"\[[^\]]*R[^\]]*\]", s):
        out["has_r_group"] = True

    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            out["rdkit_error"] = "MolFromSmiles returned None"
            return out

        out["rdkit_valid"] = 1
        out["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)

        if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            out["has_dummy_atom"] = True

    except Exception as e:
        out["rdkit_error"] = str(e)

    return out


def decide_status(qc: Dict[str, Any], no_candidate: bool = False, multi: bool = False) -> str:
    if no_candidate:
        return "review_no_candidate"
    if not qc["rdkit_valid"]:
        return "review_invalid_smiles"
    if qc["has_dummy_atom"] or qc["has_r_group"]:
        return "review_dummy_or_r_group"
    if qc["has_fragment_separator"]:
        return "review_fragmented_smiles"
    if multi:
        return "review_multiple_candidates"
    return "parsed_ok"


def call_molparser(
    api_url: str,
    image_path: Path,
    request_mode: str,
    image_field: str,
    timeout: int,
    extra_json: Dict[str, Any],
) -> Any:
    if request_mode == "file":
        with image_path.open("rb") as f:
            r = requests.post(
                api_url,
                files={image_field: (image_path.name, f, "image/png")},
                data={k: str(v) for k, v in extra_json.items()},
                timeout=timeout,
            )
    else:
        if request_mode == "base64":
            image_value = image_to_base64(image_path)
        else:
            image_value = image_to_data_url(image_path)

        payload = dict(extra_json)
        payload[image_field] = image_value

        r = requests.post(api_url, json=payload, timeout=timeout)

    r.raise_for_status()

    ctype = r.headers.get("content-type", "")
    if "json" in ctype.lower():
        return r.json()

    return extract_json_maybe(r.text)


def load_candidates(conn, doc_id: str, overwrite: bool, limit: int = 0):
    where = ["doc_id=?"]
    params = [doc_id]

    if not overwrite:
        where.append("(COALESCE(smiles, '') = '' OR source_tool='moldetv2')")

    sql = f"""
    SELECT *
    FROM stg_structure_candidate
    WHERE {' AND '.join(where)}
      AND source_tool IN ('moldetv2', 'molparser')
      AND COALESCE(image_path, '') != ''
    ORDER BY asset_id, candidate_index
    """

    rows = conn.execute(sql, params).fetchall()
    rows = [dict(r) for r in rows]

    out = []
    for r in rows:
        p = resolve_path(r["image_path"])
        if p.exists() and p.suffix.lower() in IMAGE_SUFFIX:
            out.append(r)

    return out[:limit] if limit else out


def update_candidate(conn, row: Dict[str, Any], update: Dict[str, Any]) -> None:
    cols = table_cols(conn, "stg_structure_candidate")
    data = {k: v for k, v in update.items() if k in cols}

    sets = ", ".join([f"{k}=?" for k in data.keys()])
    vals = list(data.values()) + [row["candidate_id"]]

    conn.execute(
        f"UPDATE stg_structure_candidate SET {sets} WHERE candidate_id=?",
        vals,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--api-url", default=os.getenv("MOLPARSER_API_URL", "https://ocsr.dp.tech/mol/img2mol"))
    ap.add_argument("--request-mode", choices=["data_url", "base64", "file"], default="data_url")
    ap.add_argument("--image-field", default="image")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--extra-json", default="{}", help='Extra JSON payload, e.g. {"mode":"smiles"}')
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    try:
        extra_json = json.loads(args.extra_json or "{}")
        if not isinstance(extra_json, dict):
            extra_json = {}
    except Exception:
        extra_json = {}

    conn = get_conn()
    ensure_candidate_table(conn)

    rows = load_candidates(conn, args.doc_id, overwrite=args.overwrite, limit=args.limit)

    stats = {
        "doc_id": args.doc_id,
        "num_candidates": len(rows),
        "parsed_ok": 0,
        "review": 0,
        "error": 0,
    }

    for row in tqdm(rows, desc="MolParser parse crops"):
        image_path = resolve_path(row["image_path"])
        old_ctx = jload(row.get("raw_context_json"), {})

        try:
            raw = call_molparser(
                api_url=args.api_url,
                image_path=image_path,
                request_mode=args.request_mode,
                image_field=args.image_field,
                timeout=args.timeout,
                extra_json=extra_json,
            )

            candidates = find_smiles_candidates(raw)
            no_candidate = len(candidates) == 0

            if no_candidate:
                smiles = ""
                qc = rdkit_qc("")
                status = decide_status(qc, no_candidate=True)
            else:
                smiles_values = [x["smiles"] for x in candidates]
                unique_smiles = []
                for s in smiles_values:
                    if s not in unique_smiles:
                        unique_smiles.append(s)

                smiles = unique_smiles[0]
                qc = rdkit_qc(smiles)
                status = decide_status(qc, multi=len(unique_smiles) > 1)

            raw_context = dict(old_ctx)
            raw_context["molparser"] = {
                "api_url": args.api_url,
                "request_mode": args.request_mode,
                "num_smiles_candidates": len(candidates),
                "chosen_smiles": smiles,
            }

            update_candidate(conn, row, {
                "smiles": smiles,
                "canonical_smiles": qc["canonical_smiles"],
                "rdkit_valid": int(qc["rdkit_valid"]),
                "rdkit_error": qc["rdkit_error"],
                "smiles_source": "molparser",
                "source_tool": "molparser",
                "raw_json": jdump(raw),
                "raw_output": jdump(raw),
                "raw_context_json": jdump(raw_context),
                "status": status,
            })

            if status == "parsed_ok":
                stats["parsed_ok"] += 1
            else:
                stats["review"] += 1

            conn.commit()

        except Exception as e:
            stats["error"] += 1
            raw_context = dict(old_ctx)
            raw_context["molparser_error"] = str(e)

            update_candidate(conn, row, {
                "source_tool": "molparser",
                "rdkit_valid": 0,
                "rdkit_error": str(e),
                "raw_context_json": jdump(raw_context),
                "status": "review_molparser_error",
            })
            conn.commit()

    conn.close()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
