#!/usr/bin/env python3
import os
import re
import sys
import json
import uuid
import argparse
import traceback
from pathlib import Path

from tqdm import tqdm
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn


STRUCTURE_TASK_TYPES = [
    "structure_figure",
    "sar_table",
    "supplementary_structure_table",
    "mixed_or_uncertain",
]


IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def uid(prefix, *parts):
    return prefix + "_" + uuid.uuid5(uuid.NAMESPACE_URL, "|".join(map(str, parts))).hex[:16]


def to_jsonable(x):
    try:
        json.dumps(x, ensure_ascii=False)
        return x
    except Exception:
        return str(x)


def ensure_table(conn):
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

    cols = {
        r["name"]
        for r in conn.execute("PRAGMA table_info(stg_structure_candidate)").fetchall()
    }

    wanted = {
        "candidate_id": "TEXT PRIMARY KEY",
        "doc_id": "TEXT",
        "asset_id": "TEXT",
        "image_path": "TEXT",
        "candidate_index": "INTEGER",
        "smiles": "TEXT",
        "canonical_smiles": "TEXT",
        "rdkit_valid": "INTEGER",
        "rdkit_error": "TEXT",
        "source_tool": "TEXT",
        "raw_output": "TEXT",
        "status": "TEXT",
        "created_at": "TEXT DEFAULT CURRENT_TIMESTAMP",
    }

    for col, typ in wanted.items():
        if col not in cols:
            try:
                conn.execute(f"ALTER TABLE stg_structure_candidate ADD COLUMN {col} {typ}")
            except Exception:
                pass

    conn.commit()


def import_chemeagle(chemeagle_root):
    root = Path(chemeagle_root).resolve()
    sys.path.insert(0, str(root))

    os.environ.setdefault("API_KEY", "dummy")
    os.environ.setdefault("AZURE_ENDPOINT", "dummy")
    os.environ.setdefault("API_VERSION", "dummy")
    os.environ.setdefault("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")

    old = os.getcwd()
    os.chdir(root)
    try:
        from get_R_group_sub_agent import get_multi_molecular_full_OS_with_box as get_multi_molecular_full_OS
    finally:
        os.chdir(old)

    return root, get_multi_molecular_full_OS


def call_chemeagle(func, image_path, model_name):
    old = os.getcwd()
    os.chdir(Path(os.environ.get("CHEMEAGLE_ROOT", ".")).resolve())
    try:
        if model_name:
            try:
                return func(image_path=str(image_path), model_name=model_name)
            except TypeError:
                return func(image_path=str(image_path))
        return func(image_path=str(image_path))
    finally:
        os.chdir(old)

def find_smiles_candidates(obj):
    """
    适配 ChemEAGLE 输出：
    [
      {"smiles": "...", "texts": ["27a: n = 2"], "bbox_id": ""}
    ]

    输出字段：
    smiles
    molecule_label
    molecule_texts
    bbox
    bbox_id
    raw_context
    """
    hits = []

    smiles_keys = {
        "smiles",
        "SMILES",
        "canonical_smiles",
        "Canonical_SMILES",
        "canonicalSMILES",
        "mol_smiles",
    }

    bbox_keys = {"bbox", "box", "xyxy", "position"}

    def make_label(x):
        if not isinstance(x, dict):
            return "", []

        texts = x.get("texts", [])
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list):
            texts = []

        texts = [str(t).strip() for t in texts if str(t).strip()]

        for k in ["label", "molecule_label", "compound_label", "compound", "compound_name", "name", "index"]:
            v = x.get(k)
            if isinstance(v, (str, int, float)) and str(v).strip():
                texts.insert(0, str(v).strip())

        label = "||".join(texts)
        return label, texts

    def get_bbox(x):
        if not isinstance(x, dict):
            return None
        for k in bbox_keys:
            if k in x:
                return x.get(k)
        return None

    def walk(x, path=""):
        if isinstance(x, dict):
            label, texts = make_label(x)
            bbox = get_bbox(x)
            bbox_id = x.get("bbox_id", "")

            for k, v in x.items():
                p = f"{path}.{k}" if path else str(k)

                if k in smiles_keys and isinstance(v, str) and v.strip():
                    hits.append({
                        "smiles": v.strip(),
                        "source_path": p,
                        "label": label,
                        "molecule_label": label,
                        "molecule_texts": texts,
                        "bbox": bbox,
                        "bbox_id": bbox_id,
                        "raw_context": x,
                    })

                walk(v, p)

        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(obj)

    seen = set()
    dedup = []
    for h in hits:
        key = (
            h.get("smiles", ""),
            h.get("molecule_label", ""),
            str(h.get("bbox_id", "")),
            json.dumps(h.get("bbox"), ensure_ascii=False, sort_keys=True, default=str),
        )
        if key not in seen:
            seen.add(key)
            dedup.append(h)

    return dedup

def rdkit_qc(smiles):
    qc = {
        "rdkit_valid": 0,
        "canonical_smiles": "",
        "rdkit_error": "",
        "has_dummy_atom": False,
        "has_r_group": False,
        "has_fragment_separator": "." in (smiles or ""),
    }

    s = smiles or ""

    if "*" in s:
        qc["has_dummy_atom"] = True

    if re.search(r"(^|[^A-Za-z])R\d*([^A-Za-z]|$)", s) or re.search(r"\[[^\]]*R[^\]]*\]", s):
        qc["has_r_group"] = True

    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            qc["rdkit_error"] = "MolFromSmiles returned None"
            return qc

        qc["rdkit_valid"] = 1
        qc["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)

        if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            qc["has_dummy_atom"] = True

    except Exception as e:
        qc["rdkit_error"] = str(e)

    return qc


def decide_status(qc, no_candidate=False, multi_inconsistent=False):
    if no_candidate:
        return "review_no_candidate"

    if not qc["rdkit_valid"]:
        return "review_invalid_smiles"

    if qc["has_dummy_atom"] or qc["has_r_group"]:
        return "review_dummy_or_r_group"

    if qc["has_fragment_separator"]:
        return "review_fragmented_smiles"

    if multi_inconsistent:
        return "review_multiple_candidates"

    return "ok"


def get_structure_tasks(conn, doc_id, task_types, overwrite=False, limit=None):
    marks = ",".join(["?"] * len(task_types))
    params = [doc_id] + task_types

    sql = f"""
    SELECT
        p.task_id,
        p.doc_id,
        p.asset_id,
        p.task_type,
        p.priority,
        a.asset_type,
        a.file_path
    FROM planned_tasks p
    JOIN raw_asset a ON p.asset_id = a.asset_id
    WHERE p.doc_id=?
      AND p.task_type IN ({marks})
    ORDER BY
        CASE p.priority
            WHEN 'high' THEN 1
            WHEN 'medium' THEN 2
            ELSE 3
        END,
        a.page_no,
        p.task_type
    """

    rows = conn.execute(sql, params).fetchall()

    out = []
    for r in rows:
        img = Path(r["file_path"] or "")
        if img.suffix.lower() not in IMAGE_SUFFIX:
            continue
        if not img.exists():
            continue

        if not overwrite:
            n = conn.execute(
                """
                SELECT COUNT(*)
                FROM stg_structure_candidate
                WHERE doc_id=? AND asset_id=?
                """,
                (doc_id, r["asset_id"]),
            ).fetchone()[0]
            if n > 0:
                continue

        out.append(r)

    if limit:
        out = out[:limit]

    return out


def insert_candidate(
    conn,
    doc_id,
    asset_id,
    image_path,
    candidate_index,
    smiles,
    canonical_smiles,
    rdkit_valid,
    rdkit_error,
    raw_output,
    status,
):
    candidate_id = uid(
        "struct",
        doc_id,
        asset_id,
        candidate_index,
        smiles or "",
        canonical_smiles or "",
    )

    conn.execute(
        """
        INSERT OR REPLACE INTO stg_structure_candidate
        (
            candidate_id,
            doc_id,
            asset_id,
            image_path,
            candidate_index,
            smiles,
            canonical_smiles,
            rdkit_valid,
            rdkit_error,
            source_tool,
            raw_output,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate_id,
            doc_id,
            asset_id,
            str(image_path),
            candidate_index,
            smiles or "",
            canonical_smiles or "",
            int(rdkit_valid or 0),
            rdkit_error or "",
            "ChemEAGLE",
            json.dumps(raw_output, ensure_ascii=False, default=str),
            status,
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--chemeagle-root", default=os.getenv("CHEMEAGLE_ROOT", "external/ChemEagle"))
    ap.add_argument("--model-name", default="ipm-vlm")
    ap.add_argument("--task-types", default=",".join(STRUCTURE_TASK_TYPES))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.environ["CHEMEAGLE_ROOT"] = str(Path(args.chemeagle_root).resolve())

    conn = get_conn()
    ensure_table(conn)

    task_types = [x.strip() for x in args.task_types.split(",") if x.strip()]
    tasks = get_structure_tasks(
        conn,
        args.doc_id,
        task_types,
        overwrite=args.overwrite,
        limit=args.limit or None,
    )

    chemeagle_root, func = import_chemeagle(args.chemeagle_root)

    summary = {
        "doc_id": args.doc_id,
        "num_tasks": len(tasks),
        "ok": 0,
        "review": 0,
        "failed": 0,
        "no_candidate": 0,
    }

    for t in tqdm(tasks, desc="ChemEAGLE"):
        asset_id = t["asset_id"]
        image_path = Path(t["file_path"]).resolve()

        try:
            raw = call_chemeagle(func, image_path, args.model_name)
            raw_json = to_jsonable(raw)

            hits = find_smiles_candidates(raw)

            if not hits:
                insert_candidate(
                    conn,
                    args.doc_id,
                    asset_id,
                    image_path,
                    -1,
                    "",
                    "",
                    0,
                    "no smiles candidate found",
                    raw_json,
                    "review_no_candidate",
                )
                conn.commit()
                summary["no_candidate"] += 1
                summary["review"] += 1
                continue

            qcs = [rdkit_qc(h["smiles"]) for h in hits]
            valid_canons = {
                q["canonical_smiles"]
                for q in qcs
                if q["rdkit_valid"] and q["canonical_smiles"]
            }
            multi_inconsistent = len(valid_canons) > 1

            for i, (h, qc) in enumerate(zip(hits, qcs)):
                raw_context = {
                    "task_id": t["task_id"],
                    "task_type": t["task_type"],
                    "asset_type": t["asset_type"],
                    "source_path": h.get("source_path", ""),
                    "label": h.get("label", ""),
                    "bbox": h.get("bbox"),
                    "raw_context": h.get("raw_context"),
                    "full_raw_output": raw_json,
                    "qc": {
                        "has_dummy_atom": qc["has_dummy_atom"],
                        "has_r_group": qc["has_r_group"],
                        "has_fragment_separator": qc["has_fragment_separator"],
                        "multi_inconsistent": multi_inconsistent,
                    },
                }

                status = decide_status(qc, multi_inconsistent=multi_inconsistent)

                insert_candidate(
                    conn,
                    args.doc_id,
                    asset_id,
                    image_path,
                    i,
                    h["smiles"],
                    qc["canonical_smiles"],
                    qc["rdkit_valid"],
                    qc["rdkit_error"],
                    raw_context,
                    status,
                )

                if status == "ok":
                    summary["ok"] += 1
                else:
                    summary["review"] += 1

            conn.commit()

        except Exception as e:
            err = traceback.format_exc()
            insert_candidate(
                conn,
                args.doc_id,
                asset_id,
                image_path,
                -1,
                "",
                "",
                0,
                str(e),
                {
                    "error": err,
                    "asset_id": asset_id,
                    "image_path": str(image_path),
                },
                "failed",
            )
            conn.commit()
            summary["failed"] += 1

    conn.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
