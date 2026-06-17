#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run UniParser on unique structure-related page images and write molecule objects
into stg_structure_candidate.

Design goals:
- Charge-safe: parse each page image once, deduped by file SHA1.
- Cache-safe: save UniParser result JSON locally and reuse it by default.
- Stage-limited: only creates stg_structure_candidate rows. It does not align
  compound names and does not reconstruct final agents.

Example:
python scripts/run_uni_parser.py \
  --doc-id doi_1464033328326402 \
  --api-key "$UNIPARSER_API_KEY" \
  --include-mixed

Force a real API call even if cache exists:
python scripts/run_uni_parser.py \
  --doc-id doi_1464033328326402 \
  --api-key "$UNIPARSER_API_KEY" \
  --overwrite --force-api
"""

import os
import re
import sys
import json
import uuid
import time
import hashlib
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

DEFAULT_STRUCTURE_TASK_TYPES = [
    "structure_figure",
    "sar_table",
    "supplementary_structure_table",
]

GENERIC_LABEL_RE = re.compile(
    r"^(?:"
    r"protacs?|hyt\s*molecules?|molecules?|compounds?|analogs?|analogues?|"
    r"series|degraders?|inhibitors?|ligands?|warheads?|linkers?|e3\s*ligands?"
    r")$",
    re.I,
)

PLACEHOLDER_RE = re.compile(
    r"(\*|\[\*\]|\[R\d*\]|\bR\d*\b|\bX\d*\b|\bn\s*=|\bm\s*=)",
    re.I,
)


def uid(prefix: str, *parts: Any) -> str:
    return prefix + "_" + uuid.uuid5(
        uuid.NAMESPACE_URL,
        "|".join(map(str, parts)),
    ).hex[:16]


def clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def jdump(x: Any) -> str:
    return json.dumps(x or {}, ensure_ascii=False, default=str)


def jload(x: Any, default: Any = None) -> Any:
    if x in (None, ""):
        return default if default is not None else {}
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return default if default is not None else {}


def table_cols(conn, table: str) -> set:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def require_cols(conn, table: str, cols: List[str]) -> None:
    actual = table_cols(conn, table)
    missing = [c for c in cols if c not in actual]
    if missing:
        raise RuntimeError(f"{table} missing columns: {missing}; actual={sorted(actual)}")


def ensure_candidate_table(conn) -> None:
    conn.execute("""
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


def preflight(conn) -> None:
    require_cols(conn, "planned_tasks", [
        "task_id", "doc_id", "asset_id", "asset_type", "task_type", "priority", "status",
    ])
    require_cols(conn, "raw_asset", [
        "asset_id", "doc_id", "asset_type", "page_no", "file_path", "metadata_json",
    ])
    ensure_candidate_table(conn)


def insert_dynamic(conn, table: str, data: Dict[str, Any]) -> None:
    cols = [c for c in data if c in table_cols(conn, table)]
    if not cols:
        raise RuntimeError(f"No matching columns for insert into {table}")
    vals = [data[c] for c in cols]
    marks = ",".join(["?"] * len(cols))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({marks})"
    conn.execute(sql, vals)


def resolve_path(path: Any) -> Path:
    p = Path(clean(path))
    if not p.is_absolute():
        p = ROOT / p
    return p


def file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_structure_tasks(
    conn,
    doc_id: str,
    task_types: List[str],
    limit_tasks: int = 0,
) -> List[Dict[str, Any]]:
    marks = ",".join(["?"] * len(task_types))
    params = [doc_id] + task_types

    rows = conn.execute(f"""
    SELECT
        p.task_id,
        p.doc_id,
        p.asset_id,
        p.asset_type AS planned_asset_type,
        p.task_type,
        p.priority,
        p.reason,
        p.status AS task_status,
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
      AND COALESCE(p.status, 'planned') = 'planned'
    ORDER BY
        CASE p.priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
        CASE WHEN a.page_no IS NULL THEN 999999 ELSE a.page_no END,
        p.task_type,
        p.asset_id
    """, params).fetchall()

    out = [dict(r) for r in rows]
    return out[:limit_tasks] if limit_tasks else out


def resolve_page_image(task: Dict[str, Any]) -> Optional[Path]:
    """
    Prefer the original page image from parse metadata. If unavailable, fall back
    to the asset file itself. This keeps UniParser page-level and avoids multiple
    paid calls for several crops on the same page.
    """
    meta = jload(task.get("metadata_json"), {})
    candidates = [
        meta.get("source_page_image"),
        meta.get("crop_page_image_used"),
        meta.get("page_image"),
        task.get("file_path"),
    ]

    for x in candidates:
        if not x:
            continue
        p = resolve_path(x)
        if p.exists() and p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            return p

    return None


def build_page_jobs(tasks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    jobs_by_key: Dict[str, Dict[str, Any]] = {}
    skipped = []

    for task in tasks:
        page_image = resolve_page_image(task)
        if not page_image:
            skipped.append({"task": task, "reason": "page_image_not_found"})
            continue

        try:
            page_key = file_sha1(page_image)[:16]
        except Exception as e:
            skipped.append({"task": task, "reason": f"sha1_failed: {e}"})
            continue

        if page_key not in jobs_by_key:
            jobs_by_key[page_key] = {
                "page_key": page_key,
                "page_image": str(page_image),
                "page_no": task.get("page_no"),
                "source_task_ids": [],
                "source_asset_ids": [],
                "source_task_types": [],
                "source_assets": [],
            }

        job = jobs_by_key[page_key]
        job["source_task_ids"].append(task.get("task_id"))
        job["source_asset_ids"].append(task.get("asset_id"))
        job["source_task_types"].append(task.get("task_type"))
        job["source_assets"].append({
            "task_id": task.get("task_id"),
            "asset_id": task.get("asset_id"),
            "asset_type": task.get("asset_type"),
            "task_type": task.get("task_type"),
            "page_no": task.get("page_no"),
            "file_path": task.get("file_path"),
            "figure_ref": task.get("figure_ref"),
            "table_ref": task.get("table_ref"),
        })

    jobs = list(jobs_by_key.values())
    jobs.sort(key=lambda x: (999999 if x.get("page_no") is None else x.get("page_no"), x["page_key"]))
    return jobs, skipped


def load_existing_page_keys(conn, doc_id: str) -> set:
    keys = set()
    rows = conn.execute("""
    SELECT raw_context_json
    FROM stg_structure_candidate
    WHERE doc_id=?
      AND source_tool='uniparser'
      AND COALESCE(raw_context_json, '') != ''
    """, (doc_id,)).fetchall()

    for r in rows:
        ctx = jload(r["raw_context_json"], {})
        key = clean(ctx.get("page_key"))
        if key:
            keys.add(key)
    return keys


def call_uniparser(client: Any, page_image: Path) -> Dict[str, Any]:
    res = client.trigger_snip(
        snip_path=str(page_image),
        molecule=1,
        textual=0,
    )
    token = res.get("token")
    if not token:
        raise RuntimeError(f"UniParser trigger_snip did not return token: {res}")

    result = client.get_formatted(
        token,
        content=False,
        objects=True,
        molecule_source=True,
    )
    return result


def load_cached_result(cache_path: Path) -> Optional[Dict[str, Any]]:
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cached_result(cache_path: Path, result: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def object_class(obj: Dict[str, Any]) -> str:
    return clean(obj.get("class")).lower()


def get_float_xyxy(obj: Dict[str, Any]) -> List[float]:
    box = obj.get("float_xyxy") or obj.get("bbox") or obj.get("xyxy")
    if not isinstance(box, list) or len(box) != 4:
        return []
    try:
        vals = [float(x) for x in box]
    except Exception:
        return []
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        return []
    return vals


def pixel_xyxy_from_float(box: List[float], w: int, h: int) -> List[int]:
    if not box or len(box) != 4:
        return []
    x0, y0, x1, y1 = box
    # UniParser float_xyxy is usually normalized to 0~1. If not, treat it as pixels.
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        x0, x1 = x0 * w, x1 * w
        y0, y1 = y0 * h, y1 * h
    x0 = max(0, min(w - 1, int(round(x0))))
    y0 = max(0, min(h - 1, int(round(y0))))
    x1 = max(0, min(w, int(round(x1))))
    y1 = max(0, min(h, int(round(y1))))
    if x1 <= x0 or y1 <= y0:
        return []
    return [x0, y0, x1, y1]


def crop_molecule(page_image: Path, float_xyxy: List[float], out_path: Path, pad_ratio: float = 0.04) -> Tuple[str, List[int]]:
    im = Image.open(page_image).convert("RGB")
    xyxy = pixel_xyxy_from_float(float_xyxy, im.width, im.height)
    if not xyxy:
        return "", []

    x0, y0, x1, y1 = xyxy
    bw, bh = x1 - x0, y1 - y0
    pad = max(4, int(max(bw, bh) * pad_ratio))
    x0p = max(0, x0 - pad)
    y0p = max(0, y0 - pad)
    x1p = min(im.width, x1 + pad)
    y1p = min(im.height, y1 + pad)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.crop([x0p, y0p, x1p, y1p]).save(out_path)
    return str(out_path), [x0p, y0p, x1p, y1p]


def box_center(box: List[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def label_text(label: Dict[str, Any]) -> str:
    return clean(label.get("str"))


def is_generic_label(text: str) -> bool:
    return bool(GENERIC_LABEL_RE.match(clean(text)))


def find_nearest_label(mol: Dict[str, Any], labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    mbox = get_float_xyxy(mol)
    if not mbox or not labels:
        return {}

    mcx, mcy = box_center(mbox)
    best = None
    best_score = 1e18

    for lab in labels:
        lbox = get_float_xyxy(lab)
        txt = label_text(lab)
        if not lbox or not txt:
            continue

        lcx, lcy = box_center(lbox)
        dx = abs(lcx - mcx)

        # Distance from label to molecule. Prefer labels below molecule, then right/left close.
        if lcy >= mbox[1]:
            dy = max(0.0, lbox[1] - mbox[3])
            below_bonus = -0.05
        else:
            dy = max(0.0, mbox[1] - lbox[3])
            below_bonus = 0.10

        score = dx * 0.8 + dy * 1.2 + below_bonus
        if is_generic_label(txt):
            score += 0.05

        if score < best_score:
            best = lab
            best_score = score

    if not best:
        return {}

    out = dict(best)
    out["matched_score"] = best_score
    out["is_generic_label"] = is_generic_label(label_text(best))
    return out


def clean_uniparser_smiles(raw: Any) -> str:
    s = clean(raw)
    if not s:
        return ""
    # UniParser commonly wraps molecule strings as ***SMILES***. Remove only if
    # the wrapper is present on both sides; do not blindly remove real dummy atoms.
    if s.startswith("***") and s.endswith("***") and len(s) > 6:
        s = s[3:-3].strip()
    return s


def rdkit_check(smiles: str) -> Tuple[int, str, str]:
    smiles = clean(smiles)
    if not smiles:
        return 0, "", "empty_smiles"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0, "", "rdkit_parse_failed"
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return 1, canonical, ""
    except Exception as e:
        return 0, "", str(e)[:500]


def candidate_status(smiles: str, rdkit_valid: int, is_markush: bool) -> str:
    if is_markush:
        return "review_markush"
    if not rdkit_valid:
        return "review_invalid_smiles"
    if PLACEHOLDER_RE.search(smiles or ""):
        return "review_placeholder"
    return "parsed"


def normalize_objects(result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    objs = result.get("objects") or []
    molecules = []
    labels = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        cls = object_class(o)
        if cls == "molecule":
            molecules.append(o)
        elif cls in {"moleculeid", "legend"}:
            labels.append(o)
    return molecules, labels


def delete_page_candidates(conn, doc_id: str, page_key: str) -> None:
    rows = conn.execute("""
    SELECT candidate_id, raw_context_json
    FROM stg_structure_candidate
    WHERE doc_id=? AND source_tool='uniparser'
    """, (doc_id,)).fetchall()
    for r in rows:
        ctx = jload(r["raw_context_json"], {})
        if clean(ctx.get("page_key")) == page_key:
            conn.execute("DELETE FROM stg_structure_candidate WHERE candidate_id=?", (r["candidate_id"],))


def insert_uniparser_candidates(
    conn,
    doc_id: str,
    job: Dict[str, Any],
    result: Dict[str, Any],
    out_root: Path,
) -> Dict[str, int]:
    page_key = job["page_key"]
    page_image = resolve_path(job["page_image"])
    page_out = out_root / "crops" / page_key
    page_out.mkdir(parents=True, exist_ok=True)

    molecules, labels = normalize_objects(result)
    stats = {
        "num_molecules": 0,
        "valid_smiles": 0,
        "invalid_smiles": 0,
        "markush": 0,
        "placeholder": 0,
    }

    representative_asset_id = (job.get("source_asset_ids") or [""])[0]

    for idx, mol_obj in enumerate(molecules, start=1):
        raw_smiles = mol_obj.get("str") or ""
        smiles = clean_uniparser_smiles(raw_smiles)
        is_markush = bool(mol_obj.get("is_markush"))
        rdkit_valid, canonical_smiles, rdkit_error = rdkit_check(smiles)
        status = candidate_status(smiles, rdkit_valid, is_markush)

        bbox_float = get_float_xyxy(mol_obj)
        nearest = find_nearest_label(mol_obj, labels)
        mol_label = label_text(nearest)

        candidate_id = uid(
            "uniparser_mol",
            doc_id,
            page_key,
            idx,
            bbox_float,
            smiles[:80],
        )

        crop_path = page_out / f"{candidate_id}.png"
        crop_file, crop_xyxy = crop_molecule(page_image, bbox_float, crop_path)

        raw_context = {
            "source": "uniparser",
            "page_key": page_key,
            "page_image": str(page_image),
            "page_no": job.get("page_no"),
            "source_task_ids": job.get("source_task_ids") or [],
            "source_asset_ids": job.get("source_asset_ids") or [],
            "source_task_types": job.get("source_task_types") or [],
            "source_assets": job.get("source_assets") or [],
            "uniparser_token": result.get("token"),
            "uniparser_status": result.get("status"),
            "uniparser_version": result.get("version"),
            "uniparser_cost": result.get("cost"),
            "molecule_object": mol_obj,
            "nearest_label": nearest,
            "all_labels": labels,
            "is_nearest_label_generic": bool(nearest.get("is_generic_label")) if nearest else False,
            "crop_xyxy": crop_xyxy,
        }

        bbox_json = {
            "float_xyxy": bbox_float,
            "crop_xyxy": crop_xyxy,
            "page_image": str(page_image),
        }

        row = {
            "candidate_id": candidate_id,
            "doc_id": doc_id,
            "asset_id": representative_asset_id,
            "image_path": crop_file,
            "image_name": Path(crop_file).name if crop_file else "",
            "candidate_index": idx,
            "smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "rdkit_valid": int(rdkit_valid),
            "rdkit_error": rdkit_error,
            "smiles_source": "uniparser_molecule",
            "raw_json": jdump(mol_obj),
            "status": status,
            "source_tool": "uniparser",
            "raw_output": jdump(mol_obj),
            "molecule_label": mol_label,
            "bbox_json": jdump(bbox_json),
            "raw_context_json": jdump(raw_context),
        }

        insert_dynamic(conn, "stg_structure_candidate", row)

        stats["num_molecules"] += 1
        if rdkit_valid:
            stats["valid_smiles"] += 1
        else:
            stats["invalid_smiles"] += 1
        if is_markush:
            stats["markush"] += 1
        if status == "review_placeholder":
            stats["placeholder"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run UniParser on unique structure-related pages.")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--host", default=os.getenv("UNIPARSER_HOST", "https://uniparser.dp.tech/"))
    ap.add_argument("--api-key", default=os.getenv("UNIPARSER_API_KEY", ""))
    ap.add_argument("--task-types", default=",".join(DEFAULT_STRUCTURE_TASK_TYPES))
    ap.add_argument("--include-mixed", action="store_true", help="Also parse mixed_or_uncertain tasks.")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite DB rows for already processed pages. Reuses cache unless --force-api.")
    ap.add_argument("--force-api", action="store_true", help="Call UniParser even if cached JSON exists. This can cost money.")
    ap.add_argument("--no-cache", action="store_true", help="Do not reuse local cached JSON. Does not delete old cache.")
    ap.add_argument("--dry-run", action="store_true", help="Print unique page jobs and exit without API/DB insert.")
    ap.add_argument("--max-pages", type=int, default=0, help="Debug limit on unique pages.")
    ap.add_argument("--limit-tasks", type=int, default=0, help="Debug limit on source planned_tasks before page dedupe.")
    ap.add_argument("--commit-every", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    conn = get_conn()
    preflight(conn)

    task_types = [x.strip() for x in args.task_types.split(",") if x.strip()]
    if args.include_mixed and "mixed_or_uncertain" not in task_types:
        task_types.append("mixed_or_uncertain")

    tasks = load_structure_tasks(
        conn,
        doc_id=args.doc_id,
        task_types=task_types,
        limit_tasks=args.limit_tasks,
    )
    jobs, skipped = build_page_jobs(tasks)
    if args.max_pages:
        jobs = jobs[:args.max_pages]

    out_root = Path("data/work") / args.doc_id / "uniparser"
    cache_dir = out_root / "cache"
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_keys = load_existing_page_keys(conn, args.doc_id)

    print(json.dumps({
        "doc_id": args.doc_id,
        "source_tasks": len(tasks),
        "unique_pages": len(jobs),
        "skipped_tasks": len(skipped),
        "existing_uniparser_pages": len(existing_keys),
        "task_types": task_types,
        "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2))

    if args.dry_run:
        preview = []
        for j in jobs:
            preview.append({
                "page_key": j["page_key"],
                "page_no": j.get("page_no"),
                "page_image": j["page_image"],
                "num_source_tasks": len(j.get("source_task_ids") or []),
                "source_task_types": sorted(set(j.get("source_task_types") or [])),
            })
        dry_path = out_root / "dry_run_page_jobs.json"
        dry_path.write_text(json.dumps({"jobs": preview, "skipped": skipped}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"dry_run_page_jobs": str(dry_path)}, ensure_ascii=False, indent=2))
        conn.close()
        return

    if not args.api_key and args.force_api:
        raise RuntimeError("--api-key or UNIPARSER_API_KEY is required when --force-api is used")

    client = None
    if any((j["page_key"] not in existing_keys or args.overwrite) for j in jobs):
        if not args.no_cache and not args.force_api:
            # Client is created lazily only if a cache miss occurs.
            client = None
        else:
            from uniparser_tools.api.clients import UniParserClient
            client = UniParserClient(host=args.host, api_key=args.api_key)

    stats = {
        "doc_id": args.doc_id,
        "num_source_tasks": len(tasks),
        "num_unique_pages": len(jobs),
        "num_skipped_tasks": len(skipped),
        "num_skipped_existing_pages": 0,
        "num_cache_hits": 0,
        "num_api_calls": 0,
        "num_error_pages": 0,
        "num_called_or_loaded_pages": 0,
        "num_molecules": 0,
        "valid_smiles": 0,
        "invalid_smiles": 0,
        "markush": 0,
        "placeholder": 0,
        "total_cost": 0.0,
        "page_results": [],
    }

    for n, job in enumerate(tqdm(jobs, desc="UniParser pages"), start=1):
        page_key = job["page_key"]
        page_image = resolve_path(job["page_image"])
        cache_path = cache_dir / f"{page_key}.json"

        if page_key in existing_keys and not args.overwrite:
            stats["num_skipped_existing_pages"] += 1
            stats["page_results"].append({
                "page_key": page_key,
                "page_image": str(page_image),
                "status": "skipped_existing_db",
            })
            continue

        try:
            result = None
            used_cache = False

            if not args.no_cache and not args.force_api:
                result = load_cached_result(cache_path)
                if result is not None:
                    used_cache = True
                    stats["num_cache_hits"] += 1

            if result is None:
                if not args.api_key:
                    raise RuntimeError("UNIPARSER_API_KEY is empty and cache is unavailable. Set --api-key or UNIPARSER_API_KEY.")
                if client is None:
                    from uniparser_tools.api.clients import UniParserClient
                    client = UniParserClient(host=args.host, api_key=args.api_key)
                result = call_uniparser(client, page_image)
                save_cached_result(cache_path, result)
                stats["num_api_calls"] += 1
                time.sleep(0.05)

            if args.overwrite:
                delete_page_candidates(conn, args.doc_id, page_key)

            page_stats = insert_uniparser_candidates(
                conn=conn,
                doc_id=args.doc_id,
                job=job,
                result=result,
                out_root=out_root,
            )

            stats["num_called_or_loaded_pages"] += 1
            stats["num_molecules"] += page_stats["num_molecules"]
            stats["valid_smiles"] += page_stats["valid_smiles"]
            stats["invalid_smiles"] += page_stats["invalid_smiles"]
            stats["markush"] += page_stats["markush"]
            stats["placeholder"] += page_stats["placeholder"]

            try:
                stats["total_cost"] += float(result.get("cost") or 0)
            except Exception:
                pass

            stats["page_results"].append({
                "page_key": page_key,
                "page_no": job.get("page_no"),
                "page_image": str(page_image),
                "cache_path": str(cache_path),
                "used_cache": used_cache,
                "uniparser_status": result.get("status"),
                "token": result.get("token"),
                "cost": result.get("cost"),
                **page_stats,
            })

            if args.commit_every and n % args.commit_every == 0:
                conn.commit()

        except Exception as e:
            stats["num_error_pages"] += 1
            stats["page_results"].append({
                "page_key": page_key,
                "page_no": job.get("page_no"),
                "page_image": str(page_image),
                "status": "error",
                "error": str(e),
            })
            print(f"[ERROR] page_key={page_key} image={page_image}: {e}")

    conn.commit()
    conn.close()

    report_path = out_root / "report.json"
    skipped_path = out_root / "skipped_tasks.json"
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    skipped_path.write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "doc_id": args.doc_id,
        "report_path": str(report_path),
        "skipped_tasks_path": str(skipped_path),
        "num_source_tasks": stats["num_source_tasks"],
        "num_unique_pages": stats["num_unique_pages"],
        "num_skipped_existing_pages": stats["num_skipped_existing_pages"],
        "num_cache_hits": stats["num_cache_hits"],
        "num_api_calls": stats["num_api_calls"],
        "num_error_pages": stats["num_error_pages"],
        "num_molecules": stats["num_molecules"],
        "valid_smiles": stats["valid_smiles"],
        "invalid_smiles": stats["invalid_smiles"],
        "total_cost": round(stats["total_cost"], 4),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
