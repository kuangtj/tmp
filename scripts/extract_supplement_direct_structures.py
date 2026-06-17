#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract complete compound structures directly from supplementary tables/files.

Priority source for missing structure resolution:
- raw_table.table_json from supplementary/main SAR tables
- CSV/TSV/XLSX assets registered by parse_pdf.py
- .smi/.smiles/.sdf files under raw_document.supplement_dir

Output:
- stg_structure_candidate with source_tool='supplement_direct'
- stg_component_relation with component_role='full', relation_type='exact_full_structure'
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn

try:
    from rdkit import Chem
except Exception as e:
    Chem = None
    RDKit_IMPORT_ERROR = str(e)
else:
    RDKit_IMPORT_ERROR = ""


NAME_COLS = {
    "compound", "compound id", "compound_id", "cmpd", "cmpd id", "id", "no", "no.",
    "entry", "name", "molecule", "molecule id", "molecule_id", "code", "label",
}
STRUCT_COLS = {
    "smiles", "canonical smiles", "canonical_smiles", "isomeric smiles", "isomeric_smiles",
    "cxsmiles", "cx smiles", "inchi", "structure", "molblock", "mol block", "molfile", "mol file",
}
GENERIC_NAMES = {
    "compound", "compounds", "molecule", "molecules", "protac", "protacs", "hyt", "hyt molecules",
    "series", "analogs", "analogues", "scaffold", "linker", "warhead", "e3 ligand", "r group", "r-groups",
}


def uid(prefix: str, *parts: Any) -> str:
    return prefix + "_" + hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def jdump(x: Any) -> str:
    return json.dumps(x if x is not None else {}, ensure_ascii=False, default=str)


def jload(s: str, default: Any = None) -> Any:
    try:
        return json.loads(s or "")
    except Exception:
        return default


def norm_col(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip().lower())


def clean_name(x: Any) -> str:
    s = re.sub(r"\s+", " ", str(x or "").strip())
    s = re.sub(r"^(compound|cmpd|no\.|no|entry)\s+", "", s, flags=re.I).strip()
    return s


def is_generic_name(name: str) -> bool:
    s = clean_name(name).lower().strip(" .,:;()[]{}")
    if not s:
        return True
    if s in GENERIC_NAMES:
        return True
    if len(s) > 80:
        return True
    return False


def canonicalize_structure(value: str, source_kind: str) -> Tuple[str, str, int, str]:
    if Chem is None:
        return "", "", 0, f"rdkit_import_failed:{RDKit_IMPORT_ERROR}"
    value = (value or "").strip()
    if not value:
        return "", "", 0, "empty_structure"
    mol = None
    err = ""
    try:
        if source_kind == "inchi" or value.lower().startswith("inchi="):
            mol = Chem.MolFromInchi(value, sanitize=True)
        elif source_kind in {"molblock", "molfile"} or "M  END" in value:
            mol = Chem.MolFromMolBlock(value, sanitize=True, removeHs=False)
        else:
            mol = Chem.MolFromSmiles(value, sanitize=True)
        if mol is None:
            return value, "", 0, "rdkit_parse_failed"
        can = Chem.MolToSmiles(mol, isomericSmiles=True)
        return value, can, 1, ""
    except Exception as e:
        err = str(e)
    return value, "", 0, err


def classify_structure_col(col: str) -> Optional[str]:
    c = norm_col(col)
    if c in {"inchi"}:
        return "inchi"
    if c in {"molblock", "mol block", "molfile", "mol file"}:
        return "molblock"
    if c in STRUCT_COLS:
        return "smiles"
    if "smiles" in c:
        return "smiles"
    if c == "structure":
        return "smiles"
    return None


def rows_from_table_json(table_json: str) -> List[Dict[str, Any]]:
    obj = jload(table_json, {})
    if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
        return [r for r in obj["rows"] if isinstance(r, dict)]
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    return []


def iter_raw_tables(conn, doc_id: str, task_types: Optional[set] = None) -> Iterable[Dict[str, Any]]:
    if task_types:
        rows = conn.execute(
            """
            SELECT DISTINCT t.*
            FROM raw_table t
            JOIN planned_tasks p ON p.asset_id=t.table_id AND p.doc_id=t.doc_id
            WHERE t.doc_id=? AND p.task_type IN (%s)
            """ % ",".join("?" for _ in task_types),
            (doc_id, *sorted(task_types)),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM raw_table WHERE doc_id=?", (doc_id,)).fetchall()
    for r in rows:
        yield dict(r)


def extract_from_rows(doc_id: str, table: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = rows_from_table_json(table.get("table_json") or "")
    if not rows:
        return []
    cols = list(rows[0].keys())
    name_cols = [c for c in cols if norm_col(c) in NAME_COLS or norm_col(c).replace("_", " ") in NAME_COLS]
    struct_cols = [(c, classify_structure_col(c)) for c in cols if classify_structure_col(c)]
    if not name_cols or not struct_cols:
        return []
    name_col = name_cols[0]
    out = []
    for i, row in enumerate(rows):
        name = clean_name(row.get(name_col, ""))
        if is_generic_name(name):
            continue
        for scol, skind in struct_cols:
            raw = str(row.get(scol, "") or "").strip()
            if not raw or len(raw) < 3:
                continue
            smiles, canonical, valid, err = canonicalize_structure(raw, skind or "smiles")
            if not valid:
                continue
            out.append({
                "compound_name": name,
                "smiles": smiles,
                "canonical_smiles": canonical,
                "rdkit_valid": valid,
                "rdkit_error": err,
                "smiles_source": f"raw_table:{scol}",
                "source_table_id": table.get("table_id"),
                "asset_id": table.get("table_id"),
                "source_file": table.get("file_path"),
                "source_row_index": i,
                "raw_row": row,
                "table_ref": table.get("table_ref", ""),
            })
    return out


def iter_structure_files(supp_dir: str) -> Iterable[Path]:
    if not supp_dir:
        return
    root = Path(supp_dir)
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".smi", ".smiles", ".sdf"}:
            yield p


def extract_from_smi(path: Path) -> List[Dict[str, Any]]:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line, maxsplit=1)
        smiles = parts[0]
        name = clean_name(parts[1]) if len(parts) > 1 else path.stem + f"_{i+1}"
        if is_generic_name(name):
            continue
        raw, can, valid, err = canonicalize_structure(smiles, "smiles")
        if not valid:
            continue
        out.append({"compound_name": name, "smiles": raw, "canonical_smiles": can, "rdkit_valid": 1, "rdkit_error": err, "smiles_source": "smi_file", "source_file": str(path), "source_row_index": i, "raw_row": {"line": line}})
    return out


def extract_from_sdf(path: Path) -> List[Dict[str, Any]]:
    out = []
    if Chem is None:
        return out
    suppl = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        props = {k: mol.GetProp(k) for k in mol.GetPropNames()}
        name = ""
        for k in ["Compound", "compound", "ID", "Name", "TITLE", "Molecule", "No"]:
            if props.get(k):
                name = clean_name(props[k])
                break
        if not name:
            name = clean_name(mol.GetProp("_Name") if mol.HasProp("_Name") else f"{path.stem}_{i+1}")
        if is_generic_name(name):
            continue
        can = Chem.MolToSmiles(mol, isomericSmiles=True)
        out.append({"compound_name": name, "smiles": can, "canonical_smiles": can, "rdkit_valid": 1, "rdkit_error": "", "smiles_source": "sdf_file", "source_file": str(path), "source_row_index": i, "raw_row": props})
    return out


def insert_candidate_and_relation(conn, doc_id: str, rec: Dict[str, Any], overwrite: bool) -> None:
    candidate_id = uid("suppstruct", doc_id, rec["compound_name"], rec["canonical_smiles"], rec.get("source_file", ""), rec.get("source_row_index", ""))
    relation_id = uid("scr", doc_id, candidate_id, rec["compound_name"], "full")
    raw_context = {
        "source_table_id": rec.get("source_table_id", ""),
        "source_file": rec.get("source_file", ""),
        "source_row_index": rec.get("source_row_index"),
        "table_ref": rec.get("table_ref", ""),
    }
    if overwrite:
        conn.execute("DELETE FROM stg_component_relation WHERE relation_id=?", (relation_id,))
        conn.execute("DELETE FROM stg_structure_candidate WHERE candidate_id=?", (candidate_id,))
    conn.execute(
        """
        INSERT OR IGNORE INTO stg_structure_candidate
        (candidate_id, doc_id, asset_id, compound_name, molecule_label, smiles, canonical_smiles,
         rdkit_valid, rdkit_error, source_tool, smiles_source, raw_context_json, raw_output, raw_json, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate_id, doc_id, rec.get("asset_id", ""), rec["compound_name"], rec["compound_name"],
            rec["smiles"], rec["canonical_smiles"], int(rec["rdkit_valid"]), rec.get("rdkit_error", ""),
            "supplement_direct", rec.get("smiles_source", ""), jdump(raw_context), jdump(rec), jdump(rec),
            "auto_pass_direct_full_structure",
        ),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO stg_component_relation
        (relation_id, doc_id, asset_id, candidate_id, compound_name, component_role, relation_type,
         evidence_text, figure_ref, confidence, review_required, raw_output, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            relation_id, doc_id, rec.get("asset_id", ""), candidate_id, rec["compound_name"], "full",
            "exact_full_structure", f"Direct supplementary structure for {rec['compound_name']}", "",
            1.0, 0, jdump({"source": "supplement_direct", **raw_context}), "auto_accepted",
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--task-types", default="supplementary_structure_table,sar_table")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if Chem is None:
        raise SystemExit(f"RDKit is required: {RDKit_IMPORT_ERROR}")

    task_types = {x.strip() for x in args.task_types.split(",") if x.strip()} if args.task_types else None
    conn = get_conn()
    doc = conn.execute("SELECT supplement_dir FROM raw_document WHERE doc_id=?", (args.doc_id,)).fetchone()
    if not doc:
        raise SystemExit(f"doc_id not found: {args.doc_id}")

    records: List[Dict[str, Any]] = []
    for table in iter_raw_tables(conn, args.doc_id, task_types):
        records.extend(extract_from_rows(args.doc_id, table))

    for p in iter_structure_files(doc["supplement_dir"] or ""):
        if p.suffix.lower() in {".smi", ".smiles"}:
            records.extend(extract_from_smi(p))
        elif p.suffix.lower() == ".sdf":
            records.extend(extract_from_sdf(p))

    # Detect conflicting canonical structures per compound.
    by_name: Dict[str, set] = {}
    for r in records:
        by_name.setdefault(r["compound_name"].lower(), set()).add(r["canonical_smiles"])
    conflict_names = {k for k, vals in by_name.items() if len(vals) > 1}
    for r in records:
        if r["compound_name"].lower() in conflict_names:
            r["conflict"] = True

    if args.overwrite:
        conn.execute("DELETE FROM stg_component_relation WHERE doc_id=? AND raw_output LIKE '%supplement_direct%'", (args.doc_id,))
        conn.execute("DELETE FROM stg_structure_candidate WHERE doc_id=? AND source_tool='supplement_direct'", (args.doc_id,))

    for r in records:
        insert_candidate_and_relation(conn, args.doc_id, r, overwrite=False)
    conn.commit()

    out_dir = ROOT / "data" / "staging" / args.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "supplement_direct_structures.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(jdump(r) + "\n")

    report = {
        "doc_id": args.doc_id,
        "num_records": len(records),
        "num_compounds": len({r["compound_name"].lower() for r in records}),
        "conflict_compounds": sorted(conflict_names),
        "output_jsonl": str(out_path),
    }
    (out_dir / "supplement_direct_structures_report.json").write_text(jdump(report), encoding="utf-8")
    conn.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
