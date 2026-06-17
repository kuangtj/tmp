#!/usr/bin/env python3
import os
import re
import sys
import json
import uuid
import argparse
from pathlib import Path
from collections import defaultdict, Counter

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn


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
    r = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return r is not None


def table_cols(conn, table):
    if not table_exists(conn, table):
        return set()
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def ensure_stg_agent(conn):
    if not table_exists(conn, "stg_agent"):
        conn.execute("""
        CREATE TABLE stg_agent (
            agent_id TEXT PRIMARY KEY,
            doc_id TEXT,
            name TEXT,
            agent_type TEXT,
            canonical_smiles TEXT,
            structure_diagram_refs TEXT,
            evidence_span TEXT,
            status TEXT,
            raw_output TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        return

    cols = table_cols(conn, "stg_agent")
    required = {
        "agent_id": "TEXT",
        "doc_id": "TEXT",
        "name": "TEXT",
        "agent_type": "TEXT",
        "canonical_smiles": "TEXT",
        "structure_diagram_refs": "TEXT",
        "evidence_span": "TEXT",
        "status": "TEXT",
        "raw_output": "TEXT",
        "created_at": "TEXT",
    }

    for c, typ in required.items():
        if c not in cols:
            conn.execute(f"ALTER TABLE stg_agent ADD COLUMN {c} {typ}")

    conn.commit()


def has_placeholder(smiles):
    s = smiles or ""

    if "*" in s:
        return True

    if re.search(r"\[[^\]]*R[^\]]*\]", s):
        return True

    if re.search(r"(^|[^A-Za-z])R\d*([^A-Za-z]|$)", s):
        return True

    return False


def valid_final_smiles(smiles):
    """
    Final compound-level SMILES must:
    - be RDKit valid
    - sanitize successfully
    - not contain dummy atoms
    - not contain obvious R placeholders
    - not contain '*'
    """
    s = (smiles or "").strip()

    if not s:
        return False, "", "empty_smiles"

    if has_placeholder(s):
        return False, "", "has_dummy_or_R_placeholder"

    try:
        mol = Chem.MolFromSmiles(s, sanitize=False)
        if mol is None:
            return False, "", "MolFromSmiles returned None"

        if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            return False, "", "contains_dummy_atom"

        Chem.SanitizeMol(mol)
        can = Chem.MolToSmiles(mol, canonical=True)

        if has_placeholder(can):
            return False, "", "canonical_smiles_has_placeholder"

        return True, can, ""

    except Exception as e:
        return False, "", str(e)


def load_human_corrections(path):
    """
    Optional JSON/JSONL input.

    Accepted JSONL line:
    {
      "compound_name": "compound 12",
      "human_corrected_smiles": "..."
    }
    """
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        return {}

    out = {}

    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            items = obj
        else:
            items = obj.get("items", [])
    else:
        items = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))

    for x in items:
        name = str(x.get("compound_name") or x.get("name") or "").strip()
        smi = str(x.get("human_corrected_smiles") or x.get("canonical_smiles") or "").strip()
        if name and smi:
            out[name] = smi

    return out


def load_relations(conn, doc_id):
    if not table_exists(conn, "stg_component_relation"):
        return []

    sql = """
    SELECT
        r.*,
        c.smiles,
        c.canonical_smiles AS candidate_canonical_smiles,
        c.raw_output AS candidate_raw_output,
        c.source_tool,
        q.qc_score,
        q.auto_decision,
        q.qc_flags_json,
        q.vlm_qc_json
    FROM stg_component_relation r
    LEFT JOIN stg_structure_candidate c
      ON r.candidate_id = c.candidate_id
    LEFT JOIN structure_qc_result q
      ON r.candidate_id = q.candidate_id
    WHERE r.doc_id=?
      AND COALESCE(r.compound_name, '') != ''
      AND COALESCE(r.relation_type, '') != 'no_relation'
    ORDER BY
        r.compound_name,
        r.review_required ASC,
        r.confidence DESC
    """
    return conn.execute(sql, (doc_id,)).fetchall()


def infer_agent_type(name, roles):
    n = (name or "").lower()
    roles = set(roles)

    if {"warhead", "E3_ligand", "linker"} & roles:
        return "heterobifunctional"

    if re.search(r"\bprotac\b|degrader|dbet|arv-|mz\d+|mt-\d+", n, re.I):
        return "heterobifunctional"

    return "small_molecule"


def choose_full_structure(rows):
    candidates = []

    for r in rows:
        if r["component_role"] != "full":
            continue

        smi = r["smiles"] or ""
        ok, can, err = valid_final_smiles(smi)
        if not ok:
            continue

        try:
            conf = float(r["confidence"] or 0)
        except Exception:
            conf = 0.0

        try:
            qc_score = float(r["qc_score"] or 0)
        except Exception:
            qc_score = 0.0

        candidates.append((conf, qc_score, can, r))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0]


def reconstruct_compound(compound_name, rows, human_corrections):
    roles = [r["component_role"] for r in rows]
    figure_refs = sorted({
        str(r["figure_ref"] or "").strip()
        for r in rows
        if str(r["figure_ref"] or "").strip()
    })

    evidence = " || ".join([
        str(r["evidence_text"] or "").strip()
        for r in rows
        if str(r["evidence_text"] or "").strip()
    ])[:500]

    # 1. human correction
    if compound_name in human_corrections:
        smi = human_corrections[compound_name]
        ok, can, err = valid_final_smiles(smi)
        if ok:
            return {
                "compound_name": compound_name,
                "canonical_smiles": can,
                "agent_type": infer_agent_type(compound_name, roles),
                "structure_diagram_refs": "||".join(figure_refs),
                "evidence_span": evidence or "Human corrected SMILES.",
                "status": "accepted_human_corrected",
                "review_required": False,
                "reason": "",
                "source": "human_corrected_smiles",
                "component_roles": sorted(set(roles)),
            }

        return {
            "compound_name": compound_name,
            "canonical_smiles": "",
            "agent_type": infer_agent_type(compound_name, roles),
            "structure_diagram_refs": "||".join(figure_refs),
            "evidence_span": evidence or "Human corrected SMILES invalid.",
            "status": "review_human_corrected_invalid",
            "review_required": True,
            "reason": err,
            "source": "human_corrected_smiles",
            "component_roles": sorted(set(roles)),
        }

    # 2. full structure
    chosen = choose_full_structure(rows)
    if chosen:
        conf, qc_score, can, r = chosen
        return {
            "compound_name": compound_name,
            "canonical_smiles": can,
            "agent_type": infer_agent_type(compound_name, roles),
            "structure_diagram_refs": "||".join(figure_refs),
            "evidence_span": r["evidence_text"] or evidence,
            "status": "accepted_full_structure",
            "review_required": False,
            "reason": "",
            "source": "full_structure",
            "component_roles": sorted(set(roles)),
            "selected_candidate_id": r["candidate_id"],
            "confidence": conf,
            "qc_score": qc_score,
        }

    # 3. component-only cases
    role_set = set(roles)

    if {"scaffold", "R_group"} & role_set:
        status = "review_component_reconstruction_required"
        reason = "Only scaffold/R-group/component relations found. Automatic stitching is not enabled in this minimal version."
    elif {"warhead", "linker", "E3_ligand"} & role_set:
        status = "review_warhead_linker_e3_reconstruction_required"
        reason = "Only warhead/linker/E3_ligand components found. Attachment point ordering is unresolved."
    else:
        status = "review_no_valid_full_structure"
        reason = "No valid full compound-level SMILES found."

    return {
        "compound_name": compound_name,
        "canonical_smiles": "",
        "agent_type": infer_agent_type(compound_name, roles),
        "structure_diagram_refs": "||".join(figure_refs),
        "evidence_span": evidence,
        "status": status,
        "review_required": True,
        "reason": reason,
        "source": "component_relations",
        "component_roles": sorted(set(roles)),
    }


def upsert_agent(conn, doc_id, rec):
    agent_id = uid("agent", doc_id, rec["compound_name"])

    raw_output = {
        "rt": "agent",
        "name": rec["compound_name"],
        "agent_type": rec["agent_type"],
        "canonical_smiles": rec.get("canonical_smiles", ""),
        "structure_diagram_refs": rec.get("structure_diagram_refs", ""),
        "evidence_span": rec.get("evidence_span", ""),
        "status": rec.get("status", ""),
        "review_required": rec.get("review_required", True),
        "reason": rec.get("reason", ""),
        "source": rec.get("source", ""),
        "component_roles": rec.get("component_roles", []),
        "selected_candidate_id": rec.get("selected_candidate_id", ""),
    }

    conn.execute(
        """
        INSERT OR REPLACE INTO stg_agent
        (
            agent_id,
            doc_id,
            name,
            agent_type,
            canonical_smiles,
            structure_diagram_refs,
            evidence_span,
            status,
            raw_output
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            agent_id,
            doc_id,
            rec["compound_name"],
            rec["agent_type"],
            rec.get("canonical_smiles", ""),
            rec.get("structure_diagram_refs", ""),
            rec.get("evidence_span", ""),
            rec.get("status", ""),
            jdump(raw_output),
        ),
    )

    out = dict(raw_output)
    out["agent_id"] = agent_id
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--human-corrections", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    conn = get_conn()
    ensure_stg_agent(conn)

    if args.overwrite:
        conn.execute("DELETE FROM stg_agent WHERE doc_id=?", (args.doc_id,))
        conn.commit()

    human_corrections = load_human_corrections(args.human_corrections)
    rows = load_relations(conn, args.doc_id)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["compound_name"]].append(r)

    out_dir = Path("data/staging") / args.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "stg_agent_draft.jsonl"
    report_path = out_dir / "compound_reconstruction_report.json"

    outputs = []
    stats = Counter()

    with open(jsonl_path, "w", encoding="utf-8") as fw:
        for compound_name, rels in sorted(grouped.items()):
            rec = reconstruct_compound(
                compound_name=compound_name,
                rows=rels,
                human_corrections=human_corrections,
            )

            agent = upsert_agent(conn, args.doc_id, rec)
            fw.write(jdump(agent) + "\n")
            outputs.append(agent)

            stats["agents"] += 1
            stats[f"status:{agent['status']}"] += 1
            stats[f"agent_type:{agent['agent_type']}"] += 1
            if agent.get("review_required"):
                stats["review_required"] += 1
            else:
                stats["accepted"] += 1

    conn.commit()
    conn.close()

    report = {
        "doc_id": args.doc_id,
        "num_compounds": len(grouped),
        "num_agents": len(outputs),
        "stats": dict(stats),
        "jsonl": str(jsonl_path),
        "table": "stg_agent",
        "note": (
            "Minimal Stage 9 reconstruction. Human corrected SMILES has highest priority. "
            "Valid full structures are accepted directly. Component-only reconstruction is marked for review because attachment points are unresolved."
        ),
    }

    report_path.write_text(jdump(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
