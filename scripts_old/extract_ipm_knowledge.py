#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import uuid
import argparse
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn



MODALITIES = {
    "PROTAC",
    "Molecular_Glue_Degrader",
    "Molecular_Glue_Stabilizer",
    "Hydrophobic_Tagging_Degrader",
    "SNIPER",
    "CIDEs",
    "UID",
    "LYTAC",
    "AbTAC",
    "KineTAC_or_EndoTAG",
    "MoDE_A",
    "GlueTAC",
    "Dual_Receptor_LYTAC",
    "SignalTAC",
    "AUTAC",
    "ATTEC",
    "AUTOTAC",
    "LD_ATTEC",
    "ATNC",
    "CMA_TAC_or_CMATAC",
    "Ab_CMA",
    "BacPROTAC",
    "MtPTAC",
    "RiboTAC_RNaseL",
    "RNaseH1_Gapmer_ASO",
    "DUBTAC",
    "TF_DUBTAC",
    "PHICS",
    "PhosTAC_or_PhosTAP",
    "PHORC_or_PhoRC",
    "RIPR",
    "AceTAG",
    "OGT_TAC",
    "OGA_TAC",
    "GlyTAC_or_DGlyTAC",
    "SUMOTAC_or_Other_PTM_TAC",
    "LEAPER",
    "RESTORE",
    "REPAIR",
    "RESCUE",
    "CIRTS_ADAR",
    "RIPTAC",
    "Nondegradative_Glue_or_PPI_Stabilizer",
    "PPI_Inducer",
    "BiTE",
    "TriTE_or_Multispecific_TCE",
    "NK_Engager_or_NKCE",
    "DC_T_or_Other_Immune_Bridging",
    "DAC",
    "Other",
}

RELATION_BASIS = {
    "design_class",
    "proposed_model",
    "mechanistic_validation",
}


RELATION_TYPES = {
    "targeted_degradation",
    "targeted_stabilization",
    "induced_proximity",
    "PPI_stabilization",
    "PTM_modulation",
    "RNA_modulation",
    "protein_state_modulation",
    "trafficking_rewiring",
    "cell_cell_engagement",
    "sequence_information_editing",
    "other",
}


MECHANISM_ROUTES = {
    "E3_ligase_recruitment",
    "molecular_glue_neosubstrate_recruitment",
    "hydrophobic_tagging",
    "direct_proteasome_recruitment",
    "proteasome_20S_ubiquitin_independent",
    "endocytosis_lysosome_recruitment",
    "macroautophagy_tethering",
    "chaperone_mediated_autophagy",
    "noncanonical_protease_recruitment",
    "RNA_nuclease_recruitment",
    "RNA_editing_recruitment",
    "DUB_recruitment",
    "kinase_recruitment",
    "phosphatase_recruitment",
    "KAT_recruitment",
    "HDAC_recruitment",
    "OGT_recruitment",
    "OGA_recruitment",
    "SUMO_or_other_PTM_enzyme_recruitment",
    "partner_protein_or_complex_induction",
    "immune_cell_engagement",
    "protein_quality_control_unspecified",
    "unknown",
    "other",
}


PROXIMITY_TOPOLOGIES = {
    "target_effector_inducer",
    "target_partner_inducer",
    "target_pathway_inducer",
    "target_receptor_inducer",
    "cargo_receptor_inducer",
    "rna_effector_inducer",
    "cell_cell_bridge",
    "protein_complex_stabilization",
    "no_named_effector",
    "unknown",
}


EFFECTOR_STATUS = {
    "explicit",
    "not_stated",
    "not_applicable",
    "inferred_pathway",
    "ambiguous",
}


OUTCOME_CLASSES = {
    "I_Protein_Abundance_Decrease",
    "II_Protein_Abundance_Increase",
    "III_State_Modulation",
    "IV_Sequence_Information_Editing",
    "V_Interaction_Spatial_Rewiring",
    "VI_Cell_Cell_Proximity",
}


MECHANISM_TAGS = {
    "E3_UPS",
    "Direct_Proteasome_Recruitment",
    "Proteasome_20S_Ub_Independent",
    "Endocytosis_Lysosome_eTPD",
    "Autophagy_Macroautophagy",
    "Chaperone_Mediated_Autophagy",
    "Noncanonical_Protease_System",
    "Nuclease_RNA_Decay",
    "DUB",
    "Kinase",
    "Phosphatase",
    "KAT",
    "HDAC",
    "OGT",
    "OGA",
    "SUMO_or_Other_PTM_Enzyme",
    "ADAR",
    "Partner_Protein_or_Complex",
    "T_Cell_CD3",
    "NK_Cell",
    "Other_Immune_Bridging",
    "Hydrophobic_Tagging",
    "Proteostasis_QC",
    "PPI_Stabilization",
    "PPI_Induction",
    "Cell_Cell_Bridging",
}


AGENT_TYPES = {
    "small_molecule",
    "heterobifunctional",
    "glue",
    "hydrophobic_tagging_degrader",
    "antibody",
    "oligo",
    "protein",
    "peptide",
    "cell_bridge",
    "conjugate",
    "construct",
    "mixture",
    "control",
    "other",
}


PARTICIPANT_ROLES = {
    "primary_target",
    "degradation_target",
    "stabilization_target",
    "regulated_target",
    "neosubstrate",
    "proximal_partner",
    "recruited_effector",
    "recruited_receptor",
    "trafficking_receptor",
    "autophagy_receptor_or_adapter",
    "proteasome_component",
    "nuclease_effector",
    "editing_effector",
    "PTM_writer",
    "PTM_eraser",
    "immune_target_antigen",
    "immune_effector_marker",
    "immune_cell",
    "anchor",
    "adapter",
    "scaffold",
    "cargo",
    "substrate",
    "binder_target",
    "assay_component",
    "other",
}


FUNCTIONAL_ROLES = {
    "target",
    "partner",
    "effector",
    "receptor",
    "anchor",
    "adapter",
    "scaffold",
    "enzyme",
    "nuclease",
    "editor",
    "ptm_writer",
    "ptm_eraser",
    "immune_marker",
    "cell",
    "cargo",
    "substrate",
    "assay_component",
    "other",
}


EFFECTOR_PARTICIPANT_ROLES = {
    "recruited_effector",
    "trafficking_receptor",
    "autophagy_receptor_or_adapter",
    "proteasome_component",
    "nuclease_effector",
    "editing_effector",
    "PTM_writer",
    "PTM_eraser",
    "immune_effector_marker",
}


BAD_PARTICIPANT_NAMES = {
    "protac",
    "protacs",
    "degrader",
    "degraders",
    "compound",
    "compounds",
    "molecule",
    "molecules",
    "hyt",
    "hydrophobic tagging",
    "hydrophobic tag",
    "tag",
    "linker",
    "warhead",
    "ligand",
    "degradation",
    "degradation activity",
    "dc50",
    "dmax",
    "e3 ligase",
    "ups",
    "proteasome",
    "lysosome",
    "autophagy",
    "protein quality control system",
}


def uid(prefix: str, *parts: Any) -> str:
    return prefix + "_" + uuid.uuid5(
        uuid.NAMESPACE_URL,
        "|".join(str(x) for x in parts),
    ).hex[:16]


def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


def clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def table_cols(conn, table: str) -> set:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def ensure_column(conn, table: str, col: str, typ: str) -> None:
    if col not in table_cols(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def require_cols(conn, table: str, cols: List[str]) -> None:
    actual = table_cols(conn, table)
    missing = [c for c in cols if c not in actual]
    if missing:
        raise RuntimeError(f"{table} missing columns: {missing}; actual={sorted(actual)}")

def ensure_tables(conn) -> None:
    # Stage 10 owns relation staging tables.
    conn.execute("""
    CREATE TABLE IF NOT EXISTS  stg_relation (
        stg_id TEXT PRIMARY KEY,
        relation_id TEXT,
        doc_id TEXT,

        relation_key TEXT,
        relation_type TEXT,
        relation_subtype TEXT,
        modality TEXT,

        inducer_name TEXT,
        matched_agent_name TEXT,

        mechanism_route TEXT,
        relation_basis TEXT,
        proximity_topology TEXT,
        effector_status TEXT,
        outcome_class TEXT,
        mechanism_tags TEXT,

        participants_json TEXT,
        evidence_span TEXT,

        confidence REAL,
        status TEXT,
        review_required INTEGER,
        qc_reasons TEXT,
        qc_warnings TEXT,

        record_json TEXT,
        raw_output TEXT,

        source_task_id TEXT,
        asset_id TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS stg_relation_participant (
        participant_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,

        participant_key TEXT NOT NULL,
        name TEXT NOT NULL,
        canonical_name TEXT,
        entity_type TEXT,

        relation_ids_json TEXT,
        role_entries_json TEXT,

        is_effector INTEGER,
        is_primary_readout_target INTEGER,

        ids_json TEXT,
        sequence_json TEXT,
        structure_json TEXT,
        standardization_json TEXT,

        evidence_span TEXT,
        evidence_spans_json TEXT,

        confidence REAL,
        status TEXT,
        review_required INTEGER,
        qc_reasons TEXT,
        qc_warnings TEXT,

        raw_output TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,

        UNIQUE(doc_id, participant_key)
    )
    """)

    conn.execute("""
    CREATE INDEX IF NOT EXISTS  idx_stg_relation_doc_key 
    ON stg_relation(doc_id, relation_key)
    """)

    conn.execute("""
    CREATE INDEX IF NOT EXISTS  idx_stg_relation_part_doc_key
    ON stg_relation_participant(doc_id, participant_key)
    """)

    conn.commit()


def preflight(conn) -> None:
    require_cols(conn, "planned_tasks", [
        "task_id", "doc_id", "asset_id", "asset_type",
        "task_type", "agents_json", "priority", "reason", "status",
    ])

    require_cols(conn, "raw_asset", [
        "asset_id", "doc_id", "asset_type", "page_no",
        "figure_ref", "table_ref", "file_path", "metadata_json",
    ])

    require_cols(conn, "raw_text_block", [
        "block_id", "doc_id", "page_no", "section", "text",
    ])

    require_cols(conn, "raw_figure", [
        "figure_id", "doc_id", "page_no", "figure_ref", "caption",
    ])

    require_cols(conn, "raw_table", [
        "table_id", "doc_id", "page_no", "table_ref", "table_json",
    ])

    require_cols(conn, "stg_component_relation", [
        "doc_id", "asset_id", "candidate_id", "compound_name",
        "component_role", "relation_type",
    ])

def participant_key(name: Any) -> str:
    x = clean(name).lower()
    x = x.replace("β", "beta")
    x = x.replace("α", "alpha")
    x = x.replace("κ", "kappa")
    x = re.sub(r"\bprotein\b", "", x)
    x = re.sub(r"\bgene\b", "", x)
    x = re.sub(r"[^a-z0-9]+", "", x)
    return x


def json_list(x: Any) -> List[Any]:
    if not x:
        return []
    if isinstance(x, list):
        return x
    try:
        v = json.loads(x)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def append_unique(old: Any, items: List[Any]) -> List[Any]:
    out = []
    seen = set()

    for x in json_list(old) + list(items or []):
        if x in ("", None):
            continue
        k = jdump(x) if isinstance(x, (dict, list)) else str(x)
        if k not in seen:
            seen.add(k)
            out.append(x)

    return out


def infer_participant_entity_type(name: str, entity_type_hint: str) -> str:
    n = clean(name).lower()
    h = clean(entity_type_hint).lower()

    if h in {
        "protein",
        "protein_complex",
        "rna",
        "dna",
        "oligo",
        "antibody",
        "peptide",
        "cell",
        "other",
    }:
        return h

    if any(x in n for x in ["sirna", "shrna", "mirna", "lncrna", "mrna", "rna"]):
        return "rna"

    if any(x in n for x in ["aso", "oligo", "oligonucleotide", "gapmer", "aptamer"]):
        return "oligo"

    if "dna" in n:
        return "dna"

    if any(x in n for x in ["antibody", "igg", "fab", "scfv", "nanobody"]):
        return "antibody"

    if any(x in n for x in ["complex", "crl", "vcb", "scf"]):
        return "protein_complex"

    if "cell" in n:
        return "cell"

    return "protein"
def normalize_name(x: Any) -> str:
    x = clean(x).lower()
    x = re.sub(r"^(compound|cpd|protac)\s+", "", x)
    x = re.sub(r"\s+", "", x)
    return x


def load_agents(conn, doc_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows = conn.execute("""
    SELECT agent_id, name, agent_type, canonical_smiles, status
    FROM stg_agent
    WHERE doc_id=?
      AND COALESCE(name, '') != ''
    """, (doc_id,)).fetchall()

    agents = []
    name_map = {}

    for r in rows:
        d = dict(r)
        agents.append(d)
        name_map[normalize_name(d["name"])] = d

    return agents, name_map


def load_component_names(conn, doc_id: str) -> List[str]:
    rows = conn.execute("""
    SELECT DISTINCT compound_name
    FROM stg_component_relation
    WHERE doc_id=?
      AND COALESCE(compound_name, '') != ''
    ORDER BY compound_name
    """, (doc_id,)).fetchall()

    return [r["compound_name"] for r in rows]


def match_agent(name: str, agent_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
    key = normalize_name(name)
    if key in agent_map:
        return agent_map[key]

    stripped = re.sub(r"^(compound|cpd|protac)", "", clean(name), flags=re.I).strip()
    return agent_map.get(normalize_name(stripped))


def load_tasks(conn, doc_id: str, task_types: List[str], limit: int = 0):
    marks = ",".join(["?"] * len(task_types))
    params = [doc_id] + task_types

    sql = f"""
    SELECT
        p.task_id,
        p.doc_id,
        p.asset_id,
        p.asset_type AS task_asset_type,
        p.task_type,
        p.agents_json,
        p.priority,
        p.reason,
        p.status,

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
      AND p.status='planned'

    ORDER BY
        CASE p.priority
            WHEN 'high' THEN 1
            WHEN 'medium' THEN 2
            ELSE 3
        END,
        a.page_no,
        p.task_type,
        p.task_id
    """

    if limit:
        sql += f" LIMIT {int(limit)}"

    return conn.execute(sql, params).fetchall()


def load_page_text(conn, doc_id: str, page_no: int) -> str:
    rows = conn.execute("""
    SELECT section, text
    FROM raw_text_block
    WHERE doc_id=?
      AND page_no=?
      AND (section=? OR section=? OR section=?)                  
    ORDER BY rowid
    """, (doc_id, page_no,"Text", "Section-header","Title")).fetchall()

    parts = []
    for r in rows:
        section = clean(r["section"])
        text = clean(r["text"])
        if text:
            parts.append(f"[{section}]\n{text}" if section else text)

    return "\n\n".join(parts)


def load_figures_for_asset(conn, doc_id: str, page_no: int, figure_ref: str):
    if figure_ref:
        rows = conn.execute("""
        SELECT figure_id, figure_ref, caption
        FROM raw_figure
        WHERE doc_id=?
          AND page_no=?
          AND figure_ref=?
        """, (doc_id, page_no, figure_ref)).fetchall()
    else:
        rows = conn.execute("""
        SELECT figure_id, figure_ref, caption
        FROM raw_figure
        WHERE doc_id=?
          AND page_no=?
        """, (doc_id, page_no)).fetchall()

    return [dict(r) for r in rows]


def load_tables_for_asset(conn, doc_id: str, page_no: int, table_ref: str):
    if table_ref:
        rows = conn.execute("""
        SELECT table_id, table_ref, table_json
        FROM raw_table
        WHERE doc_id=?
          AND page_no=?
          AND table_ref=?
        """, (doc_id, page_no, table_ref)).fetchall()
    else:
        rows = conn.execute("""
        SELECT table_id, table_ref, table_json
        FROM raw_table
        WHERE doc_id=?
          AND page_no=?
        """, (doc_id, page_no)).fetchall()

    return [dict(r) for r in rows]


def build_task_context(conn, task, max_chars: int) -> str:
    doc_id = task["doc_id"]
    page_no = task["page_no"]

    obj = {
        "planned_task": {
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "priority": task["priority"],
            "reason": task["reason"],
            "asset_id": task["asset_id"],
            "asset_type": task["asset_type"],
        },
        "asset": {
            "page_no": page_no,
            "figure_ref": task["figure_ref"],
            "table_ref": task["table_ref"],
            "metadata_json": task["metadata_json"],
        },
        "figures": load_figures_for_asset(conn, doc_id, page_no, task["figure_ref"]),
        "tables": load_tables_for_asset(conn, doc_id, page_no, task["table_ref"]),
        "page_text": load_page_text(conn, doc_id, page_no),
    }

    return jdump(obj)[:max_chars]



def build_article_text_evidence_context(
    conn,
    doc_id: str,
    max_chars: int = 120000,
    limit: int = 0,
) -> tuple[str, list[dict]]:
    """
    Build one article-level text input for Stage 10.

    Rules:
    - Use only planned_tasks.task_type = 'text_evidence'.
    - Sort by raw_asset.page_no, then task_id.
    - Use text from raw_text_block only.
    - Do not use image pixels, table JSON, WB quantification, or dose-response curves.
    - Return one concatenated context plus source task metadata.
    """

    sql = """
    SELECT
        p.task_id,
        p.doc_id,
        p.asset_id,
        p.task_type,
        p.priority,
        p.reason,
        p.status,
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
      AND p.task_type='text_evidence'
      AND p.status='planned'
    ORDER BY
        COALESCE(a.page_no, 999999),
        p.task_id
    """

    if limit:
        sql += f" LIMIT {int(limit)}"

    rows = conn.execute(sql, (doc_id,)).fetchall()

    blocks = []
    source_tasks = []
    total = 0

    for i, r in enumerate(rows, start=1):
        task = dict(r)
        page_no = task.get("page_no")

        page_text = load_page_text(conn, doc_id, page_no) if page_no is not None else ""
        page_text = page_text.strip()

        if not page_text:
            continue

        header = (
            f"[TEXT_EVIDENCE_TASK order={i} "
            f"task_id={task['task_id']} "
            f"asset_id={task['asset_id']} "
            f"page_no={page_no} "
            f"asset_type={task.get('asset_type', '')}]"
        )

        block = f"{header}\n{page_text}\n[/TEXT_EVIDENCE_TASK]"

        if total + len(block) + 2 > max_chars:
            break

        blocks.append(block)
        total += len(block) + 2

        source_tasks.append({
            "order": i,
            "task_id": task["task_id"],
            "asset_id": task["asset_id"],
            "page_no": page_no,
            "asset_type": task.get("asset_type", ""),
            "figure_ref": task.get("figure_ref", ""),
            "table_ref": task.get("table_ref", ""),
        })

    return "\n\n".join(blocks), source_tasks


def build_prompt(context: str, agents: List[Dict[str, Any]], component_names: List[str]) -> str:
    known_names = sorted({
        clean(a["name"])
        for a in agents
        if clean(a.get("name"))
    } | {
        clean(x)
        for x in component_names
        if clean(x)
    })

    return f"""
You are extracting induced-proximity core relations from ONE article-level ordered text_evidence context.

Return JSONL only.
No markdown. No commentary.
One JSON object per line.

Allowed rt:
relation

TASK
Extract mechanism-level or design-level induced-proximity medicine (IPM) relations.

Stage 10 relation extraction is intentionally permissive but evidence-grounded.
A Stage 10 relation is NOT a claim that the inducer is potent, effective, positive, or therapeutically useful.
A Stage 10 relation only means that a final inducer agent was designed, proposed, modeled, or experimentally tested as an induced-proximity agent for named biological participant(s).

Activity, potency, polarity, degradation percentage, DC50, Dmax, IC50, cell viability, dose response, time-course, WB quantification, and whether the relation is positive, negative, weak, moderate, or inactive belong to Stage 11 assay extraction.

CORE DEFINITION
A relation means:
one final inducer agent is designed, proposed, modeled, or experimentally shown to organize one or more biological participants through an induced-proximity mechanism.

Output one relation when the article-level context supports that:
- a final degrader, glue, engager, construct, oligo, antibody, biologic, or cell-bridging agent is designed to act on a named biological target;
- a final compound series is explicitly described as PROTACs, HyT molecules, molecular glues, LYTACs, DUBTACs, RiboTACs, BiTEs, NK engagers, PPI inducers, or another IPM modality;
- a final inducer is proposed to recruit, engage, stabilize, degrade, edit, traffic, bridge, or bring into proximity a named biological participant;
- a final inducer is shown, modeled, or proposed to form a ternary complex, quaternary complex, cell-cell bridge, or other induced-proximity complex;
- a final inducer has mechanism evidence such as target engagement, effector recruitment, ternary complex formation, proteasome/lysosome/autophagy/chaperone involvement, RNA nuclease/editing recruitment, PTM enzyme recruitment, PPI stabilization, or cell-cell engagement.

WEAK / INACTIVE RELATION RULE
Do NOT filter out weak, inactive, N.D., low-degradation, moderate, poor, failed, or negative compounds if the context supports that they are final IPM agents by design, class, figure/table title, SAR series, or mechanism description.

Specifically:
- Weak, inactive, N.D., low degradation, poor degradation, moderate degradation, no obvious degradation, or negative assay results MUST NOT invalidate a design-level relation.
- If the compound is a final IPM agent by design/class/table/figure/SAR context, extract the relation even when the assay result is weak or negative.
- Do not encode weak/negative activity in Stage 10 except indirectly through lower confidence or review_required when appropriate.
- The weak/negative outcome itself must be captured later in Stage 11 assay records.

SERIES EXPANSION RULE
If a table, figure, caption, SAR section, or design section defines a final compound series as IPM agents targeting a named biological target, output one relation per known final compound in that series.

Examples:
- "HyT compounds 14a−14f and 17a−17f targeting c-Met" -> output one relation per known final compound in that range.
- "PROTACs 22a−22g, 24, 26, and 28a−28c targeting c-Met" -> output one relation per known final compound in that range.

Do not collapse a compound series into one generic relation.
If Known compound names contains individual compounds from a range, output each individual final compound as a separate relation.

DO NOT OUTPUT A RELATION IF THE CONTEXT ONLY CONTAINS
- synthesis route
- reaction conditions
- intermediate preparation
- reagent list
- yield
- 1H NMR
- 13C NMR
- HRMS
- m/z
- ppm
- compound characterization only
- reference inhibitor only
- target ligand only
- E3 ligand only
- linker only
- hydrophobic tag only
- warhead only
- building block only

A chemical name plus NMR/HRMS/yield is not Stage 10 relation evidence.

FINAL INDUCER RULE
inducer_name must be the complete final molecule, biologic, oligo, antibody, construct, or cell-bridging agent that mediates the IPM relation.

Do not use any of the following as inducer_name:
- intermediate
- warhead
- linker
- target ligand
- E3 ligand
- CRBN ligand
- VHL ligand
- hydrophobic tag
- reagent
- reference inhibitor
- building block

Examples:
- "compound S1 was selected as the most suitable E3 ligand" -> no Stage 10 relation for S1.
- "tepotinib was selected as the c-Met ligand" -> no Stage 10 relation for tepotinib.
- "PROTACs 22a-22g targeting c-Met" -> valid design-level relations for 22a-22g if those names are in Known compound names.
- "compound 22b recruits CRBN to form a ternary complex with c-Met" -> valid mechanism-supported relation.

INDUCER NAME NORMALIZATION
- inducer_name must be exactly one name from Known compound names whenever possible.
- Normalize treatment phrases to the known compound name.

Examples:
- "compound 22b-treated" -> "22b"
- "compound 22b" -> "22b"
- "22b-treated cells" -> "22b"
- "100 nM 22b" -> "22b"

If no exact known compound name can be identified, return no line.

RELATION MULTIPLICITY
Output one relation per unique:
final inducer + mechanism_route + participant set

Do NOT output one relation per:
- assay condition
- dose
- time point
- cell line
- readout
- degradation percentage
- DC50
- Dmax
- IC50
- N.D.

For SAR/design tables, output one relation per final compound only when each row or series member corresponds to a final IPM agent and the table/title/context clearly defines the modality and target.

RELATION BASIS
Use relation_basis to describe why this relation is extracted.

Allowed relation_basis:
["design_class", "proposed_model", "mechanistic_validation"]

Definitions:
- design_class:
  The compound is a final IPM agent by design/class/series/table/figure title, but no detailed mechanism validation is shown for this specific inducer in the local context.

- proposed_model:
  The context provides docking, modeling, schematic, structural model, predicted ternary/proximity mode, or proposed induced-proximity mechanism.

- mechanistic_validation:
  The context provides experimental mechanism evidence, such as competitor rescue, E3 dependency, CRBN/VHL dependency, MG132/lysosome/autophagy inhibitor rescue, target engagement, ubiquitination, ternary complex assay, pull-down, CETSA, NanoBRET, SPR/ITC/MST ternary evidence, or other mechanism tests.

MECHANISM ROUTE DECISION RULES
- Use mechanism_route="E3_ligase_recruitment" for PROTACs and E3-recruiting molecular glue mechanisms.
- Use mechanism_route="hydrophobic_tagging" for HyT / hydrophobic-tagging degraders.
- Use mechanism_route="partner_protein_or_complex_induction" for PPI induction or PPI stabilization without degradation.
- Use the closest allowed mechanism_route from the enum.
- If unclear but still an IPM relation, use "unknown" or "other" only if those values are allowed.

Named E3 participant rule:
Add a named E3 participant such as CRBN, VHL, DCAF15, DCAF16, RNF114, IAP, MDM2, KEAP1, or β-TRCP only if at least one condition is met:

1. Specific-inducer evidence:
   The E3 is explicitly linked to the specific inducer by mechanism evidence.
   In this case, use relation_basis="mechanistic_validation" or "proposed_model".

2. Series-level design propagation:
   The article-level context explicitly states that a named E3 ligand/recruiter was selected for the designed PROTAC/E3-recruiting degrader series, and a figure/table/title/context defines a set of final compounds as PROTACs or E3-recruiting degraders targeting a named target.
   In this case, propagate the named E3 to each final PROTAC in that series.

Series-level E3 propagation is allowed only when:
- the E3 recruiter is explicitly named, such as CRBN ligand, VHL ligand, IAP ligand, MDM2 ligand, KEAP1 ligand, etc.;
- the final compounds are explicitly described as PROTACs or E3-recruiting degraders;
- the final compounds are present in Known compound names;
- there is no conflicting E3 recruiter assignment for the same compound series.

For propagated E3 relations:
- include the E3 participant with participant_role="recruited_effector" and functional_role="effector";
- set relation_basis="design_class";
- set confidence between 0.70 and 0.80;
- set review_required=true unless the exact compound-to-E3-recruiter mapping is directly stated or structurally unambiguous in the local context;
- the participant evidence_span for the E3 should cite the sentence that names the selected E3 recruiter;
- the participant evidence_span for the target should cite the sentence/table/figure title that defines the target or compound series.

If the context says PROTAC but does not name the E3 anywhere in the article-level context, do not invent CRBN/VHL. Keep only the target participant.

If the context only says proteasome-mediated or MG132 blocks degradation but no named E3 is stated, use mechanism_route="protein_quality_control_unspecified" unless the broader article-level context explicitly names the E3 recruiter for that inducer or series.

HyT rule:
- HyT means Hydrophobic Tagging.
- HyT is a modality/mechanism, not a participant or effector.
- For HyT compounds:
  modality="Hydrophobic_Tagging_Degrader"
  mechanism_route="hydrophobic_tagging"
  mechanism_tags="Hydrophobic_Tagging"
  participants usually include only the named degradation target.
- Do not use "HyT" as participant name.
- Do not invent CRBN/VHL/E3 for HyT compounds unless the article explicitly states a named effector.
- Do not discard HyT relations only because the assay activity is weak, N.D., low, or negative.

PARTICIPANT RULES
participants must be concrete biological entities.

Do not use modality names, compound classes, assay terms, chemical fragments, or pathway terms as participant names.

Forbidden participant names include:
PROTAC, degrader, compound, molecule, E3 ligase, proteasome, lysosome, degradation rate, DC50, Dmax, HyT, linker, warhead, tag, ligand.

Role assignment:
- For degraded target:
  participant_role="degradation_target"
  functional_role="target"

- For stabilized/increased target:
  participant_role="stabilization_target"
  functional_role="target"

- For generally regulated target:
  participant_role="regulated_target"
  functional_role="target"

- For recruited E3 / enzyme / nuclease / editor / receptor:
  use the most specific participant_role, such as recruited_effector, nuclease_effector, editing_effector, PTM_writer, PTM_eraser, trafficking_receptor, autophagy_receptor_or_adapter.
  functional_role="effector"

- For a protein only brought near another protein:
  participant_role="proximal_partner"
  functional_role="partner"

- For BiTE/TCE/NK engager:
  tumor antigen should be immune_target_antigen.
  CD3/CD16A/NKp46 or immune-side receptor should be immune_effector_marker.

EVIDENCE RULES
- evidence_span must be copied from the article-level text_evidence context.
- evidence_span should support the design class, proposed model, or mechanism relation.
- Good evidence spans include table titles, figure captions, design sentences, mechanism sentences, docking/modeling descriptions, proposed ternary-complex descriptions, or mechanistic validation sentences.
- Avoid using pure numeric readout spans as relation evidence.
- A sentence mentioning weak, moderate, poor, inactive, N.D., or low degradation may be used as relation evidence only if the same sentence or local context also identifies the final compound/series as an IPM modality targeting a named biological target.
- Avoid synthesis/NMR/HRMS/yield spans.
- Do not concatenate unrelated distant sentences to create artificial support.
- For series-level E3 propagation, evidence_span may combine one short E3-selection span and one short series/target-defining span using " ... " only when both spans support the same propagated relation.

Good weak-relation evidence examples:
- "We designed and synthesized several HyT molecules with different types of HyT and PROTACs with different types of linkers."
- "HyT molecules showed weak degradation activity."
- "Structures and c-Met Degradation Activity of HyT Compounds 14a−14f and 17a−17f"
- "Structures and c-Met Degradation Activity of PROTACs 22a−22g, 24, 26, and 28a−28c"

Bad weak-relation evidence examples:
- "DC50 > 1 μM" alone
- "N.D." alone
- "Dmax = 12%" alone
- a row containing only compound name, yield, NMR, HRMS, or m/z

CONFIDENCE AND REVIEW RULES
Use confidence as extraction confidence for the Stage 10 relation, not assay strength.

Suggested confidence:
- 0.90-0.98:
  specific inducer has direct experimental mechanism validation with named target and named effector.

- 0.80-0.89:
  specific inducer has proposed/modeling/ternary-complex evidence with named target and named effector.

- 0.70-0.80:
  design-class relation from clear final IPM compound series/table/figure/title/SAR context.

- 0.60-0.70:
  relation is likely but target, effector, or compound-series mapping is partially implicit.

Set review_required=true when:
- E3 is propagated from series-level design rather than directly shown for the specific inducer;
- target or effector assignment is implicit;
- relation is extracted from a broad compound range;
- evidence is design-class only and not mechanistically validated;
- the final compound-to-effector mapping may require human confirmation.

Set review_required=false when:
- the specific inducer, target, and effector are directly linked by clear mechanism/model/design evidence;
- the relation is a simple HyT/design-class relation with clear target and no named effector required.

ENUMS
Allowed relation_type:
{sorted(RELATION_TYPES)}

Allowed modality:
{sorted(MODALITIES)}

Allowed mechanism_route:
{sorted(MECHANISM_ROUTES)}

Allowed outcome_class:
{sorted(OUTCOME_CLASSES)}
Use || to join multiple outcome_class values.

Allowed mechanism_tags:
{sorted(MECHANISM_TAGS)}
Use || to join multiple mechanism_tags.

Allowed participant_role:
{sorted(PARTICIPANT_ROLES)}

Allowed functional_role:
{sorted(FUNCTIONAL_ROLES)}

SCHEMA
{{"rt":"relation","relation_type":"targeted_degradation","modality":"PROTAC","inducer_name":"22b","mechanism_route":"E3_ligase_recruitment","relation_basis":"mechanistic_validation","outcome_class":"I_Protein_Abundance_Decrease","mechanism_tags":"E3_UPS","participants":[{{"name":"c-Met","participant_role":"degradation_target","functional_role":"target","entity_type_hint":"protein","variant_text":"","evidence_span":""}},{{"name":"CRBN","participant_role":"recruited_effector","functional_role":"effector","entity_type_hint":"protein","variant_text":"","evidence_span":""}}],"evidence_span":"","confidence":0.0,"review_required":false}}

GOOD EXAMPLES

{{"rt":"relation","relation_type":"targeted_degradation","modality":"PROTAC","inducer_name":"22b","mechanism_route":"E3_ligase_recruitment","relation_basis":"mechanistic_validation","outcome_class":"I_Protein_Abundance_Decrease","mechanism_tags":"E3_UPS","participants":[{{"name":"c-Met","participant_role":"degradation_target","functional_role":"target","entity_type_hint":"protein","variant_text":"","evidence_span":"compound 22b can bind the c-Met kinase domain"}},{{"name":"CRBN","participant_role":"recruited_effector","functional_role":"effector","entity_type_hint":"protein","variant_text":"","evidence_span":"can recruit CRBN to form a ternary complex with c-Met kinase domain and compound 22b"}}],"evidence_span":"compound 22b can bind the c-Met kinase domain ... can recruit CRBN to form a ternary complex with c-Met kinase domain and compound 22b","confidence":0.9,"review_required":false}}

{{"rt":"relation","relation_type":"targeted_degradation","modality":"Hydrophobic_Tagging_Degrader","inducer_name":"14a","mechanism_route":"hydrophobic_tagging","relation_basis":"design_class","outcome_class":"I_Protein_Abundance_Decrease","mechanism_tags":"Hydrophobic_Tagging","participants":[{{"name":"c-Met","participant_role":"degradation_target","functional_role":"target","entity_type_hint":"protein","variant_text":"","evidence_span":"Structures and c-Met Degradation Activity of HyT Compounds 14a−14f and 17a−17f"}}],"evidence_span":"Structures and c-Met Degradation Activity of HyT Compounds 14a−14f and 17a−17f","confidence":0.75,"review_required":false}}

{{"rt":"relation","relation_type":"targeted_degradation","modality":"PROTAC","inducer_name":"22e","mechanism_route":"E3_ligase_recruitment","relation_basis":"design_class","outcome_class":"I_Protein_Abundance_Decrease","mechanism_tags":"E3_UPS","participants":[{{"name":"c-Met","participant_role":"degradation_target","functional_role":"target","entity_type_hint":"protein","variant_text":"","evidence_span":"Structures and c-Met Degradation Activity of PROTACs 22a−22g, 24, 26, and 28a−28c"}},{{"name":"CRBN","participant_role":"recruited_effector","functional_role":"effector","entity_type_hint":"protein","variant_text":"","evidence_span":"compound S1 was selected as the most suitable E3 ligand"}}],"evidence_span":"compound S1 was selected as the most suitable E3 ligand ... Structures and c-Met Degradation Activity of PROTACs 22a−22g, 24, 26, and 28a−28c","confidence":0.75,"review_required":true}}

BAD EXAMPLES. DO NOT OUTPUT THESE
- {{"rt":"relation","relation_type":"other","inducer_name":"","participants":[]}}
- inducer_name="compound 22b-treated"
- inducer_name="compound S1" when S1 is only an E3 ligand or CRBN ligand
- inducer_name="tepotinib" when tepotinib is only the target ligand/reference inhibitor
- relation based only on NMR/HRMS/yield/synthesis text
- participant name "HyT"
- participant name "PROTAC"
- participant name "E3 ligase" without a concrete named E3
- participant name "proteasome" when only a degradation pathway is described
- relation for an intermediate, linker, warhead, hydrophobic tag, or building block
- one relation per dose/time/cell-line assay condition
- one relation based only on DC50, Dmax, IC50, N.D., or degradation percentage

Known compound names from Stage 8/9:
{json.dumps(known_names[:500], ensure_ascii=False)}

ARTICLE TEXT_EVIDENCE CONTEXT:
{context}

""".strip()


def call_llm(client: OpenAI, model: str, prompt: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def parse_jsonl(text: str) -> List[Dict[str, Any]]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:jsonl|json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            x = json.loads(line)
            if isinstance(x, dict):
                out.append(x)
        except Exception:
            pass

    if out:
        return out

    try:
        x = json.loads(text)
        if isinstance(x, list):
            return [i for i in x if isinstance(i, dict)]
        if isinstance(x, dict):
            return [x]
    except Exception:
        pass

    return []


def evidence_ok(evidence: str, context: str) -> bool:
    ev = clean(evidence)
    if not ev or len(ev) < 8:
        return False

    ctx = clean(context)
    if ev in ctx:
        return True

    tokens = [t for t in re.split(r"\W+", ev) if len(t) >= 4]
    if len(tokens) < 4:
        return False

    ctx_lower = ctx.lower()
    hit = sum(1 for t in tokens[:12] if t.lower() in ctx_lower)
    return hit >= min(6, len(tokens))


def normalize_allowed(value: Any, allowed: set, default: str, reasons: List[str], reason: str) -> str:
    value = clean(value)
    if value not in allowed:
        reasons.append(reason)
        return default
    return value


def validate_joined_enum(
    rec: Dict[str, Any],
    field: str,
    allowed: set,
    reasons: List[str],
    missing_reason: str,
    invalid_reason: str,
) -> None:
    raw = clean(rec.get(field))
    vals = [x.strip() for x in raw.split("||") if x.strip()]
    valid = [x for x in vals if x in allowed]

    if not valid:
        reasons.append(missing_reason)
        rec[field] = ""
        return

    if len(valid) != len(vals):
        reasons.append(invalid_reason)

    rec[field] = "||".join(valid)


def make_relation_key(rec: Dict[str, Any]) -> str:
    inducer = clean(rec.get("inducer_name"))
    route = clean(rec.get("mechanism_route") or "unknown")

    parts = []
    for p in rec.get("participants", []):
        name = clean(p.get("name"))
        role = clean(p.get("participant_role"))
        if name:
            parts.append(f"{role}:{name}")

    return clean(f"{inducer}|{route}|{'||'.join(sorted(parts))}")


def validate_relation(
    rec: Dict[str, Any],
    context: str,
    agent_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    reasons = []

    def required_allowed(field: str, allowed: set, default: str, missing_reason: str, invalid_reason: str) -> str:
        raw = clean(rec.get(field))
        if not raw:
            reasons.append(missing_reason)
            rec[field] = default
            return default
        rec[field] = normalize_allowed(raw, allowed, default, reasons, invalid_reason)
        return rec[field]

    rec["relation_type"] = required_allowed(
        "relation_type", RELATION_TYPES, "other",
        "missing_relation_type", "invalid_relation_type",
    )

    rec["modality"] = required_allowed(
        "modality", MODALITIES, "Other",
        "missing_modality", "invalid_modality",
    )

    rec["mechanism_route"] = required_allowed(
        "mechanism_route", MECHANISM_ROUTES, "unknown",
        "missing_mechanism_route", "invalid_mechanism_route",
    )

    rec["relation_basis"] = required_allowed(
        "relation_basis", RELATION_BASIS, "design_class",
        "missing_relation_basis", "invalid_relation_basis",
    )

    rec["relation_subtype"] = clean(
        rec.get("relation_subtype")
        or rec.get("modality")
        or "Other"
    )

    raw_topology = clean(rec.get("proximity_topology"))
    rec["proximity_topology"] = (
        normalize_allowed(raw_topology, PROXIMITY_TOPOLOGIES, "unknown", reasons, "invalid_proximity_topology")
        if raw_topology else ""
    )

    raw_effector_status = clean(rec.get("effector_status"))
    rec["effector_status"] = (
        normalize_allowed(raw_effector_status, EFFECTOR_STATUS, "not_stated", reasons, "invalid_effector_status")
        if raw_effector_status else ""
    )

    validate_joined_enum(
        rec,
        "outcome_class",
        OUTCOME_CLASSES,
        reasons,
        "invalid_or_missing_outcome_class",
        "invalid_outcome_class_removed",
    )

    validate_joined_enum(
        rec,
        "mechanism_tags",
        MECHANISM_TAGS,
        reasons,
        "missing_mechanism_tags",
        "invalid_mechanism_tags_removed",
    )

    # Normalize inducer to Stage 8/9 final agent name.
    raw_inducer = clean(rec.get("inducer_name"))
    matched_agent = None

    if not raw_inducer:
        reasons.append("missing_inducer_name")
        rec["inducer_name"] = ""
    else:
        variants = [raw_inducer]

        v = re.sub(r"(?i)^compound\s+", "", raw_inducer).strip()
        v = re.sub(r"(?i)^cpd\.?\s+", "", v).strip()
        v = re.sub(r"(?i)^protac\s+", "", v).strip()
        v = re.sub(r"(?i)-treated\\b.*$", "", v).strip()
        v = re.sub(r"(?i)\\btreated\\b.*$", "", v).strip()
        v = re.sub(r"(?i)\\btreatment\\b.*$", "", v).strip()
        v = re.sub(r"\\([^)]*\\)", "", v).strip()
        if v:
            variants.append(v)

        for m in re.finditer(r"\\b[A-Za-z]?\\d+[A-Za-z]?\\b", raw_inducer):
            variants.append(m.group(0))

        seen_variants = []
        for v in variants:
            v = clean(v)
            if v and v not in seen_variants:
                seen_variants.append(v)

        for v in seen_variants:
            matched_agent = match_agent(v, agent_map)
            if matched_agent:
                rec["inducer_name"] = matched_agent["name"]
                break

        if not matched_agent:
            rec["inducer_name"] = raw_inducer
            reasons.append("inducer_not_aligned_to_stage9_agent")

    rec["matched_agent_name"] = matched_agent["name"] if matched_agent else ""

    participants = rec.get("participants")
    if not isinstance(participants, list):
        participants = []
        reasons.append("participants_not_list")

    clean_parts = []
    has_effector = False
    has_primary = False

    primary_roles = {
        "primary_target",
        "degradation_target",
        "stabilization_target",
        "regulated_target",
        "immune_target_antigen",
        "cargo",
        "substrate",
    }

    for p in participants:
        if not isinstance(p, dict):
            continue

        name = clean(p.get("name"))
        if not name:
            continue

        if name.lower() in BAD_PARTICIPANT_NAMES:
            reasons.append(f"invalid_participant_name:{name}")
            continue

        participant_role = normalize_allowed(
            p.get("participant_role") or "other",
            PARTICIPANT_ROLES,
            "other",
            reasons,
            f"invalid_participant_role:{name}",
        )

        functional_role = normalize_allowed(
            p.get("functional_role") or "other",
            FUNCTIONAL_ROLES,
            "other",
            reasons,
            f"invalid_functional_role:{name}",
        )

        is_effector = (
            as_bool(p.get("is_effector"))
            or participant_role in EFFECTOR_PARTICIPANT_ROLES
            or functional_role == "effector"
        )

        is_primary = (
            as_bool(p.get("is_primary_readout_target"))
            or participant_role in primary_roles
        )

        if is_effector:
            has_effector = True
            if participant_role not in EFFECTOR_PARTICIPANT_ROLES and functional_role != "effector":
                reasons.append(f"effector_role_inconsistent:{name}")

        if is_primary:
            has_primary = True

        clean_parts.append({
            "name": name,
            "participant_role": participant_role,
            "functional_role": functional_role,
            "entity_type_hint": clean(p.get("entity_type_hint") or "protein"),
            "is_effector": is_effector,
            "is_primary_readout_target": is_primary,
            "variant_text": clean(p.get("variant_text")),
            "evidence_span": clean(p.get("evidence_span") or rec.get("evidence_span")),
        })

    if not clean_parts:
        reasons.append("missing_participants")

    if not has_primary:
        reasons.append("missing_primary_target_or_readout_participant")

    if not rec["effector_status"]:
        if rec["mechanism_route"] == "hydrophobic_tagging":
            rec["effector_status"] = "not_applicable"
        elif has_effector:
            rec["effector_status"] = "explicit"
        else:
            rec["effector_status"] = "not_stated"

    if rec["effector_status"] == "explicit" and not has_effector:
        reasons.append("effector_status_explicit_but_no_effector_participant")

    if not rec["proximity_topology"]:
        if rec["mechanism_route"] == "hydrophobic_tagging":
            rec["proximity_topology"] = "no_named_effector"
        elif has_effector:
            rec["proximity_topology"] = "target_effector_inducer"
        else:
            rec["proximity_topology"] = "unknown"

    if rec["mechanism_route"] == "hydrophobic_tagging":
        rec["effector_status"] = "not_applicable"
        rec["proximity_topology"] = "no_named_effector"
        if has_effector:
            reasons.append("hydrophobic_tagging_should_not_have_named_effector")
        if rec["modality"] == "Other":
            rec["modality"] = "Hydrophobic_Tagging_Degrader"

    evidence = clean(rec.get("evidence_span"))
    warnings = []

    # Evidence weak matching is not a blocking QC failure.
    # OCR, table serialization, and context truncation often break exact evidence matching.
    if not evidence:
        warnings.append("missing_evidence_span")
    elif not evidence_ok(evidence, context):
        warnings.append("evidence_weak_match")

    rec["participants"] = clean_parts
    rec["participants_json"] = jdump(clean_parts)
    rec["evidence_span"] = evidence
    rec["relation_key"] = make_relation_key(rec)

    rec["qc_reasons"] = reasons
    rec["qc_warnings"] = warnings

    rec["review_required"] = bool(as_bool(rec.get("review_required")) or reasons)

    if rec["review_required"]:
        rec["status"] = "review_required"
    elif warnings:
        rec["status"] = "auto_pass_with_warning"
    else:
        rec["status"] = "auto_pass"

    return rec



def validate_record(
    rec: Dict[str, Any],
    context: str,
    agent_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    rt = clean(rec.get("rt"))
    rec["rt"] = rt

    try:
        rec["confidence"] = float(rec.get("confidence", 0) or 0)
    except Exception:
        rec["confidence"] = 0.0

    if rt != "relation":
        return None

    return validate_relation(rec, context, agent_map)


def upsert_relation(conn, doc_id: str, rec: Dict[str, Any], task) -> str:
    relation_key = rec.get("relation_key")
    stg_id = uid("stg_relation", doc_id, relation_key)
    relation_id = uid("relation", doc_id, relation_key)

    conn.execute("""
    INSERT OR REPLACE INTO stg_relation
    (
        stg_id,
        relation_id,
        doc_id,

        relation_key,
        relation_type,
        relation_subtype,
        modality,

        inducer_name,
        matched_agent_name,

        mechanism_route,
        relation_basis,
        proximity_topology,
        effector_status,
        outcome_class,
        mechanism_tags,

        participants_json,
        evidence_span,

        confidence,
        status,
        review_required,
        qc_reasons,
        qc_warnings,

        record_json,
        raw_output,

        source_task_id,
        asset_id
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        stg_id,
        relation_id,
        doc_id,

        relation_key,
        rec.get("relation_type", ""),
        rec.get("relation_subtype", ""),
        rec.get("modality", ""),

        rec.get("inducer_name", ""),
        rec.get("matched_agent_name", ""),

        rec.get("mechanism_route", ""),
        rec.get("relation_basis", ""),
        rec.get("proximity_topology", ""),
        rec.get("effector_status", ""),
        rec.get("outcome_class", ""),
        rec.get("mechanism_tags", ""),

        rec.get("participants_json", ""),
        rec.get("evidence_span", ""),

        rec.get("confidence", 0),
        rec.get("status", ""),
        int(rec.get("review_required", False)),
        jdump(rec.get("qc_reasons", [])),
        jdump(rec.get("qc_warnings", [])),

        jdump(rec),
        jdump(rec),

        task["task_id"],
        task["asset_id"],
    ))

    return relation_id

def insert_relation_participants(conn, doc_id: str, rec: Dict[str, Any], relation_id: str) -> None:
    """
    stg_relation_participant is an article-level participant index.

    One participant name appears once per article:
        UNIQUE(doc_id, participant_key)

    Relation-specific information is accumulated in:
        relation_ids_json
        role_entries_json
        evidence_spans_json
    """

    relation_key = clean(rec.get("relation_key"))

    for p in rec.get("participants", []):
        name = clean(p.get("name"))
        if not name:
            continue

        pkey = participant_key(name)
        if not pkey:
            continue

        participant_id = uid("participant", doc_id, pkey)

        participant_role = clean(p.get("participant_role"))
        functional_role = clean(p.get("functional_role"))
        entity_type_hint = clean(p.get("entity_type_hint") or "protein")
        entity_type = infer_participant_entity_type(name, entity_type_hint)

        is_effector = int(as_bool(p.get("is_effector")))
        is_primary = int(as_bool(p.get("is_primary_readout_target")))

        evidence_span = clean(p.get("evidence_span") or rec.get("evidence_span"))

        role_entry = {
            "relation_id": relation_id,
            "relation_key": relation_key,
            "inducer_name": rec.get("inducer_name", ""),
            "modality": rec.get("modality", ""),
            "mechanism_route": rec.get("mechanism_route", ""),
            "participant_role": participant_role,
            "functional_role": functional_role,
            "entity_type_hint": entity_type_hint,
            "is_effector": bool(is_effector),
            "is_primary_readout_target": bool(is_primary),
            "variant_text": clean(p.get("variant_text")),
            "evidence_span": evidence_span,
        }

        old = conn.execute("""
        SELECT
            relation_ids_json,
            role_entries_json,
            evidence_spans_json,
            confidence
        FROM stg_relation_participant
        WHERE doc_id=?
          AND participant_key=?
        """, (doc_id, pkey)).fetchone()

        if old:
            relation_ids = append_unique(old["relation_ids_json"], [relation_id])
            role_entries = append_unique(old["role_entries_json"], [role_entry])
            evidence_spans = append_unique(
                old["evidence_spans_json"],
                [evidence_span] if evidence_span else [],
            )

            confidence = max(
                float(old["confidence"] or 0),
                float(rec.get("confidence", 0) or 0),
            )

            conn.execute("""
            UPDATE stg_relation_participant
            SET
                relation_ids_json=?,
                role_entries_json=?,

                is_effector=CASE
                    WHEN is_effector=1 OR ?=1 THEN 1
                    ELSE 0
                END,
                is_primary_readout_target=CASE
                    WHEN is_primary_readout_target=1 OR ?=1 THEN 1
                    ELSE 0
                END,

                evidence_span=CASE
                    WHEN COALESCE(evidence_span, '') = '' THEN ?
                    ELSE evidence_span
                END,
                evidence_spans_json=?,

                confidence=?,
                status=CASE
                    WHEN status='review_required' OR ?='review_required' THEN 'review_required'
                    WHEN status='auto_pass_with_warning' OR ?='auto_pass_with_warning' THEN 'auto_pass_with_warning'
                    ELSE 'auto_pass'
                END,

                raw_output=?
            WHERE doc_id=?
              AND participant_key=?
            """, (
                jdump(relation_ids),
                jdump(role_entries),

                is_effector,
                is_primary,

                evidence_span,
                jdump(evidence_spans),

                confidence,
                rec.get("status", ""),
                rec.get("status", ""),

                jdump({
                    "merge_mode": "stage10_deduplicated_participant_update",
                    "latest_relation_id": relation_id,
                    "latest_relation_key": relation_key,
                    "latest_participant": p,
                }),

                doc_id,
                pkey,
            ))

        else:
            conn.execute("""
            INSERT INTO stg_relation_participant
            (
                participant_id,
                doc_id,

                participant_key,
                name,
                canonical_name,
                entity_type,

                relation_ids_json,
                role_entries_json,

                is_effector,
                is_primary_readout_target,

                ids_json,
                sequence_json,
                structure_json,
                standardization_json,

                evidence_span,
                evidence_spans_json,

                confidence,
                status,
                review_required,
                qc_reasons,
                qc_warnings,

                raw_output
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                participant_id,
                doc_id,

                pkey,
                name,
                "",
                entity_type,

                jdump([relation_id]),
                jdump([role_entry]),

                is_effector,
                is_primary,

                "{}",
                "{}",
                "{}",
                "{}",

                evidence_span,
                jdump([evidence_span] if evidence_span else []),

                rec.get("confidence", 0),
                rec.get("status", ""),
                0,
                "[]",
                "[]",

                jdump({
                    "merge_mode": "stage10_deduplicated_participant_insert",
                    "relation_id": relation_id,
                    "relation_key": relation_key,
                    "participant": p,
                }),
            ))

            
def insert_record(conn, doc_id: str, rec: Dict[str, Any], task) -> None:
    if rec["rt"] != "relation":
        return

    relation_id = upsert_relation(conn, doc_id, rec, task)
    insert_relation_participants(conn, doc_id, rec, relation_id)



def rec_key(rec: Dict[str, Any]) -> Tuple[str, str]:
    if rec["rt"] == "relation":
        return ("relation", rec.get("relation_key"))
    return (rec["rt"], jdump(rec))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "ipm-llm"))
    ap.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    ap.add_argument("--max-context-chars", type=int, default=640000)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of text_evidence tasks for debugging.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = OpenAI(
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
    )

    conn = get_conn()
    ensure_tables(conn)
    preflight(conn)

    if args.overwrite:
        conn.execute("DELETE FROM stg_relation WHERE doc_id=?", (args.doc_id,))
        conn.execute("DELETE FROM stg_relation_participant WHERE doc_id=?", (args.doc_id,))
        conn.commit()

    agents, agent_map = load_agents(conn, args.doc_id)
    component_names = load_component_names(conn, args.doc_id)

    article_context, source_tasks = build_article_text_evidence_context(
        conn=conn,
        doc_id=args.doc_id,
        max_chars=args.max_context_chars,
        limit=args.limit,
    )

    out_dir = Path("data/staging") / args.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    article_context_path = out_dir / "stage10_article_text_evidence_context.txt"
    raw_path = out_dir / "stage10_ipm_relations_raw_llm.jsonl"
    valid_path = out_dir / "stage10_ipm_relations_validated.jsonl"
    report_path = out_dir / "stage10_ipm_relations_report.json"

    article_context_path.write_text(article_context, encoding="utf-8")

    stats = Counter()
    seen = set()
    valid_records = []

    pseudo_task = {
        "task_id": "stage10_article_text_evidence",
        "asset_id": "stage10_article_text_evidence",
        "task_type": "text_evidence_article",
    }

    if not article_context.strip():
        stats["empty_article_text_evidence_context"] += 1
        raw_text = ""
    else:
        prompt = build_prompt(article_context, agents, component_names)

        try:
            raw_text = call_llm(
                client=client,
                model=args.llm_model,
                prompt=prompt,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            stats["llm_error"] += 1
            raw_text = ""
            raw_path.write_text(
                jdump({
                    "doc_id": args.doc_id,
                    "source_task_type": "text_evidence_article",
                    "error": str(e),
                    "num_text_evidence_tasks": len(source_tasks),
                    "article_context_chars": len(article_context),
                    "article_context_path": str(article_context_path),
                }) + "\n",
                encoding="utf-8",
            )

    with open(raw_path, "w", encoding="utf-8") as fraw, open(valid_path, "w", encoding="utf-8") as fvalid:
        fraw.write(jdump({
            "doc_id": args.doc_id,
            "source_task_type": "text_evidence_article",
            "num_text_evidence_tasks": len(source_tasks),
            "source_tasks": source_tasks,
            "article_context_chars": len(article_context),
            "article_context_path": str(article_context_path),
            "raw_text": raw_text,
        }) + "\n")

        for rec in parse_jsonl(raw_text):
            rec = validate_record(rec, article_context, agent_map)
            if not rec:
                stats["invalid_or_unsupported_rt"] += 1
                continue

            rec["_source_task_id"] = pseudo_task["task_id"]
            rec["_source_asset_id"] = pseudo_task["asset_id"]
            rec["_source_task_type"] = pseudo_task["task_type"]
            rec["_source_text_evidence_task_ids"] = [x["task_id"] for x in source_tasks]

            key = rec_key(rec)
            if key in seen:
                stats["deduped"] += 1
                continue

            seen.add(key)
            valid_records.append(rec)
            fvalid.write(jdump(rec) + "\n")

            stats[f"rt:{rec['rt']}"] += 1
            stats[f"status:{rec.get('status', '')}"] += 1
            stats[f"basis:{rec.get('relation_basis', '')}"] += 1
            stats[f"modality:{rec.get('modality', '')}"] += 1
            stats[f"mechanism_route:{rec.get('mechanism_route', '')}"] += 1

            if rec.get("review_required"):
                stats["review_required"] += 1
            else:
                stats["auto_pass"] += 1

            if not args.dry_run:
                insert_record(conn, args.doc_id, rec, pseudo_task)

        if not args.dry_run:
            conn.commit()

    conn.close()

    report = {
        "doc_id": args.doc_id,
        "mode": "single_article_text_evidence_input",
        "num_text_evidence_tasks": len(source_tasks),
        "article_context_chars": len(article_context),
        "article_context_path": str(article_context_path),
        "num_valid_records": len(valid_records),
        "stats": dict(stats),
        "raw_llm_jsonl": str(raw_path),
        "validated_jsonl": str(valid_path),
        "tables": [
            "stg_relation",
            "stg_relation_participant",
        ],
        "note": (
            "Stage 10 concatenates all text_evidence task texts by page order into one article-level input, "
            "calls the LLM once, and extracts article-level IPM relations. "
            "It does not use raw images, tables, WB figures, dose-response curves, or assay quantification."
        ),
    }

    report_path.write_text(jdump(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()