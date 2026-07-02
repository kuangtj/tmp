#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter

import requests
from tqdm import tqdm
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn


ENTITY_TYPES = {
    "protein",
    "protein_complex",
    "rna",
    "dna",
    "oligo",
    "antibody",
    "peptide",
    "cell",
    "other",
}

TEXT_FILE_SUFFIXES = {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl"}

UNIPROT_RE = re.compile(
    r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b"
)
PDB_RE = re.compile(r"\b[0-9][A-Za-z0-9]{3}\b")
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$", re.I)
PDB_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZUO]+$", re.I)
NA_RE = re.compile(r"^[ACGTU]+$", re.I)

TAXON_NAME_MAP = {
    "human": "9606",
    "homo sapiens": "9606",
    "h. sapiens": "9606",
    "mouse": "10090",
    "mus musculus": "10090",
    "m. musculus": "10090",
    "rat": "10116",
    "rattus norvegicus": "10116",
    "r. norvegicus": "10116",
    "e. coli": "562",
    "escherichia coli": "562",
    "chinese hamster": "10029",
    "cricetulus griseus": "10029",
    "african green monkey": "60711",
    "chlorocebus sabaeus": "60711",
}


def clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


def jload(x: Any, default: Any) -> Any:
    if x in ("", None):
        return default
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return default


def require_cols(conn, table: str, cols: List[str]) -> None:
    actual = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    missing = [c for c in cols if c not in actual]
    if missing:
        raise RuntimeError(f"{table} missing columns: {missing}; actual={sorted(actual)}")


def preflight(conn) -> None:
#    require_cols(conn, "stg_relation_participant", [
#        "participant_id",
#        "doc_id",
#        "participant_key",
#        "name",
#        "canonical_name",
#        "entity_type",
#        "relation_ids_json",
#        "role_entries_json",
#        "ids_json",
#        "sequence_json",
#        "structure_json",
#        "standardization_json",
#        "evidence_span",
#        "evidence_spans_json",
#        "confidence",
#        "status",
#        "review_required",
#        "qc_reasons",
#        "qc_warnings",
#        "raw_output",
#    ])
    # 把这段校验要求删掉，或者改为新版的列名
    require_cols(conn, "stg_relation_participant", [
        "participant_id", "entity_name", "relation_id", "evidence_text", "qc_reasons_json"
    ])

    require_cols(conn, "raw_text_block", [
        "block_id", "doc_id", "page_no", "section", "text",
    ])

    require_cols(conn, "raw_table", [
        "table_id", "doc_id", "page_no", "table_ref", "table_json",
    ])

    require_cols(conn, "raw_figure", [
        "figure_id", "doc_id", "page_no", "figure_ref", "caption",
    ])

    require_cols(conn, "raw_asset", [
        "asset_id", "doc_id", "asset_type", "file_path", "metadata_json",
    ])


def norm_key(x: Any) -> str:
    x = clean(x).lower()
    x = x.replace("β", "beta").replace("α", "alpha").replace("κ", "kappa")
    x = re.sub(r"\bprotein\b|\bgene\b", "", x)
    x = re.sub(r"[^a-z0-9]+", "", x)
    return x


def dedupe(xs: List[Any]) -> List[Any]:
    out, seen = [], set()
    for x in xs:
        if x in ("", None):
            continue
        k = jdump(x) if isinstance(x, (dict, list)) else str(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def merge_dict_lists(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        old = out.get(k, [])
        if isinstance(old, list) and isinstance(v, list):
            out[k] = dedupe(old + v)
        elif v not in ("", None, [], {}):
            out[k] = v
    return out


def resolve_path(path: Any) -> Path:
    p = Path(clean(path))
    if not p.is_absolute():
        p = ROOT / p
    return p


def read_text_file(path: Any, max_chars: int = 30000) -> str:
    p = resolve_path(path)
    if not p.exists() or not p.is_file() or p.suffix.lower() not in TEXT_FILE_SUFFIXES:
        return ""

    try:
        return p.read_text(encoding="utf-8")[:max_chars]
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8-sig", errors="ignore")[:max_chars]
    except Exception:
        return ""


def empty_ids() -> Dict[str, List[str]]:
    return {
        "uniprot": [],
        "pdb": [],
        "hgnc": [],
        "ncbi_gene": [],
        "ensembl": [],
        "rnacentral": [],
        "chembl_target": [],
        "other": [],
    }


def empty_seq() -> Dict[str, Any]:
    return {
        "aa_sequence": [],
        "dna_sequence": [],
        "rna_sequence": [],
        "oligo_sequence": [],
        "modified_oligo_sequence": [],
        "sequence_note": "",
    }


def empty_structure() -> Dict[str, Any]:
    return {
        "pdb_ids": [],
        "structure_source": "",
        "construct": "",
        "domain": "",
        "mutation": "",
        "variant": "",
        "isoform": "",
        "organism": "",
        "structure_note": "",
    }


def normalize_taxon_id(x: Any) -> str:
    x = clean(x)
    return x if re.fullmatch(r"\d+", x) else ""


def infer_taxon_from_organism_name(name: Any) -> str:
    n = clean(name).lower().replace("_", " ")
    n = re.sub(r"\s+", " ", n)
    return TAXON_NAME_MAP.get(n, "")


def normalize_organism_json(x: Any) -> Dict[str, Any]:
    if not isinstance(x, dict):
        x = {}

    organism = clean(
        x.get("organism")
        or x.get("organism_name")
        or x.get("species")
        or x.get("scientific_name")
    )

    taxon_id = normalize_taxon_id(
        x.get("taxon_id")
        or x.get("ncbi_taxon_id")
        or x.get("taxonomy_id")
    )

    if not taxon_id and organism:
        taxon_id = infer_taxon_from_organism_name(organism)

    try:
        confidence = float(x.get("confidence", 0) or 0)
    except Exception:
        confidence = 0.0

    return {
        "organism": organism,
        "taxon_id": taxon_id,
        "basis": clean(x.get("basis") or x.get("organism_basis")),
        "evidence_span": clean(x.get("evidence_span") or x.get("evidence")),
        "confidence": confidence,
    }


def get_taxon_id(rec: Dict[str, Any], default_taxon_id: str) -> str:
    std = rec.get("standardization_json") or {}
    org = normalize_organism_json(std.get("organism_json") or {})
    return org.get("taxon_id") or default_taxon_id


def has_sequence_info(rec: Dict[str, Any]) -> bool:
    seq = rec.get("sequence_json") or {}
    return bool(
        seq.get("aa_sequence")
        or seq.get("dna_sequence")
        or seq.get("rna_sequence")
        or seq.get("oligo_sequence")
        or seq.get("modified_oligo_sequence")
        or seq.get("modified_sequence")
    )

#def load_participants(conn, doc_id: str) -> List[Dict[str, Any]]:
#    rows = conn.execute("""
#    SELECT *
#    FROM stg_relation_participant
#    WHERE doc_id=?
#      AND COALESCE(name, '') != ''
#    ORDER BY name
#    """, (doc_id,)).fetchall()
#
#    return [dict(r) for r in rows]

def load_participants(conn, doc_id):
    rows = conn.execute("""
        SELECT 
            participant_id,                      -- 保留原名给 line 1605 用
            participant_id AS participant_key,   -- 伪装成 participant_key 给旧逻辑用
            entity_name AS name,                 -- 将 entity_name 伪装成 name
            '[]' AS relation_ids_json,           -- 伪造空数组
            '[]' AS role_entries_json,           -- 伪造空数组
            evidence_text AS evidence_span,      -- 将 evidence_text 伪装成 evidence_span
            '[]' AS evidence_spans_json,         -- 伪造空数组
            qc_reasons_json AS qc_reasons,       -- 映射警告字段
            '[]' AS qc_warnings
        FROM stg_relation_participant
        WHERE doc_id = ?
    """, (doc_id,)).fetchall()
    
    # 这一行极其重要！把你查到的数据变成字典列表返回给外面的 main 函数
    return [dict(r) for r in rows]

def participant_payload(participants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in participants:
        out.append({
            "participant_id": p["participant_id"],
            "participant_key": p["participant_key"],
            "name": p["name"],
            "entity_type": p.get("entity_type") or "",
            "relation_ids_json": jload(p.get("relation_ids_json"), []),
            "role_entries_json": jload(p.get("role_entries_json"), []),
        })
    return out


def add_block(blocks: List[str], title: str, body: Any, max_body_chars: int, budget: Dict[str, int]) -> None:
    if not isinstance(body, str):
        body = jdump(body)

    body = body.strip()
    if not body:
        return

    body = body[:max_body_chars]
    block = f"\n\n## {title}\n{body}"

    if budget["used"] + len(block) > budget["max"]:
        return

    blocks.append(block)
    budget["used"] += len(block)


ID_CONTEXT_RE = re.compile(
    r"\b("
    r"PDB|PDB\s*ID|PDB\s*code|Protein\s+Data\s+Bank|"
    r"UniProt|UniProtKB|Swiss-Prot|accession|"
    r"HGNC|NCBI|RefSeq|GenBank|Ensembl|RNAcentral|ChEMBL|"
    r"taxonomy|taxon|taxonomy\s*ID|NCBI\s*Taxonomy|"
    r"crystal\s+structure|crystal\s+structures|"
    r"structure\s+of|structures\s+of|"
    r"docking|docking\s+study|molecular\s+docking|"
    r"homology\s+model|model\s+of|models\s+of|"
    r"sequence|amino\s+acid\s+sequence|protein\s+sequence|"
    r"oligo|oligonucleotide|siRNA|sgRNA|gRNA|guide\s+RNA|ASO|"
    r"organism|species|human|Homo\s+sapiens|mouse|Mus\s+musculus|"
    r"rat|Rattus\s+norvegicus|E\.?\s*coli|Escherichia\s+coli"
    r")\b",
    re.I,
)


def split_sentences(text: str) -> List[str]:
    text = clean(text)
    if not text:
        return []

    # Keep it simple and robust for OCR text.
    parts = re.split(r"(?<=[。！？.!?])\s+", text)
    out = []
    for p in parts:
        p = clean(p)
        if p:
            out.append(p)
    return out


def participant_aliases_for_context(participants: List[Dict[str, Any]]) -> List[str]:
    aliases = set()

    for p in participants:
        for k in ["name", "canonical_name"]:
            v = clean(p.get(k))
            if v:
                aliases.add(v)

        # Add common aliases for known proteins.
        n = norm_key(p.get("name"))
        if n in {"crbn", "cereblon"}:
            aliases.update(["CRBN", "cereblon", "Cereblon"])

        if n in {"cmet", "met", "hgfr", "hepatocytegrowthfactorreceptor"}:
            aliases.update([
                "c-Met",
                "cMet",
                "MET",
                "HGFR",
                "hepatocyte growth factor receptor",
            ])

    # Remove very short or very generic aliases.
    bad = {
        "protein",
        "gene",
        "cell",
        "target",
        "effector",
        "compound",
        "ligand",
        "receptor",
    }

    out = []
    for a in aliases:
        aa = clean(a)
        if not aa:
            continue
        if aa.lower() in bad:
            continue
        if len(aa) < 3:
            continue
        out.append(aa)

    # Long aliases first helps matching more specific names.
    return sorted(set(out), key=len, reverse=True)


def mentions_participant(text: str, aliases: List[str]) -> bool:
    if not text:
        return False

    for a in aliases:
        if not a:
            continue

        # For short uppercase aliases like MET/CRBN, use word boundary.
        if len(a) <= 5:
            if re.search(rf"\b{re.escape(a)}\b", text, re.I):
                return True
        else:
            if a.lower() in text.lower():
                return True

    return False


def has_id_or_sequence_signal(text: str) -> bool:
    if not text:
        return False

    if ID_CONTEXT_RE.search(text):
        return True

    # Explicit PDB-like IDs alone are not enough because years like 2024 match.
    # Keep PDB-like tokens only if nearby structural words exist.
    if PDB_RE.search(text) and re.search(
        r"\b(PDB|crystal|structure|docking|model|Protein Data Bank)\b",
        text,
        re.I,
    ):
        return True

    if UNIPROT_RE.search(text) and re.search(
        r"\b(UniProt|UniProtKB|Swiss-Prot|accession)\b",
        text,
        re.I,
    ):
        return True

    return False


def extract_id_focused_windows(
    text: str,
    participant_aliases: List[str],
    window: int = 1,
    max_chars: int = 6000,
) -> str:
    """
    Extract local windows around ID/sequence/organism-related sentences.

    Keep:
    - sentences with ID-related keywords
    - sentences with participant name + structure/model/sequence/organism signals
    - one neighboring sentence before/after
    """
    sentences = split_sentences(text)
    if not sentences:
        return ""

    keep_idx = set()

    for i, s in enumerate(sentences):
        signal = has_id_or_sequence_signal(s)
        participant_hit = mentions_participant(s, participant_aliases)

        if signal:
            # Prefer ID signal directly.
            for j in range(max(0, i - window), min(len(sentences), i + window + 1)):
                keep_idx.add(j)

        elif participant_hit and re.search(
            r"\b(PDB|UniProt|accession|sequence|species|organism|"
            r"crystal|structure|docking|model|human|mouse|Homo sapiens|Mus musculus)\b",
            s,
            re.I,
        ):
            for j in range(max(0, i - window), min(len(sentences), i + window + 1)):
                keep_idx.add(j)

    if not keep_idx:
        return ""

    picked = [sentences[i] for i in sorted(keep_idx)]
    return " ".join(picked)[:max_chars].strip()


def maybe_add_id_context_block(
    blocks: List[str],
    title: str,
    body: Any,
    max_body_chars: int,
    budget: Dict[str, int],
    participant_aliases: List[str],
) -> None:
    if not isinstance(body, str):
        body = jdump(body)

    focused = extract_id_focused_windows(
        body,
        participant_aliases=participant_aliases,
        window=1,
        max_chars=max_body_chars,
    )

    if focused:
        add_block(blocks, title, focused, max_body_chars, budget)

def build_article_context(
    conn,
    doc_id: str,
    max_chars: int,
    participants: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build an ID-focused Stage 12 context.

    Stage 12 only needs:
    - PDB IDs
    - UniProt accessions
    - other explicit database IDs
    - exact printed sequences
    - species / organism evidence

    Therefore do not send the whole paper context.
    Keep only local windows around ID/sequence/organism signals.
    """
    participants = participants or []
    participant_aliases = participant_aliases_for_context(participants)

    blocks = []
    budget = {"used": 0, "max": max_chars}

    # 0. Add participant list as a compact anchor.
    add_block(
        blocks,
        "CANDIDATE_PARTICIPANTS",
        json.dumps(participant_payload(participants), ensure_ascii=False),
        20000,
        budget,
    )

    # 1. Existing participant evidence can be useful as a name anchor.
    participant_evidence_parts = []
    for p in participants:
        evs = []

        ev = clean(p.get("evidence_span"))
        if ev:
            evs.append(ev)

        for x in jload(p.get("evidence_spans_json"), []):
            if isinstance(x, str) and clean(x):
                evs.append(clean(x))
            elif isinstance(x, dict):
                e = clean(x.get("evidence_span") or x.get("text") or x.get("evidence"))
                if e:
                    evs.append(e)

        if evs:
            participant_evidence_parts.append({
                "participant_id": p.get("participant_id"),
                "name": p.get("name"),
                "entity_type": p.get("entity_type"),
                "evidence": dedupe(evs)[:5],
            })

    if participant_evidence_parts:
        maybe_add_id_context_block(
            blocks,
            "PARTICIPANT_EXISTING_EVIDENCE",
            participant_evidence_parts,
            30000,
            budget,
            participant_aliases,
        )

    # 2. Text blocks: include Title / Section-header / Text / Caption,
    # including supplementary text sections.
    text_rows = conn.execute("""
    SELECT block_id, page_no, section, text
    FROM raw_text_block
    WHERE doc_id=?
      AND section IN (
        'Title',
        'Section-header',
        'Text',
        'Caption',
        'supplementary_Title',
        'supplementary_Section-header',
        'supplementary_Text',
        'supplementary_Caption'
      )
      AND COALESCE(text, '') != ''
    ORDER BY page_no, rowid
    """, (doc_id,)).fetchall()

    page_parts: Dict[Any, List[str]] = {}
    for r in text_rows:
        page_parts.setdefault(r["page_no"], []).append(
            f"[{r['section']}] {clean(r['text'])}"
        )

    for page_no in sorted(page_parts, key=lambda x: 999999 if x is None else x):
        page_text = "\n".join(page_parts[page_no])
        maybe_add_id_context_block(
            blocks,
            f"PAGE_ID_CONTEXT page_no={page_no}",
            page_text,
            18000,
            budget,
            participant_aliases,
        )

    # 3. Figure captions: docking / model / PDB information often appears here.
    fig_rows = conn.execute("""
    SELECT figure_id, page_no, figure_ref, caption
    FROM raw_figure
    WHERE doc_id=?
      AND COALESCE(caption, '') != ''
    ORDER BY page_no, figure_ref
    """, (doc_id,)).fetchall()

    for r in fig_rows:
        caption_text = jdump(dict(r))
        maybe_add_id_context_block(
            blocks,
            f"FIGURE_ID_CONTEXT page_no={r['page_no']} figure_ref={r['figure_ref']}",
            caption_text,
            8000,
            budget,
            participant_aliases,
        )

    # 4. Tables: only keep table rows/text with ID/sequence/organism signal.
    table_rows = conn.execute("""
    SELECT table_id, page_no, table_ref, table_json
    FROM raw_table
    WHERE doc_id=?
      AND COALESCE(table_json, '') != ''
    ORDER BY page_no, table_ref
    """, (doc_id,)).fetchall()

    for r in table_rows:
        table_text = jdump(dict(r))
        maybe_add_id_context_block(
            blocks,
            f"TABLE_ID_CONTEXT page_no={r['page_no']} table_ref={r['table_ref']}",
            table_text,
            20000,
            budget,
            participant_aliases,
        )

    # 5. Text assets / supplement files:
    # Only read textual files and keep ID-focused windows.
    asset_rows = conn.execute("""
    SELECT asset_id, asset_type, file_path, metadata_json
    FROM raw_asset
    WHERE doc_id=?
      AND (
        asset_type LIKE '%supplement%'
        OR asset_type LIKE '%table%'
        OR asset_type LIKE '%text%'
      )
    ORDER BY asset_id
    """, (doc_id,)).fetchall()

    for r in asset_rows:
        preview = read_text_file(r["file_path"], max_chars=80000)

        if preview:
            maybe_add_id_context_block(
                blocks,
                f"ASSET_ID_CONTEXT asset_id={r['asset_id']} type={r['asset_type']} path={r['file_path']}",
                preview,
                30000,
                budget,
                participant_aliases,
            )
        else:
            meta_text = jdump(dict(r))
            maybe_add_id_context_block(
                blocks,
                f"ASSET_ID_METADATA asset_id={r['asset_id']} type={r['asset_type']}",
                meta_text,
                5000,
                budget,
                participant_aliases,
            )

    context = "".join(blocks).strip()

    # Safety fallback:
    # If nothing was found, return a minimal broad context rather than empty context.
    if not context:
        fallback_rows = conn.execute("""
        SELECT block_id, page_no, section, text
        FROM raw_text_block
        WHERE doc_id=?
          AND section IN ('Title', 'Section-header', 'Text', 'Caption')
          AND COALESCE(text, '') != ''
        ORDER BY page_no, rowid
        LIMIT 80
        """, (doc_id,)).fetchall()

        fallback_parts = []
        for r in fallback_rows:
            fallback_parts.append(
                f"[page_no={r['page_no']} section={r['section']}] {clean(r['text'])}"
            )

        context = "\n".join(fallback_parts)[:max_chars]

    return context


def build_prompt(participants: List[Dict[str, Any]], context: str) -> str:
    return f"""
You are Stage 12 of an induced-proximity medicine extraction pipeline.

Return JSONL only.
No markdown.
No commentary.
One JSON object per line.

Task:
For EVERY candidate participant, extract ONLY information explicitly visible in the provided evidence context:

1. PDB IDs
2. UniProt accessions
3. Other explicitly named database IDs
4. Exact sequences printed in the text
5. Species / organism names printed in the text

Do not standardize participants.
Do not complete missing IDs.
Do not use external biological knowledge.
Do not infer IDs from participant names.
Do not infer taxon IDs from species names.
Do not output HGNC, NCBI Gene, Ensembl, RNAcentral, ChEMBL, RefSeq, or GenBank IDs unless that exact database ID is visibly present in the evidence context.

Output rule:
- Output exactly one JSON object for every candidate participant.
- Use only candidate participants listed below.
- If no ID, sequence, or organism is found for a participant, output empty fields.
- Keep evidence_span short, preferably <= 300 characters.
- Do not output long evidence spans.
- Do not output long generated text.
- Do not output any ID unless the exact ID string appears in the evidence context.
- Do not output any sequence unless the exact sequence appears in the evidence context.
- Do not output taxon_id unless an explicit taxonomy ID is printed in the evidence context.

PDB rule:
- Extract PDB IDs only when they are explicitly visible near PDB / PDB ID / PDB code / Protein Data Bank / crystal structure / docking / model / structure text.
- If a PDB ID appears near a candidate participant name in the same sentence, parentheses, table row, figure caption, or evidence block, assign it to that participant.
- If one sentence contains multiple participant-PDB pairs, assign each PDB ID to the locally matched participant.
- Do not omit explicit PDB IDs just because no sequence or organism is shown.

UniProt rule:
- Extract UniProt accessions only when the exact accession is explicitly visible near UniProt / UniProtKB / Swiss-Prot / accession text.
- Do not fill UniProt accessions from memory.

Other ID rule:
- Put other database IDs only in ids_json.other.
- Each item in ids_json.other must be short and must include the visible database name and ID, for example "HGNC:12345" or "RefSeq:NM_000000".
- Do not generate long IDs.
- Do not output RNAcentral IDs unless an exact RNAcentral ID is visibly present.

Sequence rule:
- Extract protein, DNA, RNA, oligo, or modified oligo sequences only when the exact sequence is printed in the evidence context.
- Do not retrieve sequence from PDB, UniProt, gene name, protein name, or organism.
- Preserve modified oligo sequences in the paper-native representation.

Organism rule:
- Copy species / organism names only when explicitly written, such as human, Homo sapiens, mouse, Mus musculus, rat, E. coli.
- Do not convert human to 9606.
- Do not convert Homo sapiens to 9606.
- taxon_id must remain empty unless an explicit taxonomy ID is printed.

Evidence rule:
- evidence_span must be copied from the evidence context.
- evidence_span should include the participant name and extracted ID/sequence/organism whenever possible.
- If evidence is absent, leave evidence_span empty.

Allowed rt:
participant_standardization

Schema:
{{"rt":"participant_standardization","participant_id":"","name":"","organism_json":{{"organism":"","taxon_id":"","evidence_span":""}},"ids_json":{{"pdb":[],"uniprot":[],"other":[]}},"sequence_json":{{"aa_sequence":[],"dna_sequence":[],"rna_sequence":[],"oligo_sequence":[],"modified_sequence":[]}},"evidence_span":"","confidence":0.0}}

Example:

Evidence context:
"The models of ProteinA and ProteinB were obtained from the crystal structures of ProteinA (PDB code: 1ABC) and ProteinB (PDB code: 2DEF)."

Correct outputs:
{{"rt":"participant_standardization","participant_id":"participant_proteina","name":"ProteinA","organism_json":{{"organism":"","taxon_id":"","evidence_span":""}},"ids_json":{{"pdb":["1ABC"],"uniprot":[],"other":[]}},"sequence_json":{{"aa_sequence":[],"dna_sequence":[],"rna_sequence":[],"oligo_sequence":[],"modified_sequence":[]}},"evidence_span":"The models of ProteinA and ProteinB were obtained from the crystal structures of ProteinA (PDB code: 1ABC) and ProteinB (PDB code: 2DEF).","confidence":0.95}}
{{"rt":"participant_standardization","participant_id":"participant_proteinb","name":"ProteinB","organism_json":{{"organism":"","taxon_id":"","evidence_span":""}},"ids_json":{{"pdb":["2DEF"],"uniprot":[],"other":[]}},"sequence_json":{{"aa_sequence":[],"dna_sequence":[],"rna_sequence":[],"oligo_sequence":[],"modified_sequence":[]}},"evidence_span":"The models of ProteinA and ProteinB were obtained from the crystal structures of ProteinA (PDB code: 1ABC) and ProteinB (PDB code: 2DEF).","confidence":0.95}}

Bad outputs:
- Filling UniProt accession from memory.
- Filling HGNC, NCBI Gene, Ensembl, RNAcentral, or ChEMBL IDs not visibly present in the evidence context.
- Filling taxon_id=9606 only because the text says human.
- Generating long IDs.
- Generating long evidence_span.
- Outputting a sequence from UniProt or PDB instead of the article text.

Candidate participants:
{json.dumps(participant_payload(participants), ensure_ascii=False)}

Evidence context:
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


def evidence_weak_match(evidence: str, context: str) -> bool:
    ev = clean(evidence)
    if not ev:
        return False
    if ev in context:
        return True

    tokens = [t.lower() for t in re.split(r"\W+", ev) if len(t) >= 4]
    if not tokens:
        return False

    ctx = context.lower()
    hit = sum(1 for t in tokens[:20] if t in ctx)
    return hit >= min(6, len(tokens))



def value_in_context(value: Any, context: str) -> bool:
    v = clean(value)
    if not v:
        return False
    return v in context or v.upper() in context.upper()


def keep_context_values(values: Any, context: str, max_len: int = 120) -> List[str]:
    out = []
    if not isinstance(values, list):
        return out
    for x in values:
        v = clean(x)
        if not v or len(v) > max_len:
            continue
        if value_in_context(v, context):
            out.append(v)
    return dedupe(out)


def sanitize_llm_ids(raw_ids: Dict[str, Any], context: str) -> Dict[str, List[str]]:
    if not isinstance(raw_ids, dict):
        raw_ids = {}

    pdb = []
    for x in keep_context_values(raw_ids.get("pdb", []), context, max_len=4):
        x = x.upper()
        if PDB_RE.fullmatch(x):
            pdb.append(x)

    uniprot = []
    for x in keep_context_values(raw_ids.get("uniprot", []), context, max_len=12):
        x = x.upper()
        if UNIPROT_RE.fullmatch(x):
            uniprot.append(x)

    other = []
    for x in keep_context_values(raw_ids.get("other", []), context, max_len=120):
        # Keep other IDs only when the string contains an explicit database-like prefix.
        if re.search(r"\b(HGNC|NCBI|Ensembl|RNAcentral|ChEMBL|RefSeq|GenBank|accession|ID)\b", x, re.I):
            other.append(x)

    return {
        "uniprot": dedupe(uniprot),
        "pdb": dedupe(pdb),
        "hgnc": [],
        "ncbi_gene": [],
        "ensembl": [],
        "rnacentral": [],
        "chembl_target": [],
        "other": dedupe(other),
    }


def sanitize_llm_sequence_json(raw_seq: Dict[str, Any], context: str) -> Dict[str, Any]:
    if not isinstance(raw_seq, dict):
        raw_seq = {}

    aa = []
    for x in keep_context_values(raw_seq.get("aa_sequence", []), context, max_len=20000):
        v = clean(x).upper()
        if len(v) >= 20 and AA_RE.fullmatch(v):
            aa.append(v)

    dna = []
    for x in keep_context_values(raw_seq.get("dna_sequence", []), context, max_len=20000):
        v = clean(x).upper()
        if len(v) >= 8 and NA_RE.fullmatch(v):
            dna.append(v)

    rna = []
    for x in keep_context_values(raw_seq.get("rna_sequence", []), context, max_len=20000):
        v = clean(x).upper().replace("T", "U")
        if len(v) >= 8 and NA_RE.fullmatch(v):
            rna.append(v)

    oligo = keep_context_values(raw_seq.get("oligo_sequence", []), context, max_len=2000)
    modified = keep_context_values(
        (raw_seq.get("modified_oligo_sequence", []) or []) + (raw_seq.get("modified_sequence", []) or []),
        context,
        max_len=2000,
    )

    return {
        "aa_sequence": dedupe(aa),
        "dna_sequence": dedupe(dna),
        "rna_sequence": dedupe(rna),
        "oligo_sequence": dedupe(oligo),
        "modified_oligo_sequence": dedupe(modified),
        "sequence_note": clean(raw_seq.get("sequence_note"))[:500],
    }


def sanitize_llm_organism_json(raw_org: Dict[str, Any], context: str) -> Dict[str, Any]:
    if not isinstance(raw_org, dict):
        raw_org = {}

    organism = clean(
        raw_org.get("organism")
        or raw_org.get("organism_name")
        or raw_org.get("species")
        or raw_org.get("scientific_name")
    )
    taxon_id = normalize_taxon_id(
        raw_org.get("taxon_id")
        or raw_org.get("ncbi_taxon_id")
        or raw_org.get("taxonomy_id")
    )
    evidence = clean(raw_org.get("evidence_span") or raw_org.get("evidence"))

    if organism and not value_in_context(organism, context):
        organism = ""
    # A taxon ID is preserved only if the exact numeric ID is printed in the paper context.
    if taxon_id and not value_in_context(taxon_id, context):
        taxon_id = ""
    if evidence and len(evidence) > 500:
        evidence = evidence[:500]
    if evidence and not evidence_weak_match(evidence, context):
        evidence = ""

    return {
        "organism": organism,
        "taxon_id": taxon_id,
        "basis": clean(raw_org.get("basis") or raw_org.get("organism_basis"))[:120],
        "evidence_span": evidence,
        "confidence": float(raw_org.get("confidence") or 0) if str(raw_org.get("confidence") or "").replace('.', '', 1).isdigit() else 0.0,
    }


def normalize_llm_record(
    rec: Dict[str, Any],
    participant_by_id: Dict[str, Dict[str, Any]],
    participant_by_key: Dict[str, Dict[str, Any]],
    context: str,
) -> Optional[Dict[str, Any]]:
    if clean(rec.get("rt")) != "participant_standardization":
        return None

    pid = clean(rec.get("participant_id"))
    name = clean(rec.get("name"))
    participant = participant_by_id.get(pid)

    if participant is None and name:
        participant = participant_by_key.get(norm_key(name))
        if participant:
            pid = participant["participant_id"]

    if participant is None:
        return None

    # Hard filtering: keep only values that are visibly present in the supplied paper context.
    raw_ids = rec.get("ids_json") or {}
    raw_seq = rec.get("sequence_json") or {}
    raw_org = rec.get("organism_json") or {}

    ids = sanitize_llm_ids(raw_ids, context)
    seq = sanitize_llm_sequence_json(raw_seq, context)
    organism_json = sanitize_llm_organism_json(raw_org, context)

    struct = empty_structure()
    struct["pdb_ids"] = dedupe(ids.get("pdb", []))
    if struct["pdb_ids"]:
        struct["structure_source"] = "PDB"

    canonical = clean(rec.get("canonical_name")) or clean(participant.get("canonical_name")) or participant["name"]
    # Do not allow external alias normalization. If canonical spelling is not in context, fall back to participant name.
    if canonical != participant["name"] and not value_in_context(canonical, context):
        canonical = participant["name"]

    entity_type = clean(rec.get("entity_type")) or clean(participant.get("entity_type")) or "other"
    if entity_type not in ENTITY_TYPES:
        entity_type = clean(participant.get("entity_type")) or "other"

    evidence = clean(rec.get("evidence_span"))
    if evidence and len(evidence) > 500:
        evidence = evidence[:500]
    if evidence and not evidence_weak_match(evidence, context):
        evidence = ""

    warnings = []
    reasons = []

    # Mark if LLM produced values that were discarded by the hard filter.
    raw_id_count = sum(len(v) for v in raw_ids.values() if isinstance(v, list)) if isinstance(raw_ids, dict) else 0
    kept_id_count = len(ids.get("pdb", [])) + len(ids.get("uniprot", [])) + len(ids.get("other", []))
    if raw_id_count > kept_id_count:
        warnings.append("llm_ids_removed_by_context_filter")

    raw_seq_count = sum(len(v) for v in raw_seq.values() if isinstance(v, list)) if isinstance(raw_seq, dict) else 0
    kept_seq_count = sum(len(v) for k, v in seq.items() if isinstance(v, list))
    if raw_seq_count > kept_seq_count:
        warnings.append("llm_sequences_removed_by_context_filter")

    if clean(raw_org.get("taxon_id")) and not organism_json.get("taxon_id"):
        warnings.append("llm_taxon_id_removed_by_context_filter")

    if organism_json["organism"] and not organism_json["taxon_id"]:
        warnings.append("organism_without_explicit_taxon_id")

    try:
        conf = float(rec.get("confidence", 0) or 0)
    except Exception:
        conf = 0.0

    return {
        "participant_id": pid,
        "participant": participant,
        "canonical_name": canonical,
        "entity_type": entity_type,
        "ids_json": ids,
        "sequence_json": seq,
        "structure_json": struct,
        "standardization_json": {
            "source": "paper_or_supplement_llm",
            "evidence_source": clean(rec.get("evidence_source")) or "paper_or_supplement_llm",
            "organism_json": organism_json,
            "external_lookup_used": False,
            "hard_filter": {
                "ids_must_appear_in_context": True,
                "sequences_must_appear_in_context": True,
                "taxon_id_must_appear_in_context": True,
            },
            "llm_record": rec,
        },
        "evidence_span": evidence,
        "confidence": max(conf, 0.80),
        "status": "review_required" if reasons else ("auto_pass_with_warning" if warnings else "auto_pass"),
        "review_required": bool(reasons),
        "qc_reasons": reasons,
        "qc_warnings": warnings,
        "raw_output": rec,
    }

def http_json(url: str, timeout: int = 20) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def extract_taxon_from_polymer_entity(data: Dict[str, Any]) -> Dict[str, Any]:
    candidates = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            tax = x.get("ncbi_taxonomy_id") or x.get("taxonomy_id")
            sci = x.get("scientific_name") or x.get("organism_scientific_name")
            if tax or sci:
                candidates.append({"taxon_id": clean(tax), "organism": clean(sci)})
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for i in x:
                walk(i)

    walk(data)

    for c in candidates:
        if c.get("taxon_id") or c.get("organism"):
            return c

    return {}


def extract_uniprot_from_polymer_entity(data: Dict[str, Any]) -> List[str]:
    out = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            db = clean(x.get("database_name") or x.get("db_name") or x.get("resource_name")).lower()
            acc = clean(x.get("database_accession") or x.get("accession") or x.get("identifier"))
            if acc and ("uniprot" in db or UNIPROT_RE.fullmatch(acc.upper())):
                out.append(acc.upper())
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for i in x:
                walk(i)

    walk(data)
    return dedupe(out)



def participant_alias_keys(name: str) -> List[str]:
    base = norm_key(name)
    aliases = {base}

    if base in {"crbn", "cereblon"}:
        aliases.update({"crbn", "cereblon"})

    if base in {"cmet", "met", "hgfr", "hepatocytegrowthfactorreceptor"}:
        aliases.update({
            "cmet",
            "met",
            "hgfr",
            "hepatocytegrowthfactorreceptor",
            "hepatocytegrowthfactorreceptorprotein",
        })

    return [x for x in aliases if x]


def clean_pdb_sequence(seq: Any) -> str:
    seq = clean(seq)
    seq = re.sub(r"[^A-Za-z]", "", seq).upper()
    return seq


def query_pdb_sequences(pdb_id: str, participant_name: str, taxon_id: str = "") -> Dict[str, Any]:
    pdb_id = clean(pdb_id).upper()
    if not PDB_RE.fullmatch(pdb_id):
        return {"source": "rcsb_pdb", "pdb_id": pdb_id, "error": "invalid_pdb_id"}

    entry = http_json(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
    if not entry:
        return {"source": "rcsb_pdb", "pdb_id": pdb_id, "error": "entry_not_found"}

    entity_ids = (
        entry.get("rcsb_entry_container_identifiers", {})
        .get("polymer_entity_ids", [])
    )

    polymer_records = []
    aliases = participant_alias_keys(participant_name)

    for entity_id in entity_ids:
        data = http_json(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}")
        if not data:
            continue

        entity_poly = data.get("entity_poly") or {}
        sequence = clean_pdb_sequence(
            entity_poly.get("pdbx_seq_one_letter_code_can")
            or entity_poly.get("pdbx_seq_one_letter_code")
        )

        desc = clean(
            (data.get("rcsb_polymer_entity") or {}).get("pdbx_description")
            or (data.get("entity") or {}).get("pdbx_description")
            or (data.get("struct_asym") or {}).get("details")
            or ""
        )

        polymer_type = clean(entity_poly.get("type"))
        organism = extract_taxon_from_polymer_entity(data)
        uniprots = extract_uniprot_from_polymer_entity(data)

        desc_key = norm_key(desc)

        score = 0
        if any(a and a in desc_key for a in aliases):
            score += 10

        if taxon_id and clean(organism.get("taxon_id")) == clean(taxon_id):
            score += 3

        if "polypeptide" in polymer_type.lower():
            score += 2

        if sequence and len(sequence) >= 20 and PDB_AA_RE.fullmatch(sequence):
            score += 1

        polymer_records.append({
            "pdb_id": pdb_id,
            "entity_id": str(entity_id),
            "description": desc,
            "polymer_type": polymer_type,
            "sequence": sequence,
            "organism": organism,
            "uniprot": uniprots,
            "score": score,
        })

    if not polymer_records:
        return {"source": "rcsb_pdb", "pdb_id": pdb_id, "error": "no_polymer_entities"}

    valid_records = [
        r for r in polymer_records
        if r.get("sequence")
        and len(r["sequence"]) >= 20
        and PDB_AA_RE.fullmatch(r["sequence"])
    ]

    if not valid_records:
        return {
            "source": "rcsb_pdb",
            "pdb_id": pdb_id,
            "error": "no_valid_polymer_sequence",
            "polymer_records": polymer_records,
        }

    valid_records.sort(key=lambda x: x["score"], reverse=True)
    chosen = valid_records[0]

    # 如果多个 entity 分数相同，保留 warning，但仍然返回最高候选。
    ambiguous = (
        len(valid_records) > 1
        and valid_records[0]["score"] == valid_records[1]["score"]
    )

    return {
        "source": "rcsb_pdb",
        "pdb_id": pdb_id,
        "chosen": chosen,
        "ambiguous": ambiguous,
        "polymer_records": polymer_records,
    }

def query_uniprot(name: str, taxon_id: str, timeout: int = 20) -> Dict[str, Any]:
    name = clean(name)
    taxon_id = clean(taxon_id)

    if not name or not taxon_id:
        return {"source": "uniprot_rest", "error": "missing_name_or_taxon"}

    query = f'(gene_exact:{name} OR protein_name:"{name}") AND organism_id:{taxon_id} AND reviewed:true'

    try:
        r = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={
                "query": query,
                "format": "json",
                "size": 1,
                "fields": "accession,id,gene_names,protein_name,organism_name,sequence",
            },
            timeout=timeout,
        )

        if r.status_code != 200:
            return {
                "source": "uniprot_rest",
                "query": query,
                "error": f"HTTP {r.status_code}",
            }

        results = r.json().get("results") or []
        if not results:
            return {
                "source": "uniprot_rest",
                "query": query,
                "error": "not_found",
            }

        x = results[0]
        genes = []
        for g in x.get("genes", []) or []:
            v = (g.get("geneName") or {}).get("value")
            if v:
                genes.append(v)

        protein_name = (
            x.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", "")
        )

        return {
            "source": "uniprot_rest",
            "query": query,
            "accession": x.get("primaryAccession", ""),
            "entry_id": x.get("uniProtkbId", ""),
            "gene_names": genes,
            "protein_name": protein_name,
            "organism": (x.get("organism") or {}).get("scientificName", ""),
            "sequence": (x.get("sequence") or {}).get("value", ""),
        }

    except Exception as e:
        return {
            "source": "uniprot_rest",
            "query": query,
            "error": str(e),
        }


def merge_sequence_fallback(rec: Dict[str, Any], default_taxon_id: str) -> Dict[str, Any]:
    if has_sequence_info(rec):
        return rec

    rec.setdefault("qc_warnings", [])

    ids = rec.get("ids_json") or empty_ids()
    seq = rec.get("sequence_json") or empty_seq()
    struct = rec.get("structure_json") or empty_structure()
    std = dict(rec.get("standardization_json") or {})

    name = clean(
        (rec.get("participant") or {}).get("name")
        or rec.get("canonical_name")
        or rec.get("name")
    )

    taxon_id = get_taxon_id(rec, default_taxon_id)
    org = normalize_organism_json(std.get("organism_json") or {})

    if not org.get("taxon_id") and default_taxon_id:
        if "missing_taxon_for_sequence_fallback" not in rec["qc_warnings"]:
            rec["qc_warnings"].append("missing_taxon_for_sequence_fallback")
        std["organism_json"] = {
            **org,
            "taxon_id": default_taxon_id,
            "basis": org.get("basis") or "default_human_fallback",
        }
        taxon_id = default_taxon_id

    pdb_ids = dedupe(ids.get("pdb", []) + struct.get("pdb_ids", []))

    # 关键修正：只要有 PDB ID，就直接查 PDB sequence。
    # 不再要求 entity_type 必须是 protein / peptide / antibody。
    if pdb_ids:
        pdb_results = []

        for pdb_id in pdb_ids:
            pdb_result = query_pdb_sequences(
                pdb_id,
                participant_name=name,
                taxon_id=taxon_id,
            )
            pdb_results.append(pdb_result)

            chosen = pdb_result.get("chosen") or {}
            sequence = clean_pdb_sequence(chosen.get("sequence"))

            if sequence and len(sequence) >= 20 and PDB_AA_RE.fullmatch(sequence):
                seq["aa_sequence"] = dedupe((seq.get("aa_sequence") or []) + [sequence])

            for acc in chosen.get("uniprot") or []:
                acc = clean(acc).upper()
                if UNIPROT_RE.fullmatch(acc):
                    ids["uniprot"] = dedupe((ids.get("uniprot") or []) + [acc])

            chosen_org = chosen.get("organism") or {}
            if chosen_org and not (std.get("organism_json") or {}).get("organism"):
                std["organism_json"] = normalize_organism_json({
                    "organism": chosen_org.get("organism"),
                    "taxon_id": chosen_org.get("taxon_id"),
                    "basis": "pdb_polymer_entity_source",
                    "confidence": 0.80,
                })

            if pdb_result.get("ambiguous"):
                if "pdb_polymer_entity_ambiguous" not in rec["qc_warnings"]:
                    rec["qc_warnings"].append("pdb_polymer_entity_ambiguous")

        std["pdb_sequence_fallback"] = {
            "source": "rcsb_pdb",
            "results": pdb_results,
        }

        rec["ids_json"] = ids
        rec["sequence_json"] = seq
        rec["standardization_json"] = std

        if has_sequence_info(rec):
            if clean(rec.get("entity_type")) in {"", "other"}:
                rec["entity_type"] = "protein"

            rec["confidence"] = max(float(rec.get("confidence") or 0), 0.85)

            if rec.get("status") == "auto_pass":
                rec["status"] = "auto_pass_with_external_fallback"

            return rec

        if "pdb_sequence_fallback_failed" not in rec["qc_warnings"]:
            rec["qc_warnings"].append("pdb_sequence_fallback_failed")

    # 没有 PDB 或 PDB 没拿到 sequence 时，才走 UniProt。
    entity_type = clean(rec.get("entity_type"))
    if entity_type not in {"protein", "peptide", "antibody"}:
        return rec

    if not has_sequence_info(rec):
        uniprot = query_uniprot(name, taxon_id=taxon_id)

        std["uniprot_sequence_fallback"] = {
            "source": "uniprot",
            "taxon_id": taxon_id,
            "query_name": name,
            "result": uniprot,
        }

        if uniprot.get("accession"):
            ids["uniprot"] = dedupe((ids.get("uniprot") or []) + [uniprot["accession"]])

            if uniprot.get("sequence"):
                seq["aa_sequence"] = dedupe((seq.get("aa_sequence") or []) + [uniprot["sequence"]])

            if not clean(rec.get("canonical_name")) or rec.get("canonical_name") == name:
                if uniprot.get("gene_names"):
                    rec["canonical_name"] = uniprot["gene_names"][0]
                elif uniprot.get("protein_name"):
                    rec["canonical_name"] = uniprot["protein_name"]

            rec["ids_json"] = ids
            rec["sequence_json"] = seq
            rec["standardization_json"] = std
            rec["confidence"] = max(float(rec.get("confidence") or 0), 0.85)

            if rec.get("status") == "auto_pass":
                rec["status"] = "auto_pass_with_external_fallback"

        else:
            rec["standardization_json"] = std
            if "uniprot_sequence_fallback_failed" not in rec["qc_warnings"]:
                rec["qc_warnings"].append("uniprot_sequence_fallback_failed")
            if rec.get("status") == "auto_pass":
                rec["status"] = "auto_pass_with_warning"

    return rec

def make_empty_record(participant: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "participant_id": participant["participant_id"],
        "participant": participant,
        "canonical_name": clean(participant.get("canonical_name")) or participant["name"],
        "entity_type": clean(participant.get("entity_type")) or "other",
        "ids_json": empty_ids(),
        "sequence_json": empty_seq(),
        "structure_json": empty_structure(),
        "standardization_json": {
            "source": "no_llm_record",
            "organism_json": {},
            "external_lookup_used": False,
        },
        "evidence_span": clean(participant.get("evidence_span")),
        "confidence": float(participant.get("confidence") or 0.65),
        "status": "auto_pass_with_warning",
        "review_required": False,
        "qc_reasons": [],
        "qc_warnings": ["missing_llm_participant_record"],
        "raw_output": {},
    }


def update_participant(conn, rec: Dict[str, Any]) -> None:
    # 为了兼容新数据库，将拆分的 reasons 和 warnings 打包成一个 JSON
    combined_qc = {
        "reasons": rec.get("qc_reasons", []),
        "warnings": rec.get("qc_warnings", [])
    }
    
    conn.execute("""
    UPDATE stg_relation_participant
    SET
        canonical_name=?,
        entity_type=?,
        ids_json=?,
        sequence_json=?,
        structure_json=?,
        standardization_json=?,
        evidence_text=CASE
            WHEN COALESCE(?, '') != '' THEN ?
            ELSE evidence_text
        END,
        confidence=?,
        status=?,
        review_required=?,
        qc_reasons_json=?,
        raw_output=?,
        updated_at=CURRENT_TIMESTAMP
    WHERE participant_id=?
    """, (
        rec["canonical_name"],
        rec["entity_type"],
        jdump(rec["ids_json"]),
        jdump(rec["sequence_json"]),
        jdump(rec["structure_json"]),
        jdump(rec["standardization_json"]),
        rec["evidence_span"],  # Python 字典里仍然叫 evidence_span，映射给 SQL 第一个 ?
        rec["evidence_span"],  # 映射给 SQL 第二个 ?
        rec["confidence"],
        rec["status"],
        int(rec.get("review_required", 0)),
        jdump(combined_qc),    # 写入新的 qc_reasons_json 列
        jdump(rec.get("raw_output", {})),
        rec["participant_id"],
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "ipm-vlm"))
    ap.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    ap.add_argument("--max-context-chars", type=int, default=640000)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--default-taxon-id", default="9606")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = get_conn()
    preflight(conn)

    participants = load_participants(conn, args.doc_id)
    participant_by_id = {p["participant_id"]: p for p in participants}
    participant_by_key = {norm_key(p["name"]): p for p in participants}

    context = build_article_context(
        conn,
        args.doc_id,
        args.max_context_chars,
        participants=participants,
)
    prompt = build_prompt(participants, context)

    client = OpenAI(base_url=args.llm_base_url, api_key=args.llm_api_key)
    raw_text = call_llm(client, args.llm_model, prompt, args.max_tokens)

    out_dir = Path("data/staging") / args.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    context_path = out_dir / "stage12_participant_context.txt"
    raw_path = out_dir / "stage12_participant_llm_raw.jsonl"
    valid_path = out_dir / "stage12_relation_participants_updated.jsonl"
    report_path = out_dir / "stage12_relation_participants_report.json"

    context_path.write_text(context, encoding="utf-8")
    raw_path.write_text(jdump({
        "doc_id": args.doc_id,
        "num_participants": len(participants),
        "context_chars": len(context),
        "context_path": str(context_path),
        "raw_text": raw_text,
    }) + "\n", encoding="utf-8")

    stats = Counter()
    llm_records: Dict[str, Dict[str, Any]] = {}

    for rec in parse_jsonl(raw_text):
        nr = normalize_llm_record(rec, participant_by_id, participant_by_key, context)
        if not nr:
            stats["invalid_llm_record"] += 1
            continue

        llm_records[nr["participant_id"]] = nr
        stats["llm_record"] += 1

    final_records = []

    for p in tqdm(participants, desc="Stage 12 update participants"):
        rec = llm_records.get(p["participant_id"]) or make_empty_record(p)

        before_seq = has_sequence_info(rec)
        rec = merge_sequence_fallback(rec, default_taxon_id=args.default_taxon_id)
        after_seq = has_sequence_info(rec)

        if not before_seq and after_seq:
            stats["sequence_fallback_success"] += 1
        elif not after_seq:
            stats["sequence_missing_after_fallback"] += 1

        final_records.append(rec)

        stats["participant"] += 1
        stats[f"status:{rec['status']}"] += 1
        stats[f"entity_type:{rec['entity_type']}"] += 1
        stats[f"source:{rec['standardization_json'].get('source', '')}"] += 1

        if rec["ids_json"].get("pdb"):
            stats["has_pdb"] += 1
        if rec["ids_json"].get("uniprot"):
            stats["has_uniprot"] += 1
        if rec["sequence_json"].get("aa_sequence"):
            stats["has_aa_sequence"] += 1
        if rec["sequence_json"].get("rna_sequence") or rec["sequence_json"].get("oligo_sequence"):
            stats["has_na_or_oligo_sequence"] += 1

        if not args.dry_run:
            update_participant(conn, rec)

    if not args.dry_run:
        conn.commit()

    conn.close()

    with open(valid_path, "w", encoding="utf-8") as f:
        for r in final_records:
            f.write(jdump(r) + "\n")

    report = {
        "doc_id": args.doc_id,
        "mode": "stage12_llm_organism_then_pdb_or_uniprot_sequence_fallback",
        "num_participants": len(participants),
        "context_chars": len(context),
        "context_path": str(context_path),
        "default_taxon_id": args.default_taxon_id,
        "stats": dict(stats),
        "raw_llm_jsonl": str(raw_path),
        "validated_jsonl": str(valid_path),
        "tables": ["stg_relation_participant"],
        "note": (
            "Stage 12 updates stg_relation_participant in place. "
            "LLM first extracts organism_json, taxon_id, evidence, PDB/UniProt/sequence/construct/domain/mutation. "
            "If sequence is missing, PDB sequence fallback is tried first; otherwise UniProt fallback uses LLM taxon_id, "
            "or default human taxon with missing_taxon_for_sequence_fallback warning."
        ),
    }

    report_path.write_text(jdump(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
