#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract IPM design/mechanism relations from parsed/planned article evidence.

This version intentionally keeps the user's  prompt almost unchanged.
Minimal changes vs the uploaded prompt:
1) supplementary text/tables are allowed as evidence;
2) a new final compound/construct name may be output when it is visible in context.

Writes:
- stg_agent
- stg_relation
- stg_relation_participant

Does NOT extract assay measurements; use extract_assays.py separately.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn


RELATION_TYPES = {
    "targeted_degradation", "targeted_stabilization", "state_modulation",
    "sequence_editing", "interaction_spatial_rewiring", "cell_cell_proximity",
    "induced_proximity", "other",
}
MODALITIES = {
    "PROTAC", "Molecular_Glue_Degrader", "Hydrophobic_Tagging_Degrader", "HyT",
    "SNIPER", "LYTAC", "AbTAC", "GlueTAC", "AUTAC", "ATTEC",
    "RiboTAC_RNaseL", "RNaseH1_Gapmer_ASO", "DUBTAC", "TF-DUBTAC",
    "PHICS", "PhosTAC_or_PhosTAP", "AceTAG", "OGT_TAC", "OGA_TAC",
    "SUMOTAC_or_Other_PTM_TAC", "LEAPER", "RESTORE", "REPAIR", "RESCUE",
    "CIRTS_ADAR", "RIPTAC", "Nondegradative_Glue_or_PPI_Stabilizer",
    "PPI_Inducer", "BiTE", "TriTE_or_Multispecific_TCE", "NK_Engager_or_NKCE",
    "Other",
}
MECHANISM_ROUTES = {
    "E3_ligase_recruitment", "hydrophobic_tagging", "protein_quality_control_unspecified",
    "Endocytosis_Lysosome_eTPD", "autophagy", "Autophagy", "DUB", "Kinase",
    "Phosphatase", "PTM_enzyme_recruitment", "PTM_Enzyme", "ADAR",
    "nuclease_RNA_decay", "Nuclease_RNA_Decay", "partner_protein_or_complex_induction",
    "Partner_Protein_or_Complex", "T_Cell_CD3", "NK_Cell", "Other_Immune_Bridging",
    "unknown", "Unknown", "other", "Other",
}
OUTCOME_CLASSES = {
    "I_Protein_Abundance_Decrease",
    "II_Protein_Abundance_Increase",
    "III_State_Modulation",
    "IV_Sequence_Information_Editing",
    "V_Interaction_Spatial_Rewiring",
    "VI_Cell_Cell_Proximity",
    "Other",
}
MECHANISM_TAGS = {
    "E3_UPS", "Hydrophobic_Tagging", "Proteasome", "Lysosome", "Autophagy",
    "DUB", "Kinase", "Phosphatase", "PTM", "ADAR", "RNA_Decay",
    "PPI_Induction", "PPI_Stabilization", "Cell_Cell_Bridging", "Other",
}
PARTICIPANT_ROLES = {
    "degradation_target", "stabilization_target", "regulated_target",
    "recruited_effector", "nuclease_effector", "editing_effector",
    "PTM_writer", "PTM_eraser", "trafficking_receptor",
    "autophagy_receptor_or_adapter", "proximal_partner",
    "immune_target_antigen", "immune_effector_marker", "target", "effector",
    "partner", "cell", "other",
}
FUNCTIONAL_ROLES = {"target", "effector", "partner", "cell", "other"}
ENTITY_TYPES = {"compound", "protein", "RNA", "DNA", "oligo", "antibody", "peptide", "cell", "construct", "other"}
RELATION_BASIS = {"design_class", "proposed_model", "mechanistic_validation"}

GENERIC_BAD_NAMES = {
    "compound", "compounds", "molecule", "molecules", "protac", "protacs", "degrader", "degraders",
    "hyt molecules", "series", "analogs", "analogues", "linker", "warhead", "tag", "ligand",
    "e3 ligand", "target ligand", "crbn ligand", "vhl ligand", "hydrophobic tag",
    "intermediate", "intermediates", "reagent", "building block", "e3 ligase",
    "protein", "target", "effector", "proteasome", "lysosome", "hyt",
}
NEGATIVE_TEXT = re.compile(r"\b(NMR|HRMS|yield|synthesis|synthesized|preparation|purification|LC-MS|m/z|ppm)\b", re.I)
RANGE_RE = re.compile(r"\b\d+[a-z]?\s*[-–−]\s*\d+[a-z]?\b", re.I)
LIST_SEP_RE = re.compile(r"\b\d+[a-z]?\s*(?:,|/|;|\band\b|\bor\b)\s*\d+[a-z]?\b", re.I)
DOSE_PREFIX_RE = re.compile(r"^\s*(?:at\s+)?(?:\d+(?:\.\d+)?\s*)?(?:nM|μM|uM|mM|mg/kg|µM)\s+", re.I)
TREATMENT_SUFFIX_RE = re.compile(r"(?:-treated|\s+treated\s+cells?|\s+treatment)$", re.I)
COMPOUND_PREFIX_RE = re.compile(r"^(?:compound|cmpd|molecule|agent)\s+", re.I)

PROMPT_TEMPLATE = 'You are extracting induced-proximity core relations from ONE article-level ordered text_evidence context.\n\nReturn JSONL only.\nNo markdown. No commentary.\nOne JSON object per line.\n\nAllowed rt:\nrelation\n\nTASK\nExtract mechanism-level or design-level induced-proximity medicine (IPM) relations.\n\nRelation extraction is intentionally permissive but evidence-grounded.\nA relation is NOT a claim that the inducer is potent, effective, positive, or therapeutically useful.\nA relation only means that a final inducer agent was designed, proposed, modeled, or experimentally tested as an induced-proximity agent for named biological participant(s).\n\nActivity, potency, polarity, degradation percentage, DC50, Dmax, IC50, cell viability, dose response, time-course, WB quantification, and whether the relation is positive, negative, weak, moderate, or inactive belong to assay extraction.\n\nSUPPLEMENTARY EVIDENCE RULE\nARTICLE TEXT_EVIDENCE CONTEXT may include main-text evidence and supplementary text, tables, figure captions, DOCX/TXT/CSV/XLSX-derived tables, and supplementary-PDF OCR blocks. Supplementary evidence is valid for this stage if it supports a design-level or mechanism-level IPM relation. However, pure supplementary structure, SMILES, synthesis, NMR, HRMS, yield, or characterization evidence alone must not trigger a  relation.\n\nCORE DEFINITION\nA relation means:\none final inducer agent is designed, proposed, modeled, or experimentally shown to organize one or more biological participants through an induced-proximity mechanism.\n\nOutput one relation when the article-level context supports that:\n- a final degrader, glue, engager, construct, oligo, antibody, biologic, or cell-bridging agent is designed to act on a named biological target;\n- a final compound series is explicitly described as PROTACs, HyT molecules, molecular glues, LYTACs, DUBTACs, RiboTACs, BiTEs, NK engagers, PPI inducers, or another IPM modality;\n- a final inducer is proposed to recruit, engage, stabilize, degrade, edit, traffic, bridge, or bring into proximity a named biological participant;\n- a final inducer is shown, modeled, or proposed to form a ternary complex, quaternary complex, cell-cell bridge, or other induced-proximity complex;\n- a final inducer has mechanism evidence such as target engagement, effector recruitment, ternary complex formation, proteasome/lysosome/autophagy/chaperone involvement, RNA nuclease/editing recruitment, PTM enzyme recruitment, PPI stabilization, or cell-cell engagement.\n\nWEAK / INACTIVE RELATION RULE\nDo NOT filter out weak, inactive, N.D., low-degradation, moderate, poor, failed, or negative compounds if the context supports that they are final IPM agents by design, class, figure/table title, SAR series, or mechanism description.\n\nSpecifically:\n- Weak, inactive, N.D., low degradation, poor degradation, moderate degradation, no obvious degradation, or negative assay results MUST NOT invalidate a design-level relation.\n- If the compound is a final IPM agent by design/class/table/figure/SAR context, extract the relation even when the assay result is weak or negative.\n- Do not encode weak/negative activity except indirectly through lower confidence or review_required when appropriate.\n- The weak/negative outcome itself must be captured later in assay records.\n\nSERIES EXPANSION RULE\nIf a table, figure, caption, SAR section, or design section defines a final compound series as IPM agents targeting a named biological target, output one relation per known final compound in that series.\n\nExamples:\n- "HyT compounds 14a−14f and 17a−17f targeting c-Met" -> output one relation per known final compound in that range.\n- "PROTACs 22a−22g, 24, 26, and 28a−28c targeting c-Met" -> output one relation per known final compound in that range.\n\nDo not collapse a compound series into one generic relation.\nIf Known compound names contains individual compounds from a range, output each individual final compound as a separate relation.\n\nDO NOT OUTPUT A RELATION IF THE CONTEXT ONLY CONTAINS\n- synthesis route\n- reaction conditions\n- intermediate preparation\n- reagent list\n- yield\n- 1H NMR\n- 13C NMR\n- HRMS\n- m/z\n- ppm\n- compound characterization only\n- reference inhibitor only\n- target ligand only\n- E3 ligand only\n- linker only\n- hydrophobic tag only\n- warhead only\n- building block only\n\nA chemical name plus NMR/HRMS/yield is not relation evidence.\n\nFINAL INDUCER RULE\ninducer_name must be the complete final molecule, biologic, oligo, antibody, construct, or cell-bridging agent that mediates the IPM relation.\n\nDo not use any of the following as inducer_name:\n- intermediate\n- warhead\n- linker\n- target ligand\n- E3 ligand\n- CRBN ligand\n- VHL ligand\n- hydrophobic tag\n- reagent\n- reference inhibitor\n- building block\n\nExamples:\n- "compound S1 was selected as the most suitable E3 ligand" -> no relation for S1.\n- "tepotinib was selected as the c-Met ligand" -> no relation for tepotinib.\n- "PROTACs 22a-22g targeting c-Met" -> valid design-level relations for 22a-22g if those names are in Known compound names.\n- "compound 22b recruits CRBN to form a ternary complex with c-Met" -> valid mechanism-supported relation.\n\nINDUCER NAME NORMALIZATION\n- inducer_name must be exactly one name from Known compound names whenever possible.\n- Normalize treatment phrases to the known compound name.\n\nExamples:\n- "compound 22b-treated" -> "22b"\n- "compound 22b" -> "22b"\n- "22b-treated cells" -> "22b"\n- "100 nM 22b" -> "22b"\n\nIf no exact known compound name can be identified, you may output a NEW inducer_name only when all conditions are met:\n- it is one specific final IPM compound/construct name visible verbatim in ARTICLE TEXT_EVIDENCE CONTEXT;\n- it is not a generic class name, series label, range, list, fragment, linker, ligand, reagent, intermediate, or building block;\n- it is supported by design-class, proposed-model, or mechanistic relation evidence.\nFor such new names, use the exact visible final compound/construct name. These new names will be inserted with missing structure for later structure-resolution tasks.\nIf none of the above is satisfied, return no line.\n\nRELATION MULTIPLICITY\nOutput one relation per unique:\nfinal inducer + mechanism_route + participant set\n\nImportant multiplicity rule:\n- The same inducer may support multiple valid relations in one paper when the biological participant set differs.\n- Do not assume a molecule has only one target or one effector across the whole article.\n- If the article reports the same inducer against Brd2, Brd3, and Brd4 in different experiments or mechanism contexts, keep those as separate relations whenever the local evidence identifies different participant sets.\n- Article-level molecule hints are non-exclusive context only; local evidence wins.\n\nDo NOT output one relation per:\n- assay condition\n- dose\n- time point\n- cell line\n- readout\n- degradation percentage\n- DC50\n- Dmax\n- IC50\n- N.D.\n\nFor SAR/design tables, output one relation per final compound only when each row or series member corresponds to a final IPM agent and the table/title/context clearly defines the modality and target.\n\nRELATION BASIS\nUse relation_basis to describe why this relation is extracted.\n\nAllowed relation_basis:\n["design_class", "proposed_model", "mechanistic_validation"]\n\nDefinitions:\n- design_class:\n  The compound is a final IPM agent by design/class/series/table/figure title, but no detailed mechanism validation is shown for this specific inducer in the local context.\n\n- proposed_model:\n  The context provides docking, modeling, schematic, structural model, predicted ternary/proximity mode, or proposed induced-proximity mechanism.\n\n- mechanistic_validation:\n  The context provides experimental mechanism evidence, such as competitor rescue, E3 dependency, CRBN/VHL dependency, MG132/lysosome/autophagy inhibitor rescue, target engagement, ubiquitination, ternary complex assay, pull-down, CETSA, NanoBRET, SPR/ITC/MST ternary evidence, or other mechanism tests.\n\nMECHANISM ROUTE DECISION RULES\n- Use mechanism_route="E3_ligase_recruitment" for PROTACs and E3-recruiting molecular glue mechanisms.\n- Use mechanism_route="hydrophobic_tagging" for HyT / hydrophobic-tagging degraders.\n- Use mechanism_route="partner_protein_or_complex_induction" for PPI induction or PPI stabilization without degradation.\n- Use the closest allowed mechanism_route from the enum.\n- If unclear but still an IPM relation, use "unknown" or "other" only if those values are allowed.\n\nNamed E3 participant rule:\nAdd a named E3 participant such as CRBN, VHL, DCAF15, DCAF16, RNF114, IAP, MDM2, KEAP1, or β-TRCP only if at least one condition is met:\n\n1. Specific-inducer evidence:\n   The E3 is explicitly linked to the specific inducer by mechanism evidence.\n   In this case, use relation_basis="mechanistic_validation" or "proposed_model".\n\n2. Series-level design propagation:\n   The article-level context explicitly states that a named E3 ligand/recruiter was selected for the designed PROTAC/E3-recruiting degrader series, and a figure/table/title/context defines a set of final compounds as PROTACs or E3-recruiting degraders targeting a named target.\n   In this case, propagate the named E3 to each final PROTAC in that series.\n\nSeries-level E3 propagation is allowed only when:\n- the E3 recruiter is explicitly named, such as CRBN ligand, VHL ligand, IAP ligand, MDM2 ligand, KEAP1 ligand, etc.;\n- the final compounds are explicitly described as PROTACs or E3-recruiting degraders;\n- the final compounds are present in Known compound names;\n- there is no conflicting E3 recruiter assignment for the same compound series.\n\nFor propagated E3 relations:\n- include the E3 participant with participant_role="recruited_effector" and functional_role="effector";\n- set relation_basis="design_class";\n- set confidence between 0.70 and 0.80;\n- set review_required=true unless the exact compound-to-E3-recruiter mapping is directly stated or structurally unambiguous in the local context;\n- the participant evidence_span for the E3 should cite the sentence that names the selected E3 recruiter;\n- the participant evidence_span for the target should cite the sentence/table/figure title that defines the target or compound series.\n\nIf the context says PROTAC but does not name the E3 anywhere in the article-level context, do not invent CRBN/VHL. Keep only the target participant.\n\nIf the context only says proteasome-mediated or MG132 blocks degradation but no named E3 is stated, use mechanism_route="protein_quality_control_unspecified" unless the broader article-level context explicitly names the E3 recruiter for that inducer or series.\n\nHyT rule:\n- HyT means Hydrophobic Tagging.\n- HyT is a modality/mechanism, not a participant or effector.\n- For HyT compounds:\n  modality="Hydrophobic_Tagging_Degrader"\n  mechanism_route="hydrophobic_tagging"\n  mechanism_tags="Hydrophobic_Tagging"\n  participants usually include only the named degradation target.\n- Do not use "HyT" as participant name.\n- Do not invent CRBN/VHL/E3 for HyT compounds unless the article explicitly states a named effector.\n- Do not discard HyT relations only because the assay activity is weak, N.D., low, or negative.\n\nPARTICIPANT RULES\nparticipants must be concrete biological entities.\n\nDo not use modality names, compound classes, assay terms, chemical fragments, or pathway terms as participant names.\n\nForbidden participant names include:\nPROTAC, degrader, compound, molecule, E3 ligase, proteasome, lysosome, degradation rate, DC50, Dmax, HyT, linker, warhead, tag, ligand.\n\nRole assignment:\n- For degraded target:\n  participant_role="degradation_target"\n  functional_role="target"\n\n- For stabilized/increased target:\n  participant_role="stabilization_target"\n  functional_role="target"\n\n- For generally regulated target:\n  participant_role="regulated_target"\n  functional_role="target"\n\n- For recruited E3 / enzyme / nuclease / editor / receptor:\n  use the most specific participant_role, such as recruited_effector, nuclease_effector, editing_effector, PTM_writer, PTM_eraser, trafficking_receptor, autophagy_receptor_or_adapter.\n  functional_role="effector"\n\n- For a protein only brought near another protein:\n  participant_role="proximal_partner"\n  functional_role="partner"\n\n- For BiTE/TCE/NK engager:\n  tumor antigen should be immune_target_antigen.\n  CD3/CD16A/NKp46 or immune-side receptor should be immune_effector_marker.\n\nEVIDENCE RULES\n- evidence_span must be copied from the article-level text_evidence context.\n- evidence_span should support the design class, proposed model, or mechanism relation.\n- Good evidence spans include table titles, figure captions, design sentences, mechanism sentences, docking/modeling descriptions, proposed ternary-complex descriptions, or mechanistic validation sentences.\n- Avoid using pure numeric readout spans as relation evidence.\n- A sentence mentioning weak, moderate, poor, inactive, N.D., or low degradation may be used as relation evidence only if the same sentence or local context also identifies the final compound/series as an IPM modality targeting a named biological target.\n- Avoid synthesis/NMR/HRMS/yield spans.\n- Do not concatenate unrelated distant sentences to create artificial support.\n- For series-level E3 propagation, evidence_span may combine one short E3-selection span and one short series/target-defining span using " ... " only when both spans support the same propagated relation.\n\nGood weak-relation evidence examples:\n- "We designed and synthesized several HyT molecules with different types of HyT and PROTACs with different types of linkers."\n- "HyT molecules showed weak degradation activity."\n- "Structures and c-Met Degradation Activity of HyT Compounds 14a−14f and 17a−17f"\n- "Structures and c-Met Degradation Activity of PROTACs 22a−22g, 24, 26, and 28a−28c"\n\nBad weak-relation evidence examples:\n- "DC50 > 1 μM" alone\n- "N.D." alone\n- "Dmax = 12%" alone\n- a row containing only compound name, yield, NMR, HRMS, or m/z\n\nCONFIDENCE AND REVIEW RULES\nUse confidence as extraction confidence for the relation, not assay strength.\n\nSuggested confidence:\n- 0.90-0.98:\n  specific inducer has direct experimental mechanism validation with named target and named effector.\n\n- 0.80-0.89:\n  specific inducer has proposed/modeling/ternary-complex evidence with named target and named effector.\n\n- 0.70-0.80:\n  design-class relation from clear final IPM compound series/table/figure/title/SAR context.\n\n- 0.60-0.70:\n  relation is likely but target, effector, or compound-series mapping is partially implicit.\n\nSet review_required=true when:\n- E3 is propagated from series-level design rather than directly shown for the specific inducer;\n- target or effector assignment is implicit;\n- relation is extracted from a broad compound range;\n- evidence is design-class only and not mechanistically validated;\n- the final compound-to-effector mapping may require human confirmation.\n\nSet review_required=false when:\n- the specific inducer, target, and effector are directly linked by clear mechanism/model/design evidence;\n- the relation is a simple HyT/design-class relation with clear target and no named effector required.\n\nENUMS\nAllowed relation_type:\n{RELATION_TYPES}\n\nAllowed modality:\n{MODALITIES}\n\nAllowed mechanism_route:\n{MECHANISM_ROUTES}\n\nAllowed outcome_class:\n{OUTCOME_CLASSES}\nUse || to join multiple outcome_class values.\n\nAllowed mechanism_tags:\n{MECHANISM_TAGS}\nUse || to join multiple mechanism_tags.\n\nAllowed participant_role:\n{PARTICIPANT_ROLES}\n\nAllowed functional_role:\n{FUNCTIONAL_ROLES}\n\nSCHEMA\n{{"rt":"relation","relation_type":"targeted_degradation","modality":"PROTAC","inducer_name":"22b","mechanism_route":"E3_ligase_recruitment","relation_basis":"mechanistic_validation","outcome_class":"I_Protein_Abundance_Decrease","mechanism_tags":"E3_UPS","participants":[{{"name":"c-Met","participant_role":"degradation_target","functional_role":"target","entity_type_hint":"protein","variant_text":"","evidence_span":""}},{{"name":"CRBN","participant_role":"recruited_effector","functional_role":"effector","entity_type_hint":"protein","variant_text":"","evidence_span":""}}],"evidence_span":"","confidence":0.0,"review_required":false}}\n\nGOOD EXAMPLES\n\n{{"rt":"relation","relation_type":"targeted_degradation","modality":"PROTAC","inducer_name":"22b","mechanism_route":"E3_ligase_recruitment","relation_basis":"mechanistic_validation","outcome_class":"I_Protein_Abundance_Decrease","mechanism_tags":"E3_UPS","participants":[{{"name":"c-Met","participant_role":"degradation_target","functional_role":"target","entity_type_hint":"protein","variant_text":"","evidence_span":"compound 22b can bind the c-Met kinase domain"}},{{"name":"CRBN","participant_role":"recruited_effector","functional_role":"effector","entity_type_hint":"protein","variant_text":"","evidence_span":"can recruit CRBN to form a ternary complex with c-Met kinase domain and compound 22b"}}],"evidence_span":"compound 22b can bind the c-Met kinase domain ... can recruit CRBN to form a ternary complex with c-Met kinase domain and compound 22b","confidence":0.9,"review_required":false}}\n\n{{"rt":"relation","relation_type":"targeted_degradation","modality":"Hydrophobic_Tagging_Degrader","inducer_name":"14a","mechanism_route":"hydrophobic_tagging","relation_basis":"design_class","outcome_class":"I_Protein_Abundance_Decrease","mechanism_tags":"Hydrophobic_Tagging","participants":[{{"name":"c-Met","participant_role":"degradation_target","functional_role":"target","entity_type_hint":"protein","variant_text":"","evidence_span":"Structures and c-Met Degradation Activity of HyT Compounds 14a−14f and 17a−17f"}}],"evidence_span":"Structures and c-Met Degradation Activity of HyT Compounds 14a−14f and 17a−17f","confidence":0.75,"review_required":false}}\n\n{{"rt":"relation","relation_type":"targeted_degradation","modality":"PROTAC","inducer_name":"22e","mechanism_route":"E3_ligase_recruitment","relation_basis":"design_class","outcome_class":"I_Protein_Abundance_Decrease","mechanism_tags":"E3_UPS","participants":[{{"name":"c-Met","participant_role":"degradation_target","functional_role":"target","entity_type_hint":"protein","variant_text":"","evidence_span":"Structures and c-Met Degradation Activity of PROTACs 22a−22g, 24, 26, and 28a−28c"}},{{"name":"CRBN","participant_role":"recruited_effector","functional_role":"effector","entity_type_hint":"protein","variant_text":"","evidence_span":"compound S1 was selected as the most suitable E3 ligand"}}],"evidence_span":"compound S1 was selected as the most suitable E3 ligand ... Structures and c-Met Degradation Activity of PROTACs 22a−22g, 24, 26, and 28a−28c","confidence":0.75,"review_required":true}}\n\nBAD EXAMPLES. DO NOT OUTPUT THESE\n- {{"rt":"relation","relation_type":"other","inducer_name":"","participants":[]}}\n- inducer_name="compound 22b-treated"\n- inducer_name="compound S1" when S1 is only an E3 ligand or CRBN ligand\n- inducer_name="tepotinib" when tepotinib is only the target ligand/reference inhibitor\n- relation based only on NMR/HRMS/yield/synthesis text\n- participant name "HyT"\n- participant name "PROTAC"\n- participant name "E3 ligase" without a concrete named E3\n- participant name "proteasome" when only a degradation pathway is described\n- relation for an intermediate, linker, warhead, hydrophobic tag, or building block\n- one relation per dose/time/cell-line assay condition\n- one relation based only on DC50, Dmax, IC50, N.D., or degradation percentage\n\nKnown compound names:\n{known_names}\n\nARTICLE TEXT_EVIDENCE CONTEXT:\n{context}\n'


def uid(prefix: str, *parts: Any) -> str:
    return prefix + "_" + hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


def jdump(x: Any) -> str:
    return json.dumps(x if x is not None else {}, ensure_ascii=False, default=str)


def clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def norm_name(x: str) -> str:
    s = clean(x).lower()
    s = COMPOUND_PREFIX_RE.sub("", s)
    s = DOSE_PREFIX_RE.sub("", s)
    s = TREATMENT_SUFFIX_RE.sub("", s)
    return s.strip(" .,:;()[]{}")


def is_generic_name(x: str) -> bool:
    s = norm_name(x)
    return (not s) or len(s) > 120 or s in GENERIC_BAD_NAMES


def is_range_or_list_name(x: str) -> bool:
    s = clean(x)
    return bool(RANGE_RE.search(s) or LIST_SEP_RE.search(s))


def visible_in_context(name: str, context: str) -> bool:
    n = clean(name)
    if not n:
        return False
    if n in context:
        return True
    pat = re.compile(rf"(?<![A-Za-z0-9])(?:compound\s+|cmpd\s+|molecule\s+)?{re.escape(n)}(?![A-Za-z0-9])", re.I)
    return bool(pat.search(context))


def evidence_ok(evidence: str, context: str) -> bool:
    ev = clean(evidence)
    if len(ev) < 8:
        return False
    if ev in context:
        return True
    compact_ev = re.sub(r"\s+", " ", ev).lower()
    compact_ctx = re.sub(r"\s+", " ", context).lower()
    if compact_ev in compact_ctx:
        return True
    toks = [t.lower() for t in re.split(r"\W+", compact_ev) if len(t) >= 4]
    return len(toks) >= 4 and sum(1 for t in toks[:14] if t in compact_ctx) >= min(6, len(toks))


def extract_json_records(text: str) -> List[Dict[str, Any]]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:jsonl|json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    out = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            pass
    if out:
        return out
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("relations"), list):
            return [x for x in obj["relations"] if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        return []
    return []


def asset_is_supplementary(asset_type: str, section: str = "", file_path: str = "", metadata_json: str = "") -> bool:
    s = " ".join([asset_type or "", section or "", file_path or "", metadata_json or ""]).lower()
    return "supp" in s or "supplement" in s


def _context_scope_keep(is_supp: bool, scope: str) -> bool:
    return scope == "all" or (scope == "main" and not is_supp) or (scope == "supplementary" and is_supp)


def load_planned_pages(conn, doc_id: str) -> set:
    pages = set()
    try:
        rows = conn.execute(
            """
            SELECT a.page_no
            FROM planned_tasks p
            LEFT JOIN raw_asset a ON a.asset_id=p.asset_id
            WHERE p.doc_id=?
              AND p.task_type IN (
                'text_evidence','mechanism_figure','mixed_or_uncertain',
                'structure_figure','sar_table','supplementary_structure_table',
                'assay_table','supplementary_assay_table'
              )
            """,
            (doc_id,),
        ).fetchall()
        for r in rows:
            if r["page_no"] is not None:
                pages.add(int(r["page_no"]))
    except Exception:
        pass
    return pages


def load_context(conn, doc_id: str, max_context_chars: int, context_scope: str = "all", context_mode: str = "planned") -> str:
    planned_pages = load_planned_pages(conn, doc_id)
    main_chunks: List[str] = []
    supp_chunks: List[str] = []

    def add(is_supp: bool, chunk: str) -> None:
        if clean(chunk) and _context_scope_keep(is_supp, context_scope):
            (supp_chunks if is_supp else main_chunks).append(chunk)

    rows = conn.execute(
        """
        SELECT block_id, page_no, section, text, metadata_json
        FROM raw_text_block
        WHERE doc_id=?
        ORDER BY CASE WHEN page_no IS NULL THEN 999999 ELSE page_no END, block_id
        """,
        (doc_id,),
    ).fetchall()
    for r in rows:
        txt = clean(r["text"])
        if not txt:
            continue
        is_supp = asset_is_supplementary("", r["section"], "", r["metadata_json"])
        if context_mode == "planned" and not is_supp and planned_pages and r["page_no"] is not None and int(r["page_no"]) not in planned_pages:
            continue
        add(is_supp, f"[TEXT block_id={r['block_id']} page={r['page_no']} section={r['section']} supp={int(is_supp)}]\n{txt}")

    for r in conn.execute("SELECT figure_id, page_no, figure_ref, file_path, caption, metadata_json FROM raw_figure WHERE doc_id=? ORDER BY page_no, figure_id", (doc_id,)).fetchall():
        cap = clean(r["caption"])
        if not cap:
            continue
        is_supp = asset_is_supplementary("figure", "", r["file_path"], r["metadata_json"])
        if context_mode == "planned" and not is_supp and planned_pages and r["page_no"] is not None and int(r["page_no"]) not in planned_pages:
            continue
        add(is_supp, f"[FIGURE figure_id={r['figure_id']} page={r['page_no']} ref={r['figure_ref']} supp={int(is_supp)}]\n{cap}")

    for r in conn.execute("SELECT table_id, page_no, table_ref, file_path, table_json, metadata_json FROM raw_table WHERE doc_id=? ORDER BY page_no, table_id", (doc_id,)).fetchall():
        s = r["table_json"] or ""
        if not s:
            continue
        is_supp = asset_is_supplementary("table", "", r["file_path"], (r["metadata_json"] or "") + " " + (r["table_ref"] or ""))
        if context_mode == "planned" and not is_supp and planned_pages and r["page_no"] is not None and int(r["page_no"]) not in planned_pages:
            continue
        add(is_supp, f"[TABLE table_id={r['table_id']} page={r['page_no']} ref={r['table_ref']} supp={int(is_supp)}]\n{s[:12000]}")

    if context_scope == "main":
        return "\n\n".join(main_chunks)[:max_context_chars]
    if context_scope == "supplementary":
        return "\n\n".join(supp_chunks)[:max_context_chars]

    main_text = "\n\n".join(main_chunks)
    supp_text = "\n\n".join(supp_chunks)
    if not supp_text:
        return main_text[:max_context_chars]
    supp_budget = max(int(max_context_chars * 0.35), min(len(supp_text), 30000))
    main_budget = max_context_chars - supp_budget
    return (main_text[:main_budget] + "\n\n" + supp_text[:supp_budget]).strip()


def load_known_names(conn, doc_id: str) -> List[str]:
    names = []

    def add_name(value: Any) -> None:
        n = clean(value)
        if n and n not in names and not is_generic_name(n):
            names.append(n)

    sqls = [
        "SELECT name AS name FROM stg_agent WHERE doc_id=? AND COALESCE(name,'')!=''",
        "SELECT DISTINCT compound_name AS name FROM stg_component_relation WHERE doc_id=? AND COALESCE(compound_name,'')!=''",
        "SELECT DISTINCT compound_name AS name FROM stg_structure_candidate WHERE doc_id=? AND COALESCE(compound_name,'')!=''",
        "SELECT DISTINCT molecule_label AS name FROM stg_structure_candidate WHERE doc_id=? AND COALESCE(molecule_label,'')!=''",
    ]
    for sql in sqls:
        try:
            for r in conn.execute(sql, (doc_id,)).fetchall():
                add_name(r["name"])
        except Exception:
            continue

    try:
        for r in conn.execute("SELECT aliases_json FROM stg_agent WHERE doc_id=? AND COALESCE(aliases_json,'')!=''", (doc_id,)).fetchall():
            aliases = json.loads(r["aliases_json"] or "[]")
            if isinstance(aliases, list):
                for alias in aliases:
                    add_name(alias)
    except Exception:
        pass

    index_path = ROOT / "data" / "staging" / doc_id / "global_molecule_index.json"
    if index_path.exists():
        try:
            obj = json.loads(index_path.read_text(encoding="utf-8"))
            for mol in obj.get("molecules", []):
                if not isinstance(mol, dict):
                    continue
                add_name(mol.get("canonical_name"))
                add_name(mol.get("display_name"))
                for alias in mol.get("aliases", []) if isinstance(mol.get("aliases"), list) else []:
                    add_name(alias)
        except Exception:
            pass

    return sorted(names, key=lambda x: (len(x), x.lower()))


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + chunk_size])
        i += max(1, chunk_size - overlap)
    return out


def build_prompt(context: str, known_names: Sequence[str]) -> str:
    return PROMPT_TEMPLATE.format(
        RELATION_TYPES=RELATION_TYPES,
        MODALITIES=MODALITIES,
        MECHANISM_ROUTES=MECHANISM_ROUTES,
        OUTCOME_CLASSES=OUTCOME_CLASSES,
        MECHANISM_TAGS=MECHANISM_TAGS,
        PARTICIPANT_ROLES=PARTICIPANT_ROLES,
        FUNCTIONAL_ROLES=FUNCTIONAL_ROLES,
        known_names=json.dumps(known_names[:500], ensure_ascii=False),
        context=context,
        json=json,
        sorted=sorted,
    )


def normalize_inducer_name(raw_name: str, known_names: Sequence[str], context: str, strict_known: bool, reasons: List[str]) -> str:
    name = clean(raw_name)
    if not name:
        reasons.append("missing_inducer_name")
        return ""
    name = DOSE_PREFIX_RE.sub("", name)
    name = COMPOUND_PREFIX_RE.sub("", name)
    name = TREATMENT_SUFFIX_RE.sub("", name).strip(" .,:;()[]{}")
    known_by_norm = {}
    for k in sorted(known_names, key=lambda x: (len(norm_name(x)), len(x), x.lower())):
        known_by_norm.setdefault(norm_name(k), k)
    if norm_name(name) in known_by_norm:
        return known_by_norm[norm_name(name)]
    for k in sorted(known_names, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(k)}(?![A-Za-z0-9])", name, re.I):
            return k
    if is_generic_name(name):
        reasons.append("generic_inducer_name")
        return ""
    if is_range_or_list_name(name):
        reasons.append("range_or_list_inducer_name")
        return ""
    if strict_known:
        reasons.append("unknown_inducer_name_strict_mode")
        return ""
    if visible_in_context(name, context):
        reasons.append("new_inducer_name_from_context")
        return name
    reasons.append("new_inducer_name_not_visible_in_context")
    return name


def normalize_relation_type(x: str) -> str:
    v = clean(x) or "other"
    if v in RELATION_TYPES:
        return v
    if v.lower() in {"targeted degradation", "degradation"}:
        return "targeted_degradation"
    return "other"


def normalize_modality(x: str) -> str:
    v = clean(x) or "Other"
    if v in MODALITIES:
        return "Hydrophobic_Tagging_Degrader" if v == "HyT" else v
    return "Other"


def normalize_mechanism_route(x: str) -> str:
    v = clean(x) or "unknown"
    if v in MECHANISM_ROUTES:
        return v
    lower = v.lower()
    if "e3" in lower or "ligase" in lower:
        return "E3_ligase_recruitment"
    if "hydrophobic" in lower:
        return "hydrophobic_tagging"
    if "partner" in lower or "ppi" in lower:
        return "partner_protein_or_complex_induction"
    return "unknown"


def normalize_outcome_classes(x: str) -> str:
    vals = [clean(v) for v in str(x or "").split("||") if clean(v)]
    vals = [v if v in OUTCOME_CLASSES else "Other" for v in vals]
    return "||".join(vals) if vals else "Other"


def participant_role_to_simple_role(participant_role: str, functional_role: str) -> str:
    f = clean(functional_role)
    if f == "partner":
        return "recruited_entity"
    if f in {"target", "effector", "cell", "other"}:
        return f
    pr = clean(participant_role)
    if pr in {"degradation_target", "stabilization_target", "regulated_target", "immune_target_antigen", "target"}:
        return "target"
    if pr in {"recruited_effector", "nuclease_effector", "editing_effector", "PTM_writer", "PTM_eraser", "trafficking_receptor", "autophagy_receptor_or_adapter", "immune_effector_marker", "effector"}:
        return "effector"
    if pr in {"proximal_partner", "partner"}:
        return "recruited_entity"
    if "cell" in pr:
        return "cell"
    return "other"


def entity_type_from_hint(x: str) -> str:
    v = clean(x)
    if v in ENTITY_TYPES:
        return v
    lv = v.lower()
    if "rna" in lv:
        return "RNA"
    if "dna" in lv:
        return "DNA"
    if "oligo" in lv or "aso" in lv or "sirna" in lv or "sgrna" in lv:
        return "oligo"
    if "antibody" in lv or "scfv" in lv or "nanobody" in lv:
        return "antibody"
    if "cell" in lv:
        return "cell"
    if "construct" in lv:
        return "construct"
    if "peptide" in lv:
        return "peptide"
    return "protein" if v else "other"


def find_evidence_source(context: str, evidence: str) -> Dict[str, Any]:
    ev = clean(evidence)
    if not ev:
        return {}
    idx = context.find(ev)
    if idx < 0:
        idx = context.lower().find(ev.lower())
    if idx < 0:
        return {}
    header_start = context.rfind("[", 0, idx)
    header_end = context.find("]", header_start, min(len(context), header_start + 500))
    header = context[header_start:header_end + 1] if header_start >= 0 and header_end > header_start else ""
    out: Dict[str, Any] = {"source_header": header}
    for key, pat in {
        "source_block_id": r"block_id=([^\s\]]+)",
        "source_table_id": r"table_id=([^\s\]]+)",
        "source_figure_id": r"figure_id=([^\s\]]+)",
        "source_page_no": r"page=([^\s\]]+)",
    }.items():
        m = re.search(pat, header)
        if m:
            val: Any = m.group(1)
            if key == "source_page_no":
                try:
                    val = int(val) if val not in {"None", "null", ""} else None
                except Exception:
                    val = None
            out[key] = val
    return out


def validate_relation(rec: Dict[str, Any], context: str, known_names: Sequence[str], strict_known: bool) -> Optional[Dict[str, Any]]:
    if rec.get("rt") != "relation":
        return None
    reasons: List[str] = []
    rec["relation_type"] = normalize_relation_type(rec.get("relation_type"))
    rec["modality"] = normalize_modality(rec.get("modality"))
    rec["mechanism_route"] = normalize_mechanism_route(rec.get("mechanism_route"))
    rec["relation_basis"] = clean(rec.get("relation_basis")) if clean(rec.get("relation_basis")) in RELATION_BASIS else "design_class"
    rec["outcome_class"] = normalize_outcome_classes(rec.get("outcome_class"))
    rec["mechanism_tags"] = clean(rec.get("mechanism_tags"))
    rec["evidence_span"] = clean(rec.get("evidence_span"))
    rec["inducer_name"] = normalize_inducer_name(rec.get("inducer_name", ""), known_names, context, strict_known, reasons)
    try:
        rec["confidence"] = max(0.0, min(1.0, float(rec.get("confidence", 0.0))))
    except Exception:
        rec["confidence"] = 0.0
        reasons.append("invalid_confidence")
    if not rec["inducer_name"]:
        reasons.append("invalid_or_missing_inducer")
    if not evidence_ok(rec["evidence_span"], context):
        reasons.append("evidence_span_not_found_or_empty")
    if NEGATIVE_TEXT.search(rec["evidence_span"]):
        reasons.append("synthesis_or_characterization_like_evidence")

    clean_parts: List[Dict[str, Any]] = []
    raw_parts = rec.get("participants") or []
    if not isinstance(raw_parts, list):
        raw_parts = []
    for p in raw_parts:
        if not isinstance(p, dict):
            continue
        name = clean(p.get("name") or p.get("entity_name"))
        if not name or is_generic_name(name):
            reasons.append("empty_or_generic_participant_name")
            continue
        participant_role = clean(p.get("participant_role")) or "other"
        functional_role = clean(p.get("functional_role")) or "other"
        clean_parts.append({
            "name": name,
            "participant_role": participant_role,
            "functional_role": functional_role,
            "entity_type_hint": entity_type_from_hint(p.get("entity_type_hint") or p.get("entity_type")),
            "variant_text": clean(p.get("variant_text")),
            "evidence_span": clean(p.get("evidence_span")) or rec["evidence_span"],
            "role": participant_role_to_simple_role(participant_role, functional_role),
        })
    if not clean_parts:
        reasons.append("missing_biological_participants")
    rec["participants"] = clean_parts
    rec["source"] = find_evidence_source(context, rec["evidence_span"])
    rec["qc_reasons"] = sorted(set(reasons))
    rec["review_required"] = bool(rec.get("review_required", False) or reasons or rec["confidence"] < 0.75)
    if "invalid_or_missing_inducer" in reasons or "missing_biological_participants" in reasons:
        return None
    return rec


def infer_agent_type(inducer_name: str, modality: str) -> str:
    m = modality
    n = inducer_name.lower()
    if m == "Molecular_Glue_Degrader":
        return "glue"
    if m == "Hydrophobic_Tagging_Degrader":
        return "HyT"
    if m in {"PROTAC", "SNIPER", "DUBTAC", "PhosTAC_or_PhosTAP", "TF-DUBTAC"}:
        return "heterobifunctional"
    if m in {"BiTE", "TriTE_or_Multispecific_TCE", "NK_Engager_or_NKCE", "AbTAC", "LYTAC"}:
        return "antibody"
    if "aso" in n or "sirna" in n or "sgrna" in n or "oligo" in n:
        return "oligo"
    return "small_molecule"


def insert_agent_for_inducer(conn, doc_id: str, relation_id: str, rec: Dict[str, Any]) -> Dict[str, str]:
    name = rec["inducer_name"]
    stg_id = uid("agent", doc_id, name)
    record = {
        "source": "extract_ipm_knowledge",
        "relation_id": relation_id,
        "relation_basis": rec.get("relation_basis"),
        "modality": rec.get("modality"),
        "new_name_flag": "new_inducer_name_from_context" in rec.get("qc_reasons", []),
    }
    conn.execute(
        """
        INSERT INTO stg_agent
        (stg_id, doc_id, name, normalized_name, agent_type, structure_status, evidence_json,
         record_json, confidence, review_required, qc_reasons_json, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stg_id) DO UPDATE SET
            agent_type=excluded.agent_type,
            evidence_json=excluded.evidence_json,
            record_json=excluded.record_json,
            confidence=MAX(COALESCE(stg_agent.confidence,0), COALESCE(excluded.confidence,0)),
            review_required=CASE WHEN stg_agent.review_required=1 OR excluded.review_required=1 THEN 1 ELSE 0 END,
            qc_reasons_json=excluded.qc_reasons_json,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            stg_id, doc_id, name, name.lower(), infer_agent_type(name, rec.get("modality", "Other")), "missing",
            jdump([{"relation_id": relation_id, "evidence_text": rec.get("evidence_span", "")}]),
            jdump(record), rec.get("confidence"), int(rec.get("review_required", True)),
            jdump(rec.get("qc_reasons", [])), "extracted",
        ),
    )
    return {name: stg_id}


def relation_name_from_rec(rec: Dict[str, Any]) -> str:
    targets = [p["name"] for p in rec.get("participants", []) if p.get("role") == "target"]
    effectors = [p["name"] for p in rec.get("participants", []) if p.get("role") == "effector"]
    parts = []
    if targets:
        parts.append("/".join(targets[:2]))
    if effectors:
        parts.append("/".join(effectors[:2]))
    parts.append(rec.get("inducer_name", ""))
    return "-".join([x for x in parts if x])


def insert_relation(conn, doc_id: str, rec: Dict[str, Any]) -> str:
    participant_names = ",".join([p["name"] for p in rec.get("participants", [])])
    relation_id = uid("rel", doc_id, rec.get("inducer_name", ""), rec.get("mechanism_route", ""), participant_names, rec.get("evidence_span", "")[:160])
    src = rec.get("source") or {}
    conn.execute(
        """
        INSERT OR REPLACE INTO stg_relation
        (relation_id, doc_id, relation_type, modality, outcome_class, mechanism_route, intended_effect,
         relation_name, evidence_text, evidence_source, source_block_id, source_page_no,
         record_json, raw_output, confidence, review_required, qc_reasons_json, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            relation_id, doc_id, rec["relation_type"], rec["modality"], rec["outcome_class"],
            rec["mechanism_route"], rec.get("relation_basis", ""), relation_name_from_rec(rec),
            rec["evidence_span"], "llm_text_or_supplement", src.get("source_block_id"), src.get("source_page_no"),
            jdump(rec), jdump(rec), rec.get("confidence"), int(rec.get("review_required", True)),
            jdump(rec.get("qc_reasons", [])), "pending_qc",
        ),
    )
    return relation_id


def insert_participants(conn, doc_id: str, relation_id: str, rec: Dict[str, Any], agent_map: Dict[str, str]) -> None:
    src = rec.get("source") or {}
    synthetic = {
        "name": rec["inducer_name"], "role": "inducer", "participant_role": "inducer",
        "functional_role": "inducer", "entity_type_hint": "compound",
        "variant_text": "", "evidence_span": rec.get("evidence_span", ""),
    }
    for p in [synthetic] + list(rec.get("participants", [])):
        simple_role = p.get("role") or participant_role_to_simple_role(p.get("participant_role", ""), p.get("functional_role", ""))
        entity_type = "compound" if simple_role == "inducer" else entity_type_from_hint(p.get("entity_type_hint", ""))
        participant_id = uid("part", doc_id, relation_id, simple_role, p["name"])
        raw = dict(p)
        raw.update({"parent_relation_raw_output": rec})
        conn.execute(
            """
            INSERT OR REPLACE INTO stg_relation_participant
            (participant_id, relation_id, doc_id, entity_name, canonical_name, entity_type, role, role_detail,
             species, agent_stg_id, evidence_text, source_block_id, source_page_no,
             raw_output, confidence, review_required, qc_reasons_json, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                participant_id, relation_id, doc_id, p["name"], "", entity_type, simple_role,
                p.get("participant_role", ""), "", agent_map.get(p["name"], ""), p.get("evidence_span") or rec.get("evidence_span", ""),
                src.get("source_block_id"), src.get("source_page_no"), jdump(raw), rec.get("confidence"),
                int(rec.get("review_required", True)), jdump(rec.get("qc_reasons", [])), "pending_qc",
            ),
        )


def call_llm(client: OpenAI, model: str, prompt: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "ipm-llm"))
    ap.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    ap.add_argument("--max-context-chars", type=int, default=180000)
    ap.add_argument("--context-scope", choices=["all", "main", "supplementary"], default="all")
    ap.add_argument("--context-mode", choices=["all", "planned"], default="planned")
    ap.add_argument("--chunk-size", type=int, default=180000)
    ap.add_argument("--overlap", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--strict-known-agents", action="store_true", help="Disable new inducer names; only allow Known compound names.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = get_conn()
    if args.overwrite:
        rels = conn.execute("SELECT relation_id FROM stg_relation WHERE doc_id=?", (args.doc_id,)).fetchall()
        for r in rels:
            conn.execute("DELETE FROM stg_relation_participant WHERE relation_id=?", (r["relation_id"],))
        conn.execute("DELETE FROM stg_relation WHERE doc_id=?", (args.doc_id,))
        conn.commit()

    context = load_context(conn, args.doc_id, args.max_context_chars, args.context_scope, args.context_mode)
    known_names = load_known_names(conn, args.doc_id)
    chunks = chunk_text(context, args.chunk_size, args.overlap)
    if args.limit:
        chunks = chunks[:args.limit]

    out_dir = ROOT / "data" / "staging" / args.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"ipm_knowledge_{args.context_scope}"
    raw_path = out_dir / f"{prefix}_raw_llm.jsonl"
    val_path = out_dir / f"{prefix}_validated.jsonl"
    report_path = out_dir / f"{prefix}_extraction_report.json"
    prompt_path = out_dir / f"{prefix}_prompt_template.txt"
    prompt_path.write_text(PROMPT_TEMPLATE, encoding="utf-8")

    client = OpenAI(base_url=args.llm_base_url, api_key=args.llm_api_key)
    stats = Counter()
    seen = set()

    with raw_path.open("w", encoding="utf-8") as fraw, val_path.open("w", encoding="utf-8") as fval:
        for i, ctx in enumerate(tqdm(chunks, desc="Extract IPM relations")):
            prompt = build_prompt(ctx, known_names)
            try:
                raw = call_llm(client, args.llm_model, prompt, args.max_tokens)
            except Exception as e:
                stats["llm_error"] += 1
                fraw.write(jdump({"chunk": i, "error": str(e)}) + "\n")
                continue
            fraw.write(jdump({"chunk": i, "raw_text": raw}) + "\n")
            for rec0 in extract_json_records(raw):
                stats["raw_records"] += 1
                rec = validate_relation(rec0, ctx, known_names, args.strict_known_agents)
                if not rec:
                    stats["invalid_record"] += 1
                    continue
                key = (rec.get("inducer_name"), rec.get("mechanism_route"), tuple(sorted(p["name"] for p in rec.get("participants", []))), rec.get("evidence_span", "")[:120])
                if key in seen:
                    stats["deduped"] += 1
                    continue
                seen.add(key)
                fval.write(jdump(rec) + "\n")
                if "new_inducer_name_from_context" in rec.get("qc_reasons", []):
                    stats["new_inducer_names"] += 1
                stats["review_required" if rec.get("review_required") else "accepted"] += 1
                if not args.dry_run:
                    relation_id = insert_relation(conn, args.doc_id, rec)
                    agent_map = insert_agent_for_inducer(conn, args.doc_id, relation_id, rec)
                    insert_participants(conn, args.doc_id, relation_id, rec, agent_map)
                    conn.commit()
                stats["relations"] += 1

    report = {
        "doc_id": args.doc_id,
        "context_scope": args.context_scope,
        "context_mode": args.context_mode,
        "num_chunks": len(chunks),
        "num_known_names": len(known_names),
        "strict_known_agents": bool(args.strict_known_agents),
        "stats": dict(stats),
        "raw_llm_jsonl": str(raw_path),
        "validated_jsonl": str(val_path),
        "prompt_template": str(prompt_path),
        "tables": ["stg_agent", "stg_relation", "stg_relation_participant"],
    }
    report_path.write_text(jdump(report), encoding="utf-8")
    conn.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
