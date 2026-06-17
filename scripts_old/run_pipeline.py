#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import shlex
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

STAGES = [
    "submit",
    "parse",
    "plan",
    "chemeagle",
    "supp_direct",
    "qc",
    "align",
    "reconstruct",
    "relation",
    "assay",
    "participants",
]


def run_cmd(cmd, dry_run=False):
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return ""
    p = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if p.stdout:
        print(p.stdout, end="")
    if p.stderr:
        print(p.stderr, end="", file=sys.stderr)
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.stdout or ""


def parse_doc_id_from_stdout(text):
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
        return obj.get("doc_id", "")
    except Exception:
        m = re.search(r'"doc_id"\s*:\s*"([^"]+)"', text or "")
        return m.group(1) if m else ""


def selected_stages(args):
    if args.stages:
        stages = [x.strip() for x in args.stages.split(",") if x.strip()]
    else:
        stages = list(STAGES)

    unknown = [x for x in stages if x not in STAGES]
    if unknown:
        raise SystemExit(f"Unknown stages: {unknown}; allowed={STAGES}")

    if args.from_stage:
        if args.from_stage not in STAGES:
            raise SystemExit(f"Unknown --from-stage: {args.from_stage}")
        stages = [x for x in stages if STAGES.index(x) >= STAGES.index(args.from_stage)]

    if args.to_stage:
        if args.to_stage not in STAGES:
            raise SystemExit(f"Unknown --to-stage: {args.to_stage}")
        stages = [x for x in stages if STAGES.index(x) <= STAGES.index(args.to_stage)]

    skip = {x.strip() for x in args.skip_stage.split(",") if x.strip()}
    stages = [x for x in stages if x not in skip]

    if not args.pdf and "submit" in stages:
        stages.remove("submit")

    return stages


def main():
    ap = argparse.ArgumentParser(description="Run the IPM PDF-to-staging pipeline with explicit stage control.")
    ap.add_argument("--pdf", default="", help="Input PDF. Required only when running submit stage.")
    ap.add_argument("--supp-dir", default="", help="Supplementary directory or file for submit stage.")
    ap.add_argument("--doc-id", default="", help="Existing doc_id. Required when submit is skipped.")

    ap.add_argument("--stages", default="", help="Comma-separated stages to run. Default: all stages.")
    ap.add_argument("--from-stage", default="", choices=[""] + STAGES)
    ap.add_argument("--to-stage", default="", choices=[""] + STAGES)
    ap.add_argument("--skip-stage", default="", help="Comma-separated stages to skip.")

    ap.add_argument("--overwrite", action="store_true", help="Pass --overwrite to overwrite-capable stages.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running them.")

    ap.add_argument("--dots-root", default="/root/autodl-tmp/dots.ocr")
    ap.add_argument("--dots-python", default=sys.executable)
    ap.add_argument("--num-thread", type=int, default=16)
    ap.add_argument("--skip-supp-pdf-ocr", action="store_true")

    ap.add_argument("--chemeagle-root", default=os.getenv("CHEMEAGLE_ROOT", "external/ChemEagle"))
    ap.add_argument("--vlm-model", default=os.getenv("VLLM_MODEL", "ipm-vlm"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "ipm-llm"))
    ap.add_argument("--include-page-images", action="store_true")
    ap.add_argument("--only-auto-pass", action="store_true", default=True)
    ap.add_argument("--default-taxon-id", default="9606")

    args = ap.parse_args()
    stages = selected_stages(args)

    doc_id = args.doc_id

    if "submit" in stages:
        if not args.pdf:
            raise SystemExit("--pdf is required for submit stage")
        cmd = [sys.executable, str(SCRIPTS / "submit_pdf.py"), "--pdf", args.pdf]
        if args.supp_dir:
            cmd += ["--supp-dir", args.supp_dir]
        stdout = run_cmd(cmd, args.dry_run)
        if not args.dry_run:
            doc_id = parse_doc_id_from_stdout(stdout)
            if not doc_id:
                raise SystemExit("Could not parse doc_id from submit_pdf.py output")

    if not doc_id:
        raise SystemExit("--doc-id is required when submit stage is not run")

    for stage in stages:
        if stage == "submit":
            continue

        if stage == "parse":
            cmd = [
                sys.executable, str(SCRIPTS / "parse_pdf.py"),
                "--doc-id", doc_id,
                "--dots-root", args.dots_root,
                "--dots-python", args.dots_python,
                "--num-thread", str(args.num_thread),
            ]
            if args.skip_supp_pdf_ocr:
                cmd.append("--skip-supp-pdf-ocr")

        elif stage == "plan":
            cmd = [
                sys.executable, str(SCRIPTS / "plan_paper_vlm.py"),
                "--doc-id", doc_id,
                "--model", args.vlm_model,
            ]
            if args.include_page_images:
                cmd.append("--include-page-images")

        elif stage == "chemeagle":
            cmd = [
                sys.executable, str(SCRIPTS / "run_chemeagle_structures.py"),
                "--doc-id", doc_id,
                "--chemeagle-root", args.chemeagle_root,
                "--model-name", args.vlm_model,
            ]
            if args.overwrite:
                cmd.append("--overwrite")

        elif stage == "supp_direct":
            cmd = [sys.executable, str(SCRIPTS / "extract_supplement_direct_structures.py"), "--doc-id", doc_id]
            if args.overwrite:
                cmd.append("--overwrite")

        elif stage == "qc":
            cmd = [
                sys.executable, str(SCRIPTS / "qc_structure_candidates.py"),
                "--doc-id", doc_id,
                "--vlm-model", args.vlm_model,
            ]

        elif stage == "align":
            cmd = [
                sys.executable, str(SCRIPTS / "align_compound_structure.py"),
                "--doc-id", doc_id,
                "--vlm-model", args.vlm_model,
            ]
            if args.only_auto_pass:
                cmd.append("--only-auto-pass")
            if args.overwrite:
                cmd.append("--overwrite")

        elif stage == "reconstruct":
            cmd = [sys.executable, str(SCRIPTS / "reconstruct_compounds.py"), "--doc-id", doc_id]
            if args.overwrite:
                cmd.append("--overwrite")

        elif stage == "relation":
            cmd = [
                sys.executable, str(SCRIPTS / "extract_ipm_knowledge.py"),
                "--doc-id", doc_id,
                "--llm-model", args.vlm_model,
            ]
            if args.overwrite:
                cmd.append("--overwrite")

        elif stage == "assay":
            cmd = [
                sys.executable, str(SCRIPTS / "extract_assays.py"),
                "--doc-id", doc_id,
                "--llm-model", args.vlm_model,
            ]
            if args.overwrite:
                cmd.append("--overwrite")

        elif stage == "participants":
            cmd = [
                sys.executable, str(SCRIPTS / "standardize_relation_participants.py"),
                "--doc-id", doc_id,
                "--llm-model", args.vlm_model,
                "--default-taxon-id", args.default_taxon_id,
            ]

        else:
            raise SystemExit(f"Unhandled stage: {stage}")

        run_cmd(cmd, args.dry_run)

    print(json.dumps({"doc_id": doc_id, "stages": stages, "status": "done"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
