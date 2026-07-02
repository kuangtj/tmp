#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submit one paper PDF and optional supplementary directory/file into local IPM DB.

Output:
- raw_document
- local files under data/raw/{doc_id}/
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pypdf import PdfReader
from ipm_eagle.db.sqlite import get_conn

try:
    from ipm_eagle.db.local_queue import enqueue
except Exception:  # queue is optional for local/manual runs
    enqueue = None


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(text: str, n: int = 16) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:n]


def normalize_doi(x: str) -> str:
    x = (x or "").strip().strip(". ,;()[]{}")
    return x.lower()


DOI_STRICT_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
DOI_LOOSE_RE = re.compile(r"\b10\.\d{4,9}/(?:[-._;()/:A-Z0-9]|\s){6,200}", re.I)


def extract_doi_from_text(text: str) -> str:
    candidates = []

    for m in DOI_STRICT_RE.finditer(text or ""):
        candidates.append(normalize_doi(m.group(0)))

    for m in DOI_LOOSE_RE.finditer(text or ""):
        compact = re.sub(r"\s+", "", m.group(0))
        m2 = DOI_STRICT_RE.match(compact)
        if m2:
            candidates.append(normalize_doi(m2.group(0)))

    if not candidates:
        return ""

    # Prefer the longest candidate so truncated first-page DOI strings lose to fuller matches.
    return max(candidates, key=len)


def extract_pdf_meta(pdf: Path) -> Dict[str, str]:
    text = ""
    title = ""
    doi = ""
    pmid = ""
    journal = ""
    year = ""

    try:
        reader = PdfReader(str(pdf))
        meta = reader.metadata or {}
        title = str(meta.get("/Title") or "").strip()

        for page in reader.pages[:3]:
            text += "\n" + (page.extract_text() or "")

        doi = extract_doi_from_text(text)

        m = re.search(r"\bPMID[:\s]*(\d{6,10})\b", text, re.I)
        if m:
            pmid = m.group(1).strip()

        m = re.search(r"\b(19|20)\d{2}\b", text)
        if m:
            year = m.group(0)

        if not title:
            lines = [x.strip() for x in text.splitlines() if len(x.strip()) > 20]
            title = lines[0] if lines else pdf.stem

        return {
            "ok": True,
            "title": title,
            "doi": doi,
            "pmid": pmid,
            "journal": journal,
            "year": year,
            "error": "",
            "first_pages_text_preview": text[:4000],
        }
    except Exception as e:
        return {
            "ok": False,
            "title": pdf.stem,
            "doi": "",
            "pmid": "",
            "journal": "",
            "year": "",
            "error": str(e),
            "first_pages_text_preview": "",
        }


def make_doc_id(meta: Dict[str, str], pdf: Path) -> str:
    if meta.get("doi"):
        return "doi_" + short_hash(meta["doi"].lower())
    if meta.get("pmid"):
        return "pmid_" + meta["pmid"]
    if meta.get("title"):
        return "title_" + short_hash(meta["title"].lower())
    return "file_" + sha1_file(pdf)[:16]


def copy_supplementary(src: Optional[Path], dst: Path) -> str:
    if not src:
        return ""
    src = Path(src)
    if not src.exists():
        return ""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for p in src.iterdir():
            target = dst / p.name
            if p.is_dir():
                shutil.copytree(p, target)
            else:
                shutil.copy2(p, target)
    else:
        shutil.copy2(src, dst / src.name)
    return str(dst)


def upsert_raw_document(doc_id: str, meta: Dict[str, str], pdf_dst: Path, supp_dst: str, status: str, source_pdf: Path) -> None:
    metadata = {
        "source_input_pdf": str(source_pdf),
        "pdf_sha1": sha1_file(pdf_dst) if pdf_dst.exists() else "",
        "ingest_error": meta.get("error", ""),
        "first_pages_text_preview": meta.get("first_pages_text_preview", "")[:2000],
    }
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO raw_document (
            doc_id, title, doi, pmid, journal, year,
            source_pdf_path, supplement_dir, status, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            title=excluded.title,
            doi=excluded.doi,
            pmid=excluded.pmid,
            journal=excluded.journal,
            year=excluded.year,
            source_pdf_path=excluded.source_pdf_path,
            supplement_dir=excluded.supplement_dir,
            status=excluded.status,
            metadata_json=excluded.metadata_json,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            doc_id,
            meta.get("title", ""),
            meta.get("doi", ""),
            meta.get("pmid", ""),
            meta.get("journal", ""),
            meta.get("year", ""),
            str(pdf_dst),
            supp_dst or "",
            status,
            json.dumps(metadata, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--supp-dir", default="")
    ap.add_argument("--no-enqueue", action="store_true")
    args = ap.parse_args()

    pdf = Path(args.pdf).expanduser().resolve()
    supp = Path(args.supp_dir).expanduser().resolve() if args.supp_dir else None

    if not pdf.exists() or not pdf.is_file() or pdf.stat().st_size == 0:
        doc_id = "file_" + short_hash(str(pdf))
        raw_dir = ROOT / "data" / "raw" / doc_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        pdf_dst = raw_dir / "paper.pdf"
        meta = {"title": pdf.stem, "doi": "", "pmid": "", "journal": "", "year": "", "error": "missing_or_empty_pdf"}
        upsert_raw_document(doc_id, meta, pdf_dst, "", "failed_ingest", pdf)
        print(json.dumps({"doc_id": doc_id, "status": "failed_ingest", "reason": "missing_or_empty_pdf"}, ensure_ascii=False, indent=2))
        return

    meta = extract_pdf_meta(pdf)
    doc_id = make_doc_id(meta, pdf)
    raw_dir = ROOT / "data" / "raw" / doc_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    pdf_dst = raw_dir / "paper.pdf"
    shutil.copy2(pdf, pdf_dst)

    supp_dst = ""
    status = "pending_parse" if meta.get("ok") else "failed_ingest"
    
    if supp:
        if not supp.exists():
            # 如果路径不存在，记录错误并拦截任务
            error_msg = f"supplement_dir_not_found: {args.supp_dir}"
            meta["error"] = (meta.get("error", "") + f" | {error_msg}").strip(" |")
            status = "failed_ingest"
        else:
            # 路径存在，正常拷贝
            supp_dst = copy_supplementary(supp, raw_dir / "supplementary")
    # ----------------------------------------

    upsert_raw_document(doc_id, meta, pdf_dst, supp_dst, status, pdf)

    
#    supp_dst = copy_supplementary(supp, raw_dir / "supplementary") if supp else ""
#
#    status = "pending_parse" if meta.get("ok") else "failed_ingest"
#    upsert_raw_document(doc_id, meta, pdf_dst, supp_dst, status, pdf)

    task_id = ""
    if status == "pending_parse" and not args.no_enqueue and enqueue is not None:
        task_id = enqueue(
            doc_id=doc_id,
            queue_name="cpu_queue",
            task_type="parse_pdf",
            payload={"doc_id": doc_id, "pdf_path": str(pdf_dst), "supplement_dir": str(supp_dst)},
        )

    print(json.dumps({
        "doc_id": doc_id,
        "status": status,
        "title": meta.get("title", ""),
        "doi": meta.get("doi", ""),
        "pmid": meta.get("pmid", ""),
        "pdf_path": str(pdf_dst),
        "supplement_dir": supp_dst,
        "task_id": task_id,
        "error": meta.get("error", ""),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
