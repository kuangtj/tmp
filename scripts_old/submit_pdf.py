#!/usr/bin/env python3
import re
import sys
import json
import shutil
import hashlib
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pypdf import PdfReader
from ipm_eagle.db.sqlite import get_conn
from ipm_eagle.db.local_queue import enqueue


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def extract_pdf_meta(pdf: Path):
    text = ""
    title = ""
    doi = ""
    pmid = ""

    try:
        reader = PdfReader(str(pdf))
        meta = reader.metadata or {}
        title = (meta.get("/Title") or "").strip()

        for page in reader.pages[:3]:
            text += "\n" + (page.extract_text() or "")

        m = re.search(r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b', text, re.I)
        if m:
            doi = m.group(0).rstrip(".,);").strip()

        m = re.search(r'\bPMID[:\s]*(\d{6,10})\b', text, re.I)
        if m:
            pmid = m.group(1).strip()

        if not title:
            lines = [x.strip() for x in text.splitlines() if len(x.strip()) > 20]
            title = lines[0] if lines else pdf.stem

        return {
            "ok": True,
            "title": title,
            "doi": doi,
            "pmid": pmid,
            "error": "",
        }

    except Exception as e:
        return {
            "ok": False,
            "title": pdf.stem,
            "doi": "",
            "pmid": "",
            "error": str(e),
        }


def make_doc_id(meta, pdf: Path):
    if meta.get("doi"):
        return "doi_" + short_hash(meta["doi"].lower())
    if meta.get("pmid"):
        return "pmid_" + meta["pmid"]
    if meta.get("title"):
        return "title_" + short_hash(meta["title"].lower())
    return "file_" + sha1_file(pdf)[:16]


def copy_supplementary(src: Path, dst: Path):
    if not src:
        return ""

    src = Path(src)
    if not src.exists():
        return ""

    dst.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        for p in src.iterdir():
            target = dst / p.name
            if p.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(p, target)
            else:
                shutil.copy2(p, target)
    else:
        shutil.copy2(src, dst / src.name)

    return str(dst)


def upsert_raw_document(doc_id, meta, pdf_dst, supp_dst, status):
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO raw_document (
            doc_id, title, doi, pmid, source_pdf_path, supplement_dir, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            title=excluded.title,
            doi=excluded.doi,
            pmid=excluded.pmid,
            source_pdf_path=excluded.source_pdf_path,
            supplement_dir=excluded.supplement_dir,
            status=excluded.status
        """,
        (
            doc_id,
            meta.get("title", ""),
            meta.get("doi", ""),
            meta.get("pmid", ""),
            str(pdf_dst),
            str(supp_dst) if supp_dst else "",
            status,
        ),
    )
    conn.commit()
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--supp-dir", default="")
    args = ap.parse_args()

    pdf = Path(args.pdf).resolve()
    supp = Path(args.supp_dir).resolve() if args.supp_dir else None

    if not pdf.exists() or pdf.stat().st_size == 0:
        doc_id = "file_" + short_hash(str(pdf))
        meta = {"title": pdf.stem, "doi": "", "pmid": ""}
        raw_dir = Path("data/raw") / doc_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        pdf_dst = raw_dir / "paper.pdf"
        upsert_raw_document(doc_id, meta, pdf_dst, "", "FAILED_INGEST")
        print(json.dumps({"doc_id": doc_id, "status": "FAILED_INGEST", "reason": "missing_or_empty_pdf"}, ensure_ascii=False, indent=2))
        return

    meta = extract_pdf_meta(pdf)
    doc_id = make_doc_id(meta, pdf)

    raw_dir = Path("data/raw") / doc_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    pdf_dst = raw_dir / "paper.pdf"
    shutil.copy2(pdf, pdf_dst)

    supp_dst = ""
    if supp:
        supp_dst = copy_supplementary(supp, raw_dir / "supplementary")

    status = "created" if meta["ok"] else "FAILED_INGEST"
    upsert_raw_document(doc_id, meta, pdf_dst, supp_dst, status)

    task_id = ""
    if status == "created":
        task_id = enqueue(
            doc_id=doc_id,
            queue_name="cpu_queue",
            task_type="parse_pdf",
            payload={
                "doc_id": doc_id,
                "pdf_path": str(pdf_dst),
                "supplement_dir": str(supp_dst),
            },
        )

    print(json.dumps({
        "doc_id": doc_id,
        "status": status,
        "title": meta.get("title", ""),
        "doi": meta.get("doi", ""),
        "pmid": meta.get("pmid", ""),
        "pdf_path": str(pdf_dst),
        "supplement_dir": str(supp_dst),
        "task_id": task_id,
        "error": meta.get("error", ""),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
