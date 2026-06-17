#!/usr/bin/env python3
import re
import sys
import json
import uuid
import time
import shutil
import argparse
import subprocess
from pathlib import Path

import fitz
import pandas as pd
from PIL import Image
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn


def uid(prefix, *parts):
    return prefix + "_" + uuid.uuid5(uuid.NAMESPACE_URL, "|".join(map(str, parts))).hex[:16]


def jdump(x):
    return json.dumps(x or {}, ensure_ascii=False)


def insert_text(conn, block_id, doc_id, page_no, section, text, meta=None):
    conn.execute(
        """
        INSERT OR REPLACE INTO raw_text_block
        (block_id, doc_id, page_no, section, text, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (block_id, doc_id, page_no, section, text or "", jdump(meta)),
    )


def insert_asset(conn, asset_id, doc_id, asset_type, page_no, figure_ref, table_ref, file_path, meta=None):
    conn.execute(
        """
        INSERT OR REPLACE INTO raw_asset
        (asset_id, doc_id, asset_type, page_no, figure_ref, table_ref, file_path, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (asset_id, doc_id, asset_type, page_no, figure_ref or "", table_ref or "", str(file_path), jdump(meta)),
    )


def insert_figure(conn, figure_id, doc_id, page_no, figure_ref, file_path, caption, meta=None):
    conn.execute(
        """
        INSERT OR REPLACE INTO raw_figure
        (figure_id, doc_id, page_no, figure_ref, file_path, caption, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (figure_id, doc_id, page_no, figure_ref, str(file_path), caption or "", jdump(meta)),
    )


def insert_table(conn, table_id, doc_id, page_no, table_ref, file_path, table_json=None):
    conn.execute(
        """
        INSERT OR REPLACE INTO raw_table
        (table_id, doc_id, page_no, table_ref, file_path, table_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (table_id, doc_id, page_no, table_ref, str(file_path), jdump(table_json)),
    )


TEXT_ASSET_SECTIONS = {
    "Title",
    "Section-header",
    "Text",
    "Caption",
    "supplementary_Title",
    "supplementary_Section-header",
    "supplementary_Text",
    "supplementary_Caption",
}


def write_page_text_assets(
    conn,
    doc_id,
    page_text_map,
    out_dir,
    asset_type,
    source_pdf,
    min_chars=80,
):
    """
    Create one raw_asset per page from accumulated raw_text_block records.

    page_text_map format:
    {
        page_no: [
            {
                "block_id": "...",
                "section": "Text",
                "text": "...",
            }
        ]
    }
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_assets = 0

    for page_no in sorted(page_text_map):
        blocks = page_text_map[page_no]

        parts = []
        block_ids = []
        sections = []

        for b in blocks:
            text = clean(b.get("text"))
            section = clean(b.get("section"))
            block_id = clean(b.get("block_id"))

            if not text:
                continue

            block_ids.append(block_id)
            sections.append(section)
            parts.append(f"[{section}]\n{text}")

        page_text = "\n\n".join(parts).strip()
        if len(page_text) < min_chars:
            continue

        file_path = out_dir / f"page_{int(page_no):03d}_text.md"
        file_path.write_text(page_text, encoding="utf-8")

        asset_id = uid("text_asset", doc_id, asset_type, source_pdf, page_no)

        meta = {
            "source": "raw_text_block_page_aggregation",
            "source_pdf": str(source_pdf),
            "page_no": page_no,
            "block_ids": block_ids,
            "sections": sorted(set(sections)),
            "n_blocks": len(block_ids),
            "n_chars": len(page_text),
            "ingest_policy": "page_level_text_asset",
        }

        insert_asset(
            conn,
            asset_id,
            doc_id,
            asset_type,
            page_no,
            "",
            "",
            file_path,
            meta,
        )

        n_assets += 1

    return n_assets


def run_dotsocr(input_path, out_dir, dots_root, dots_python, num_thread):
    """
    Run dots.ocr once on a PDF or image.

    Important:
    - For PDF files, pass the PDF directly to dots.ocr.
    - Do not render PDF into pages and call dots.ocr page by page.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dots_root = Path(dots_root).expanduser().resolve()
    input_path = Path(input_path).expanduser().resolve()

    before = {p.resolve() for p in out_dir.rglob("*") if p.suffix.lower() in [".json", ".md"]}
    t0 = time.time()

    cmd = [
        dots_python,
        str(dots_root / "dots_ocr" / "parser.py"),
        str(input_path),
        "--num_thread",
        str(num_thread),
        "--port",
        "8001",
        "--model_name",
        "dotsocr",
        "--output",
        str(out_dir.resolve())

    ]

    log_path = out_dir / f"{input_path.stem}.log"

    try:
        with log_path.open("w", encoding="utf-8") as f:
            subprocess.run(cmd, cwd=str(dots_root), stdout=f, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        try:
            tail = log_path.read_text(encoding="utf-8", errors="ignore")[-5000:]
        except Exception:
            tail = ""
        raise RuntimeError(
            f"DotsOCR failed for {input_path}\n"
            f"log_path={log_path}\n"
            f"command={' '.join(map(str, cmd))}\n"
            f"---- log tail ----\n{tail}"
        ) from e

    after = {p.resolve() for p in out_dir.rglob("*") if p.suffix.lower() in [".json", ".md"]}
    new_files = list(after - before)

    json_files = [p for p in new_files if p.suffix.lower() == ".json"]
    md_files = [p for p in new_files if p.suffix.lower() == ".md"]

    if not json_files:
        json_files = [
            p for p in out_dir.rglob("*.json")
            if p.stat().st_mtime >= t0 - 1
        ]

    if not md_files:
        md_files = [
            p for p in out_dir.rglob("*.md")
            if p.stat().st_mtime >= t0 - 1
        ]

    if not json_files:
        raise RuntimeError(f"DotsOCR json output not found for {input_path}; log_path={log_path}")


    return json_files,md_files,log_path



def read_text_safe(path):
    path = Path(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        return ""


def load_json_safe(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
    except Exception:
        return {}


def merge_md_files(md_paths):
    parts = []
    for i, p in enumerate(md_paths, start=1):
        if not p.exists():
            continue
        text = read_text_safe(p)
        if not text.strip():
            continue
        parts.append(
            f"\n\n<!-- DOTSOCR_MD_SOURCE index={i} path={p} -->\n\n{text}"
        )
    return "\n\n".join(parts).strip()


def load_cells_from_json_files(json_paths):
    cells = []

    for j, p in enumerate(json_paths, start=1):
        if not p.exists():
            continue
        page = re.search(r"page_(\d+)", p.name).group(1)
        obj = load_json_safe(p)
        part_cells = flatten_cells_with_page(obj)

        for c in part_cells:
            c["source_json_path"] = str(p)
            c["source_json_index"] = j
            c["page_no"] = int(page)

        cells.extend(part_cells)

    return cells

def get_text(x):
    if not isinstance(x, dict):
        return ""
    vals = []
    for k in ["text", "content", "markdown", "html", "latex"]:
        v = x.get(k)
        if isinstance(v, str) and v.strip():
            vals.append(v.strip())
    return "\n".join(vals)



TEXT_CATEGORIES = {
    "title",
    "section-header",
    "text",
    "caption",
    "footnote",
    "formula",
    "list-item",
}

DROP_TEXT_CATEGORIES = {
    "page-header",
    "page-footer",
    "page-number",
    "page_number",
    "watermark",
}

FIGURE_CATEGORIES = {
    "figure",
    "picture",
}

TABLE_CATEGORIES = {
    "table",
}

def clean(x):
    """
    Convert any value to a normalized single-line string.

    - None -> ""
    - non-string -> str(x)
    - collapse repeated whitespace/newlines/tabs into one space
    - strip leading/trailing spaces
    """
    return re.sub(r"\s+", " ", str(x or "").strip())

def norm_category(cat):
    c = clean(cat).lower()
    c = c.replace(" ", "-")
    c = c.replace("_", "-")
    return c


def is_text_block_category(cat):
    c = norm_category(cat)
    if c in DROP_TEXT_CATEGORIES:
        return False
    return c in TEXT_CATEGORIES


def is_caption_block(cat, text):
    c = norm_category(cat)
    if c == "caption":
        return True
    return bool(re.match(r"^\s*(Fig\.|Figure|Scheme|Table)\s*\d+", clean(text), re.I))


def is_figure_block_category(cat):
    return norm_category(cat) in FIGURE_CATEGORIES


def is_table_block_category(cat):
    return norm_category(cat) in TABLE_CATEGORIES


def bbox_area_ratio(xyxy, width, height):
    if not xyxy or width <= 0 or height <= 0:
        return 0.0
    x0, y0, x1, y1 = xyxy
    return max(0, x1 - x0) * max(0, y1 - y0) / float(width * height)


def valid_crop_bbox(xyxy, width, height, min_area_ratio=0.0005, max_area_ratio=0.85):
    """
    Avoid tiny OCR noise and avoid whole-page crops.
    """
    r = bbox_area_ratio(xyxy, width, height)
    return min_area_ratio <= r <= max_area_ratio


def bbox_xyxy(bbox, w, h):
    if bbox is None:
        return None

    if isinstance(bbox, dict):
        vals = [
            bbox.get("x0", bbox.get("left")),
            bbox.get("y0", bbox.get("top")),
            bbox.get("x1", bbox.get("right")),
            bbox.get("y1", bbox.get("bottom")),
        ]
    elif isinstance(bbox, list):
        if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
            vals = bbox
        elif bbox and isinstance(bbox[0], (list, tuple)):
            xs = [p[0] for p in bbox if len(p) >= 2]
            ys = [p[1] for p in bbox if len(p) >= 2]
            vals = [min(xs), min(ys), max(xs), max(ys)] if xs and ys else None
        else:
            vals = None
    else:
        vals = None

    if not vals or any(v is None for v in vals):
        return None

    x0, y0, x1, y1 = map(float, vals)

    if max(x0, y0, x1, y1) <= 1.5:
        x0, x1 = x0 * w, x1 * w
        y0, y1 = y0 * h, y1 * h

    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(0, min(w, int(x1)))
    y1 = max(0, min(h, int(y1)))

    if x1 <= x0 or y1 <= y0:
        return None

    return [x0, y0, x1, y1]


def crop(img_path, bbox, out_path):
    im = Image.open(img_path).convert("RGB")
    xyxy = bbox_xyxy(bbox, im.width, im.height)
    if not xyxy:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.crop(xyxy).save(out_path)
    return xyxy


def is_caption(cat, text):
    s = f"{cat} {text}".strip()
    return bool(re.match(r"^\s*(Fig\.|Figure|Scheme|Table)\s*\d+", text, re.I)) or "caption" in cat.lower()


def ref_from_caption(text):
    m = re.match(r"^\s*((?:Fig\.|Figure|Scheme|Table)\s*\d+[A-Za-z]?)", text, re.I)
    return m.group(1) if m else ""


def bind_captions(objects, captions):
    for cap in captions:
        ref = cap["ref"].lower()
        target_kind = "table" if ref.startswith("table") else "figure"

        cbox = cap.get("xyxy")
        if not cbox:
            continue

        cx = (cbox[0] + cbox[2]) / 2
        cy = (cbox[1] + cbox[3]) / 2

        best = None
        best_score = 1e18

        for obj in objects:
            if obj["kind"] != target_kind or obj["page_no"] != cap["page_no"]:
                continue
            obox = obj.get("xyxy")
            if not obox:
                continue

            ox = (obox[0] + obox[2]) / 2
            oy = (obox[1] + obox[3]) / 2
            score = abs(cx - ox) * 0.2 + abs(cy - oy)

            if score < best_score:
                best = obj
                best_score = score

        if best:
            best["caption"] = cap["text"]
            cap["bound_object_id"] = best["id"]
            cap["bound_object_ref"] = best["ref"]
            cap["bound_object_type"] = best["kind"]

    return objects, captions



def flatten_cells_with_page(obj):
    """
    Flatten dots.ocr JSON into layout blocks only.

    Keep only blocks that look like real layout elements:
    - must have category/type/class/label
    - must have text or bbox
    """
    cells = []

    def walk(x):
        if isinstance(x, dict):
            cat = (
                x.get("category")
                or x.get("type")
                or x.get("label")
                or x.get("class")
            )
            bbox = x.get("bbox") or x.get("box") or x.get("position")
            text = get_text(x)

            # Only accept likely layout elements.
            if cat and (bbox is not None or text):
                cells.append({
                    "category": str(cat or ""),
                    "bbox": bbox,
                    "text": text,
                    "raw": x,
                })


        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return cells


def render_pdf_pages(pdf_path, out_dir, doc_id, conn, prefix="page", register_asset=False, dpi=200):
    """
    Render PDF pages for downstream cropping only.

    Important:
    - Page images are intermediate files.
    - By default, do NOT insert page images into raw_asset.
    - raw_asset should contain semantic assets from dotsocr JSON:
      figure_image / table_image / supplementary_figure_image / supplementary_table_image.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    pages = []

    for i, page in enumerate(doc, start=0):
        img_path = out_dir / f"{prefix}_{i:03d}.png"
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        pix.save(str(img_path))
        pages.append((i, img_path))

        if register_asset:
            asset_id = uid("asset", doc_id, "page_image", prefix, i)
            insert_asset(
                conn,
                asset_id,
                doc_id,
                "page_image",
                i,
                "",
                "",
                img_path,
                {
                    "source": "pymupdf",
                    "pdf": str(pdf_path),
                    "page_no": i,
                    "note": "debug_or_intermediate_page_image",
                },
            )

    return pages
def parse_pdf_with_dotsocr_once(
    doc_id,
    pdf_path,
    work_dir,
    conn,
    dots_root,
    dots_python,
    num_thread,
    kind="main",
    supp_name="",
):
    """
    Parse one PDF using dotsocr JSON as the only database-ingestion source.

    DB ingestion rule:
    - raw_text_block: only from dotsocr JSON text-like layout blocks
    - raw_figure/raw_table/raw_asset: only from dotsocr JSON figure/table blocks
    - full markdown is saved to disk but NOT inserted into raw_text_block
    - page images are saved to disk for crop only but NOT inserted into raw_asset
    """
    pdf_path = Path(pdf_path)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", pdf_path.stem)

    if kind == "main":
        page_dir = work_dir / "pages"
        dots_dir = work_dir / "dotsocr"
        fig_dir = work_dir / "figures"
        table_dir = work_dir / "tables"
        text_dir = work_dir / "text"

        for d in [page_dir, dots_dir, fig_dir, table_dir, text_dir]:
            d.mkdir(parents=True, exist_ok=True)

        pages = render_pdf_pages(
            pdf_path,
            page_dir,
            doc_id,
            conn,
            prefix="page",
            register_asset=False,
        )
        ocr_input = pdf_path

        text_prefix = "text"
        section_prefix = ""
        fig_asset_type = "figure_image"
        table_asset_type = "table_image"
        fig_id_prefix = "fig"
        table_id_prefix = "table"
        caption_id_prefix = "caption"

    else:
        supp_work_dir = work_dir
        supp_work_dir.mkdir(parents=True, exist_ok=True)

        dst_pdf = supp_work_dir / pdf_path.name
        if pdf_path.resolve() != dst_pdf.resolve():
            shutil.copy2(pdf_path, dst_pdf)

        asset_id = uid("supp_asset", doc_id, pdf_path.name)
        insert_asset(
            conn,
            asset_id,
            doc_id,
            "supplementary_pdf",
            None,
            "",
            "",
            dst_pdf,
            {
                "source_file": str(pdf_path),
                "parsed_by": "dotsocr_json_blocks",
                "note": "original supplementary pdf; semantic blocks are inserted separately",
            },
        )

        page_dir = supp_work_dir / f"{safe}_pages"
        dots_dir = supp_work_dir / f"{safe}_dotsocr"
        fig_dir = supp_work_dir / f"{safe}_figures"
        table_dir = supp_work_dir / f"{safe}_tables"
        text_dir = supp_work_dir / f"{safe}_text"

        for d in [page_dir, dots_dir, fig_dir, table_dir, text_dir]:
            d.mkdir(parents=True, exist_ok=True)

        pages = render_supplement_pdf_pages(
            dst_pdf,
            page_dir,
            doc_id,
            conn,
            pdf_path.name,
            register_asset=False,
        )
        ocr_input = dst_pdf

        text_prefix = "supp_text"
        section_prefix = "supplementary_"
        fig_asset_type = "supplementary_figure_image"
        table_asset_type = "supplementary_table_image"
        fig_id_prefix = "supp_fig"
        table_id_prefix = "supp_table"
        caption_id_prefix = "supp_caption"

    page_img_map = {int(page_no): img_path for page_no, img_path in pages}

    json_paths, md_paths, log_path = run_dotsocr(
        ocr_input,
        dots_dir / "pdf",
        dots_root,
        dots_python,
        num_thread,
    )

    # Keep markdown on disk only. Do not insert whole markdown into raw_text_block.

    md = merge_md_files(md_paths)
    (text_dir / "full_text.md").write_text(md, encoding="utf-8")

    cells = load_cells_from_json_files(json_paths)

    (text_dir / "dotsocr_sources.json").write_text(
        json.dumps(
            {
                "num_md_files": len(md_paths),
                "num_json_files": len(json_paths),
                "md_paths": [str(x) for x in md_paths],
                "json_paths": [str(x) for x in json_paths],
                "merged_md_path": str(text_dir / "full_text.md"),
                "num_cells": len(cells),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    captions = []
    objects = []
    n_text = 0
    n_skipped = 0
    page_text_map = {}

    for i, cell in enumerate(cells):
        page_no = cell.get("page_no")
        if isinstance(page_no, str) and page_no.isdigit():
            page_no = int(page_no)

        cat = clean(cell.get("category"))
        text = clean(cell.get("text"))
        bbox = cell.get("bbox")
        norm_cat = norm_category(cat)

        page_img = page_img_map.get(page_no)

        xyxy = None
        im_w = 0
        im_h = 0

        if page_img and bbox is not None:
            try:
                im = Image.open(page_img)
                im_w, im_h = im.width, im.height
                xyxy = bbox_xyxy(bbox, im_w, im_h)
            except Exception:
                xyxy = None

        meta = {
            "source": "dotsocr_json_block",
            "pdf": str(ocr_input),
            "page_image": str(page_img) if page_img else "",
            "dotsocr_json": cell.get("source_json_path") or "",
            "dotsocr_json_index": cell.get("source_json_index"),
            "all_dotsocr_json": [str(x) for x in json_paths],
            "dotsocr_log": str(log_path) if log_path else "",
            "category": cat,
            "norm_category": norm_cat,
            "bbox": bbox,
            "xyxy": xyxy,
            "page_no_from_dotsocr": page_no,
            "raw_block": cell.get("raw"),
        }

        # 1. Text-like blocks.
        if text and is_text_block_category(cat):
            n_text += 1
            section_name = f"{section_prefix}{cat}"
            block_id = uid(
                text_prefix,
                doc_id,
                pdf_path.name if kind != "main" else "main",
                page_no,
                i,
                cat,
                text[:80],
            )

            insert_text(
                conn,
                block_id,
                doc_id,
                page_no,
                section_name,
                text,
                meta,
            )

            if page_no is not None and section_name in TEXT_ASSET_SECTIONS:
                page_text_map.setdefault(int(page_no), []).append({
                    "block_id": block_id,
                    "section": section_name,
                    "text": text,
                })

            if is_caption_block(cat, text):
                captions.append({
                    "caption_id": uid(caption_id_prefix, doc_id, pdf_path.name if kind != "main" else "main", page_no, i),
                    "page_no": page_no,
                    "ref": ref_from_caption(text),
                    "text": text,
                    "bbox": bbox,
                    "xyxy": xyxy,
                })

            continue

        # 2. Caption text whose category may not be exactly Caption.
        if text and is_caption_block(cat, text):
            n_text += 1
            insert_text(
                conn,
                uid(text_prefix, doc_id, pdf_path.name if kind != "main" else "main", page_no, i, "caption", text[:80]),
                doc_id,
                page_no,
                f"{section_prefix}Caption",
                text,
                meta,
            )
            captions.append({
                "caption_id": uid(caption_id_prefix, doc_id, pdf_path.name if kind != "main" else "main", page_no, i),
                "page_no": page_no,
                "ref": ref_from_caption(text),
                "text": text,
                "bbox": bbox,
                "xyxy": xyxy,
            })
            continue

        # 3. Figure/table blocks need image bbox.
        if not page_img or bbox is None or not xyxy:
            n_skipped += 1
            continue

        if not valid_crop_bbox(xyxy, im_w, im_h):
            n_skipped += 1
            continue

        if is_table_block_category(cat):
            count = len([
                x for x in objects
                if x["kind"] == "table" and x["page_no"] == page_no
            ]) + 1

            table_ref = (
                f"{safe}_page{int(page_no or 0):03d}_table{count:03d}"
                if kind != "main"
                else f"page{int(page_no or 0):03d}_table{count:03d}"
            )

            img_out = table_dir / f"{table_ref}.png"
            xyxy2 = crop(page_img, bbox, img_out)

            if xyxy2:
                table_id = uid(table_id_prefix, doc_id, pdf_path.name if kind != "main" else "main", page_no, i)
                json_out = table_dir / f"{table_ref}.json"
                json_out.write_text(json.dumps(cell["raw"], ensure_ascii=False, indent=2), encoding="utf-8")

                objects.append({
                    "kind": "table",
                    "id": table_id,
                    "ref": table_ref,
                    "page_no": page_no,
                    "file_path": img_out,
                    "json_path": json_out,
                    "text": text,
                    "bbox": bbox,
                    "xyxy": xyxy2,
                    "caption": "",
                    "meta": meta,
                })

        elif is_figure_block_category(cat):
            count = len([
                x for x in objects
                if x["kind"] == "figure" and x["page_no"] == page_no
            ]) + 1

            fig_ref = (
                f"{safe}_page{int(page_no or 0):03d}_figure{count:03d}"
                if kind != "main"
                else f"page{int(page_no or 0):03d}_figure{count:03d}"
            )

            img_out = fig_dir / f"{fig_ref}.png"
            xyxy2 = crop(page_img, bbox, img_out)

            if xyxy2:
                fig_id = uid(fig_id_prefix, doc_id, pdf_path.name if kind != "main" else "main", page_no, i)
                json_out = fig_dir / f"{fig_ref}.json"
                json_out.write_text(json.dumps(cell["raw"], ensure_ascii=False, indent=2), encoding="utf-8")

                objects.append({
                    "kind": "figure",
                    "id": fig_id,
                    "ref": fig_ref,
                    "page_no": page_no,
                    "file_path": img_out,
                    "json_path": json_out,
                    "text": text,
                    "bbox": bbox,
                    "xyxy": xyxy2,
                    "caption": "",
                    "meta": meta,
                })

        else:
            n_skipped += 1

    objects, captions = bind_captions(objects, captions)

    for obj in objects:
        meta = dict(obj["meta"])
        meta.update({
            "bbox": obj["bbox"],
            "xyxy": obj["xyxy"],
            "json_path": str(obj["json_path"]),
            "caption": obj["caption"],
            "text": obj["text"],
            "ingest_policy": "dotsocr_json_semantic_block_only",
        })

        if obj["kind"] == "figure":
            insert_figure(
                conn,
                obj["id"],
                doc_id,
                obj["page_no"],
                obj["ref"],
                obj["file_path"],
                obj["caption"],
                meta,
            )
            insert_asset(
                conn,
                obj["id"],
                doc_id,
                fig_asset_type,
                obj["page_no"],
                obj["ref"],
                "",
                obj["file_path"],
                meta,
            )

        elif obj["kind"] == "table":
            insert_table(
                conn,
                obj["id"],
                doc_id,
                obj["page_no"],
                obj["ref"],
                obj["file_path"],
                {
                    "source": "dotsocr_json_block",
                    "table_ref": obj["ref"],
                    "text": obj["text"],
                    "raw_block_path": str(obj["json_path"]),
                    "metadata": meta,
                },
            )
            insert_asset(
                conn,
                obj["id"],
                doc_id,
                table_asset_type,
                obj["page_no"],
                "",
                obj["ref"],
                obj["file_path"],
                meta,
            )

    (text_dir / "captions.json").write_text(
        json.dumps(captions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if kind == "main":
        text_asset_type = "text_page"
    else:
        text_asset_type = "supplementary_text_page"

    n_text_assets = write_page_text_assets(
        conn=conn,
        doc_id=doc_id,
        page_text_map=page_text_map,
        out_dir=text_dir / "text_pages",
        asset_type=text_asset_type,
        source_pdf=ocr_input,
        min_chars=80,
    )

    return {
        "pages": len(pages),
        "text_blocks": n_text,
        "figures": len([x for x in objects if x["kind"] == "figure"]),
        "tables": len([x for x in objects if x["kind"] == "table"]),
        "captions": len(captions),
        "skipped_blocks": n_skipped,
        "ingest_policy": "dotsocr_json_semantic_block_only",
    }


def parse_main_pdf(doc_id, pdf_path, work_dir, conn, dots_root, dots_python, num_thread):
    return parse_pdf_with_dotsocr_once(
        doc_id=doc_id,
        pdf_path=pdf_path,
        work_dir=work_dir,
        conn=conn,
        dots_root=dots_root,
        dots_python=dots_python,
        num_thread=num_thread,
        kind="main",
    )

def parse_supplementary_pdf(doc_id, pdf_path, supp_work_dir, conn, dots_root, dots_python, num_thread):
    return parse_pdf_with_dotsocr_once(
        doc_id=doc_id,
        pdf_path=pdf_path,
        work_dir=supp_work_dir,
        conn=conn,
        dots_root=dots_root,
        dots_python=dots_python,
        num_thread=num_thread,
        kind="supplementary",
        supp_name=Path(pdf_path).name,
    )

# should be same as dotsocr
def render_supplement_pdf_pages(pdf_path, out_dir, doc_id, conn, supp_name, register_asset=False,dpi=200):
    """
    Render supplementary PDF pages for downstream cropping only.

    By default, supplementary page images are not inserted into raw_asset.
    Only semantic figure/table crops from dotsocr JSON should be inserted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    pages = []
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(supp_name).stem)

    for i, page in enumerate(doc, start=0):
        img_path = out_dir / f"{safe}_page_{i:03d}.png"
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        pix.save(str(img_path))
        pages.append((i, img_path))

        if register_asset:
            asset_id = uid("supp_page", doc_id, supp_name, i)
            insert_asset(
                conn,
                asset_id,
                doc_id,
                "supplementary_page_image",
                i,
                "",
                "",
                img_path,
                {
                    "source": "pymupdf",
                    "supplementary_file": str(pdf_path),
                    "supplementary_name": supp_name,
                    "page_no": i,
                    "note": "debug_or_intermediate_page_image",
                },
            )

    return pages

def parse_supplementary(doc_id, supp_dir, work_dir, conn, dots_root=None, dots_python=None, num_thread=8, parse_supp_pdf=True):
    if not supp_dir:
        return 0

    supp = Path(supp_dir)
    if not supp.exists():
        return 0

    out = work_dir / "supplementary"
    out.mkdir(parents=True, exist_ok=True)

    n = 0

    for p in supp.rglob("*"):
        if not p.is_file():
            continue

        suffix = p.suffix.lower()

        if suffix in [".xlsx", ".xls"]:
            sheets = pd.read_excel(p, sheet_name=None)
            for sheet, df in sheets.items():
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sheet))
                csv_path = out / f"{p.stem}_{safe}.csv"
                df.to_csv(csv_path, index=False)

                table_id = uid("supp_table", doc_id, p.name, sheet)
                meta = {
                    "source": "supplementary_excel",
                    "source_file": str(p),
                    "sheet_name": sheet,
                    "n_rows": int(df.shape[0]),
                    "n_cols": int(df.shape[1]),
                }
                insert_table(conn, table_id, doc_id, None, f"{p.name}:{sheet}", csv_path, meta)
                insert_asset(conn, table_id, doc_id, "supplementary_table", None, "", f"{p.name}:{sheet}", csv_path, meta)
                n += 1

        elif suffix == ".csv":
            df = pd.read_csv(p)
            csv_path = out / p.name
            df.to_csv(csv_path, index=False)

            table_id = uid("supp_table", doc_id, p.name)
            meta = {
                "source": "supplementary_csv",
                "source_file": str(p),
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
            }
            insert_table(conn, table_id, doc_id, None, p.name, csv_path, meta)
            insert_asset(conn, table_id, doc_id, "supplementary_table", None, "", p.name, csv_path, meta)
            n += 1

        elif suffix == ".docx":
            text = "\n".join(x.text for x in Document(p).paragraphs)
            block_id = uid("supp_text", doc_id, p.name)
            insert_text(conn, block_id, doc_id, None, "supplementary_docx", text, {"source_file": str(p)})
            n += 1

        elif suffix in [".txt", ".md"]:
            text = p.read_text(encoding="utf-8", errors="ignore")
            block_id = uid("supp_text", doc_id, p.name)
            insert_text(conn, block_id, doc_id, None, f"supplementary_{suffix[1:]}", text, {"source_file": str(p)})
            n += 1

        elif suffix == ".pdf":
            if parse_supp_pdf and dots_root and dots_python:
                stat = parse_supplementary_pdf(doc_id, p, out, conn, dots_root, dots_python, num_thread)
                n += 1 + int(stat.get("figures", 0)) + int(stat.get("tables", 0))
            else:
                asset_id = uid("supp_asset", doc_id, p.name)
                dst = out / p.name
                shutil.copy2(p, dst)
                insert_asset(conn, asset_id, doc_id, "supplementary_pdf", None, "", "", dst, {"source_file": str(p), "parsed": False})
                n += 1

        elif suffix in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            asset_id = uid("supp_asset", doc_id, p.name)
            dst = out / p.name
            shutil.copy2(p, dst)
            insert_asset(conn, asset_id, doc_id, "supplementary_file", None, "", "", dst, {"source_file": str(p)})
            n += 1

    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--dots-root", default="/root/autodl-tmp/dots.ocr")
    ap.add_argument("--dots-python", default=sys.executable)
    ap.add_argument("--num-thread", type=int, default=16)
    ap.add_argument("--skip-supp-pdf-ocr", action="store_true", help="Register supplementary PDFs without OCR parsing.")
    args = ap.parse_args()

    conn = get_conn()
    row = conn.execute(
        "SELECT doc_id, source_pdf_path, supplement_dir FROM raw_document WHERE doc_id=?",
        (args.doc_id,),
    ).fetchone()

    if not row:
        raise SystemExit(f"doc_id not found: {args.doc_id}")

    pdf_path = Path(row["source_pdf_path"])
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    work_dir = Path("data/work") / args.doc_id
    work_dir.mkdir(parents=True, exist_ok=True)

    stat = parse_main_pdf(
        args.doc_id,
        pdf_path,
        work_dir,
        conn,
        args.dots_root,
        args.dots_python,
        args.num_thread,
    )

    n_supp = parse_supplementary(
        args.doc_id,
        row["supplement_dir"],
        work_dir,
        conn,
        args.dots_root,
        args.dots_python,
        args.num_thread,
        parse_supp_pdf=not args.skip_supp_pdf_ocr,
    )

    conn.execute("UPDATE raw_document SET status=? WHERE doc_id=?", ("parsed_dotsocr", args.doc_id))
    conn.commit()
    conn.close()

    print(json.dumps({
        "doc_id": args.doc_id,
        "status": "parsed_dotsocr",
        "work_dir": str(work_dir),
        "supplementary_items": n_supp,
        **stat,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
