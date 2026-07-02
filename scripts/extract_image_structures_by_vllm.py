#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模块: extract_image_structures.py (CoT 视觉定位增强版)
功能: 强迫 VLM 先分析数量、定位分子位置，再提取 SMILES，以消除“幻觉”。
"""

import os
import sys
import json
import uuid
import base64
import argparse
import re
from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from openai import OpenAI
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ipm_eagle.db.sqlite import get_conn

DEFAULT_STRUCTURE_TASK_TYPES = [
    "structure_figure",
    "sar_table",
    "supplementary_structure_table",
]

def clean(x: Any) -> str:
    return " ".join(str(x or "").split())

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
        status TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        source_tool TEXT,
        raw_output TEXT,
        molecule_label TEXT,
        bbox_json TEXT,
        raw_context_json TEXT
    )
    """)
    conn.commit()

def load_structure_tasks(conn, doc_id: str) -> list:
    marks = ",".join(["?"] * len(DEFAULT_STRUCTURE_TASK_TYPES))
    params = [doc_id] + DEFAULT_STRUCTURE_TASK_TYPES

    rows = conn.execute(f"""
    SELECT p.task_id, p.doc_id, p.asset_id, p.task_type, a.file_path, a.page_no
    FROM planned_tasks p
    JOIN raw_asset a ON p.asset_id = a.asset_id
    WHERE p.doc_id=? AND p.task_type IN ({marks})
    """, params).fetchall()
    
    return [dict(r) for r in rows if r['file_path'] and os.path.exists(r['file_path'])]

def encode_image_to_base64(image_path: str) -> str:
    valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
    if not image_path or not os.path.exists(image_path):
        return ""
    if Path(image_path).suffix.lower() not in valid_exts:
        return ""
        
    try:
        with Image.open(image_path) as img:
            if img.mode not in ('RGB', 'RGBA', 'L'):
                img = img.convert('RGBA')
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"⚠️ 跳过无效图片 [{Path(image_path).name}]: {e}")
        return ""

def build_ocsr_prompt() -> str:
    """🌟 核心改进：引入思维链(CoT)和视觉定位，防止大模型产生幻觉"""
    return """
You are an expert in Cheminformatics and Optical Chemical Structure Recognition (OCSR).
Analyze the provided image carefully. To avoid hallucinations, you MUST follow this strict 3-step process:

Step 1 (Analyze): Scan the entire image. Differentiate between actual 2D chemical structures, plain text, and table borders. Count exactly how many distinct chemical molecules are drawn.
Step 2 (Locate): For each molecule, identify its specific location in the image (e.g., "top-left", "center", "row 1 column 2").
Step 3 (Extract): Extract the explicit label/name next to it (if any) and convert its 2D drawing into a valid SMILES string.

Output your final response STRICTLY as a single JSON object. Do NOT wrap it in markdown. 

Expected JSON Schema:
{
  "total_molecules_found": 2,
  "molecules": [
    {
      "location": "top-left quadrant",
      "molecule_label": "Compound 1",
      "smiles": "CC1=CC=C(C=C1)C"
    },
    {
      "location": "bottom-right below the Western blot",
      "molecule_label": "MZ1",
      "smiles": "..."
    }
  ]
}
"""

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """鲁棒的 JSON 解析器，防止大模型输出 markdown 标记"""
    text = text.strip()
    # 尝试去掉可能存在的 ```json ... ```
    match = re.search(r'
http://googleusercontent.com/immersive_entry_chip/0
你可以看看这次大模型数出来的数量是不是符合这**唯一一张图**的实际情况了！
