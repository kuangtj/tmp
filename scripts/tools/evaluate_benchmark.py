#!/usr/bin/env python3
import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas is required; run this script with the ipm_eagle conda environment") from exc


DEFAULT_BENCHMARK_DIR = Path('/root/autodl-tmp/ipm_eagle/benchmark')
DEFAULT_DB_PATH = Path('/root/autodl-tmp/ipm_eagle/data/db/ipm_eagle.sqlite')


def clean(value: Any) -> str:
    if value is None:
        return ''
    text = str(value).strip()
    if text.lower() == 'nan':
        return ''
    return text


def norm(value: Any) -> str:
    return clean(value).lower()


def load_truth_rows(truth_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(truth_dir.glob('*.xlsx')):
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            if df.empty:
                continue
            doi = ''
            if 'Article DOI' in df.columns:
                doi = clean(df['Article DOI'].dropna().iloc[0]) if df['Article DOI'].dropna().shape[0] else ''
            doi = doi or path.stem.replace('_', '/', 1).replace('_', '/')
            rows.append({
                'doi': doi,
                'path': str(path),
                'sheet': sheet,
                'row_count': int(len(df)),
                'unique_targets': sorted({clean(x) for x in df.get('Target', []) if clean(x)}),
                'unique_e3': sorted({clean(x) for x in df.get('E3 ligase', []) if clean(x)}),
                'unique_names': sorted({clean(x) for x in df.get('Name', []) if clean(x)}),
                'compound_ids': sorted({clean(x) for x in df.get('Compound ID', []) if clean(x)}),
            })
    return rows


def find_doc(conn: sqlite3.Connection, doi: str) -> Dict[str, Any]:
    exact = conn.execute(
        'SELECT * FROM raw_document WHERE lower(doi)=lower(?) ORDER BY updated_at DESC LIMIT 1',
        (doi,),
    ).fetchone()
    if exact:
        return {'match_type': 'exact', 'row': exact}

    candidates = conn.execute(
        '''
        SELECT *
        FROM raw_document
        WHERE (? LIKE lower(doi) || '%') OR (lower(doi) LIKE lower(?) || '%')
        ORDER BY updated_at DESC
        ''',
        (doi.lower(), doi.lower()),
    ).fetchall()
    if candidates:
        return {'match_type': 'prefix', 'row': candidates[0], 'candidates': candidates}
    return {'match_type': 'missing', 'row': None, 'candidates': []}


def load_doc_metrics(conn: sqlite3.Connection, doc_id: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        'n_rel': 0,
        'n_assay': 0,
        'n_participant': 0,
        'n_agent': 0,
        'targets': [],
        'effectors': [],
        'inducers': [],
    }
    metrics['n_rel'] = conn.execute('SELECT COUNT(*) FROM stg_relation WHERE doc_id=?', (doc_id,)).fetchone()[0]
    metrics['n_assay'] = conn.execute('SELECT COUNT(*) FROM stg_assay WHERE doc_id=?', (doc_id,)).fetchone()[0]
    metrics['n_participant'] = conn.execute('SELECT COUNT(*) FROM stg_relation_participant WHERE doc_id=?', (doc_id,)).fetchone()[0]
    metrics['n_agent'] = conn.execute('SELECT COUNT(*) FROM stg_agent WHERE doc_id=?', (doc_id,)).fetchone()[0]
    participants = conn.execute(
        'SELECT entity_name, role FROM stg_relation_participant WHERE doc_id=?',
        (doc_id,),
    ).fetchall()
    by_role = {'target': set(), 'effector': set(), 'inducer': set()}
    for row in participants:
        role = clean(row['role'])
        name = clean(row['entity_name'])
        if role in by_role and name:
            by_role[role].add(name)
    metrics['targets'] = sorted(by_role['target'])
    metrics['effectors'] = sorted(by_role['effector'])
    metrics['inducers'] = sorted(by_role['inducer'])
    return metrics


def overlap(a: List[str], b: List[str]) -> str:
    aset = {norm(x) for x in a if clean(x)}
    bset = {norm(x) for x in b if clean(x)}
    if not aset:
        return ''
    return f"{len(aset & bset)}/{len(aset)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark-dir', default=str(DEFAULT_BENCHMARK_DIR))
    ap.add_argument('--db-path', default=str(DEFAULT_DB_PATH))
    ap.add_argument('--out-dir', default='')
    args = ap.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    truth_dir = benchmark_dir / 'groud_true_protacdb'
    out_dir = Path(args.out_dir) if args.out_dir else benchmark_dir / 'eval_outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row

    summaries = []
    for truth in load_truth_rows(truth_dir):
        doc_info = find_doc(conn, truth['doi'])
        row = doc_info['row']
        summary: Dict[str, Any] = {
            'benchmark_doi': truth['doi'],
            'truth_path': truth['path'],
            'truth_sheet': truth['sheet'],
            'truth_rows': truth['row_count'],
            'truth_targets': truth['unique_targets'],
            'truth_e3': truth['unique_e3'],
            'truth_named_compounds': truth['unique_names'],
            'doc_match_type': doc_info['match_type'],
            'doc_id': '',
            'stored_doi': '',
            'status': 'missing',
            'parse_ok': None,
            'n_rel': 0,
            'n_assay': 0,
            'n_participant': 0,
            'n_agent': 0,
            'target_overlap': '',
            'e3_overlap': '',
            'extracted_targets': [],
            'extracted_effectors': [],
            'extracted_inducers': [],
        }
        if row:
            summary['doc_id'] = clean(row['doc_id'])
            summary['stored_doi'] = clean(row['doi'])
            summary['status'] = clean(row['status'])
            meta = json.loads(row['metadata_json'] or '{}') if clean(row['metadata_json']) else {}
            parse_validation = meta.get('parse_validation') or {}
            summary['parse_ok'] = parse_validation.get('ok')
            metrics = load_doc_metrics(conn, summary['doc_id'])
            summary.update(metrics)
            summary['target_overlap'] = overlap(truth['unique_targets'], metrics['targets'])
            summary['e3_overlap'] = overlap(truth['unique_e3'], metrics['effectors'])
        summaries.append(summary)

    conn.close()

    json_path = out_dir / 'benchmark_summary.json'
    csv_path = out_dir / 'benchmark_summary.csv'
    json_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame([
        {
            'benchmark_doi': x['benchmark_doi'],
            'doc_match_type': x['doc_match_type'],
            'doc_id': x['doc_id'],
            'stored_doi': x['stored_doi'],
            'status': x['status'],
            'truth_rows': x['truth_rows'],
            'n_rel': x['n_rel'],
            'n_assay': x['n_assay'],
            'n_agent': x['n_agent'],
            'target_overlap': x['target_overlap'],
            'e3_overlap': x['e3_overlap'],
            'parse_ok': x['parse_ok'],
        }
        for x in summaries
    ]).to_csv(csv_path, index=False)

    for item in summaries:
        print(json.dumps({
            'benchmark_doi': item['benchmark_doi'],
            'doc_match_type': item['doc_match_type'],
            'doc_id': item['doc_id'],
            'stored_doi': item['stored_doi'],
            'status': item['status'],
            'truth_rows': item['truth_rows'],
            'n_rel': item['n_rel'],
            'n_assay': item['n_assay'],
            'target_overlap': item['target_overlap'],
            'e3_overlap': item['e3_overlap'],
        }, ensure_ascii=False))

    print(json.dumps({
        'summary_json': str(json_path),
        'summary_csv': str(csv_path),
        'num_benchmark_entries': len(summaries),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
