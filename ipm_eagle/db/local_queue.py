import json
import uuid
from ipm_eagle.db.sqlite import get_conn

def enqueue(doc_id, queue_name, task_type, payload):
    task_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO task_queue (
            task_id, doc_id, queue_name, task_type, payload_json, status
        )
        VALUES (?, ?, ?, ?, ?, 'pending')
        """,
        (task_id, doc_id, queue_name, task_type, json.dumps(payload, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()
    return task_id

def list_pending(queue_name=None):
    conn = get_conn()
    if queue_name:
        rows = conn.execute(
            "SELECT * FROM task_queue WHERE status='pending' AND queue_name=? ORDER BY created_at",
            (queue_name,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM task_queue WHERE status='pending' ORDER BY created_at"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
