import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_FILE = "pixeldwell.db"

def get_connection():
    return sqlite3.connect(DB_FILE)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Jobs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            created_at TEXT
        )
    ''')
    
    # Clusters table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clusters (
            id TEXT PRIMARY KEY,
            job_id TEXT,
            cluster_name TEXT,
            tags TEXT,
            FOREIGN KEY (job_id) REFERENCES jobs (id)
        )
    ''')
    
    # Images table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            cluster_id TEXT,
            filename TEXT,
            original_path TEXT,
            labeled_path TEXT,
            annotations TEXT,
            FOREIGN KEY (cluster_id) REFERENCES clusters (id)
        )
    ''')
    
    # Enhancements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS enhancements (
            id TEXT PRIMARY KEY,
            cluster_id TEXT,
            feature_name TEXT,
            result_image_path TEXT,
            status TEXT,
            FOREIGN KEY (cluster_id) REFERENCES clusters (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def create_job(job_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO jobs (id, created_at) VALUES (?, ?)", (job_id, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def insert_cluster(cluster_id: str, job_id: str, cluster_name: str, tags: list):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO clusters (id, job_id, cluster_name, tags) VALUES (?, ?, ?, ?)",
        (cluster_id, job_id, cluster_name, json.dumps(tags))
    )
    conn.commit()
    conn.close()

def insert_image(image_id: str, cluster_id: str, filename: str, original_path: str, labeled_path: str, annotations: list):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO images (id, cluster_id, filename, original_path, labeled_path, annotations) VALUES (?, ?, ?, ?, ?, ?)",
        (image_id, cluster_id, filename, original_path, labeled_path, json.dumps(annotations))
    )
    conn.commit()
    conn.close()

def insert_enhancement(enhancement_id: str, cluster_id: str, feature_name: str, result_image_path: str, status: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO enhancements (id, cluster_id, feature_name, result_image_path, status) VALUES (?, ?, ?, ?, ?)",
        (enhancement_id, cluster_id, feature_name, result_image_path, status)
    )
    conn.commit()
    conn.close()

def get_latest_job_clusters():
    """Fetches the most recent job and reconstructs the cluster data payload."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get latest job
    cursor.execute("SELECT id FROM jobs ORDER BY created_at DESC LIMIT 1")
    job_row = cursor.fetchone()
    if not job_row:
        conn.close()
        return None
    
    job_id = job_row["id"]
    
    # Get clusters for this job
    cursor.execute("SELECT id, cluster_name, tags FROM clusters WHERE job_id = ?", (job_id,))
    clusters_rows = cursor.fetchall()
    
    response_clusters = []
    
    for c_row in clusters_rows:
        cluster_id = c_row["id"]
        cluster_name = c_row["cluster_name"]
        tags = json.loads(c_row["tags"]) if c_row["tags"] else []
        
        # Get images for this cluster
        cursor.execute("SELECT original_path, labeled_path, filename FROM images WHERE cluster_id = ?", (cluster_id,))
        images_rows = cursor.fetchall()
        
        thumbnails = []
        for img in images_rows:
            thumbnails.append({
                "original": img["original_path"],
                "labeled": img["labeled_path"] if img["labeled_path"] else None,
                "filename": img["filename"]
            })
            
        if not thumbnails:
            continue
            
        response_clusters.append({
            "id": cluster_name,
            "job_id": job_id,
            "count": len(thumbnails),
            "thumbnails": thumbnails,
            "tags": tags
        })
        
    conn.close()
    return response_clusters

def get_cluster_details(job_id: str, cluster_name: str):
    """Fetches a specific cluster and its images."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cluster_id = f"{job_id}_{cluster_name}"
    
    cursor.execute("SELECT tags FROM clusters WHERE id = ?", (cluster_id,))
    c_row = cursor.fetchone()
    
    if not c_row:
        conn.close()
        return None
        
    tags = json.loads(c_row["tags"]) if c_row["tags"] else []
    
    cursor.execute("SELECT original_path, labeled_path, filename, annotations FROM images WHERE cluster_id = ?", (cluster_id,))
    images_rows = cursor.fetchall()
    
    thumbnails = []
    for img in images_rows:
        thumbnails.append({
            "original": img["original_path"],
            "labeled": img["labeled_path"] if img["labeled_path"] else None,
            "filename": img["filename"],
            "annotations": json.loads(img["annotations"]) if img["annotations"] else []
        })
        
    conn.close()
    
    return {
        "id": cluster_name,
        "job_id": job_id,
        "count": len(thumbnails),
        "thumbnails": thumbnails,
        "tags": tags
    }

# Initialize DB when the module is imported
init_db()
