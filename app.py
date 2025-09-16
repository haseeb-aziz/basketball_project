# Project layout (all files shown below)
#
# courtcoach/
# ├─ app.py
# ├─ services.py
# ├─ api_key.txt                # put your Roboflow API key here (single line)
# ├─ uploads/                   # auto-created
# ├─ outputs/                   # auto-created
# ├─ clips/                     # auto-created
# ├─ static/
# │   ├─ style.css
# │   └─ logo.svg
# └─ templates/
#     └─ index.html
#
# ============================================================================
# app.py
# ============================================================================

from __future__ import annotations
import os
import uuid
import threading
import time
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from services import process_video, Settings, read_api_key

# -------- Flask setup --------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
API_KEY_FILE = os.path.join(BASE_DIR, "api_key.txt")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

# Allowed extensions
ALLOWED_EXTS = {"mp4", "mov", "avi", "mkv"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

# ---------------- Progress + jobs ----------------
# Simple in-memory job store. In production, use Redis or a DB.
JOBS = {}
# JOBS[job_id] = {
#   'status': 'queued'|'running'|'done'|'error',
#   'progress': 0.0,                # 0..100
#   'eta_seconds': None or float,
#   'started_at': timestamp,
#   'finished_at': timestamp or None,
#   'results': dict or None,
#   'error': str or None,
# }

def _progress_cb_factory(job_id: str):
    def cb(done_frames: int, total_frames: int, elapsed_s: float):
        if total_frames > 0:
            pct = max(0.0, min(100.0, (done_frames / total_frames) * 100.0))
        else:
            pct = 0.0
        # naive ETA
        rate = (done_frames / elapsed_s) if elapsed_s > 0 else 0
        remaining = (total_frames - done_frames)
        eta = (remaining / rate) if rate > 0 else None
        job = JOBS.get(job_id)
        if job and job['status'] == 'running':
            job['progress'] = pct
            job['eta_seconds'] = eta
    return cb

@app.route("/")
def home():
    api_key = read_api_key(API_KEY_FILE)
    return render_template("index.html", api_key_present=bool(api_key), results=None, job_id=None)

@app.route("/job/<job_id>")
def job_view(job_id: str):
    api_key = read_api_key(API_KEY_FILE)
    job = JOBS.get(job_id)
    results = job.get('results') if job else None
    return render_template("index.html", api_key_present=bool(api_key), results=results, job_id=job_id)

@app.route("/progress/<job_id>")
def job_progress(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({
        "status": job["status"],
        "progress": round(job.get("progress", 0.0), 2),
        "eta_seconds": job.get("eta_seconds"),
        "results_ready": job["status"] == "done",
        "error": job.get("error"),
    })

@app.route("/upload", methods=["POST"]) 
def upload():
    # Validate API key first
    api_key = read_api_key(API_KEY_FILE)
    if not api_key:
        flash("Missing api_key.txt. Put your Roboflow API key (single line) in courtcoach/api_key.txt.")
        return redirect(url_for("home"))

    if "video" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("home"))

    file = request.files["video"]
    if file.filename == "":
        flash("No selected file.")
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Upload mp4/mov/avi/mkv.")
        return redirect(url_for("home"))

    fname = secure_filename(file.filename)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"{ts}_{fname}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    file.save(save_path)

    # Build settings
    max_dim = int(request.form.get("max_dim", 0) or 0)
    every_n = int(request.form.get("every_n", 1) or 1)
    context_sec = float(request.form.get("context_sec", 3.0) or 3.0)
    save_annotated = request.form.get("use_raw") != "on"

    settings = Settings(
        api_key=api_key,
        output_path=os.path.join(OUTPUT_DIR, f"annotated_{ts}.mp4"),
        clips_dir=CLIPS_DIR,
        max_dim=max_dim,
        every_n=every_n,
        context_sec=context_sec,
        save_annotated=save_annotated,
    )

    # Create job and kick off background thread
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        'status': 'queued', 'progress': 0.0, 'eta_seconds': None,
        'started_at': time.time(), 'finished_at': None,
        'results': None, 'error': None,
    }

    def _runner():
        JOBS[job_id]['status'] = 'running'
        try:
            results = process_video(save_path, settings, progress_callback=_progress_cb_factory(job_id))
            JOBS[job_id]['results'] = results
            JOBS[job_id]['status'] = 'done'
            JOBS[job_id]['progress'] = 100.0
            JOBS[job_id]['finished_at'] = time.time()
        except Exception as e:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = str(e)
            JOBS[job_id]['finished_at'] = time.time()

    threading.Thread(target=_runner, daemon=True).start()

    # Redirect to job page that will poll for progress
    return redirect(url_for('job_view', job_id=job_id))

@app.route("/download/<path:subdir>/<path:filename>")
def download(subdir: str, filename: str):
    safe_root = OUTPUT_DIR if subdir == "outputs" else CLIPS_DIR
    return send_from_directory(safe_root, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
