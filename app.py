import os
import io
import threading
import sqlite3
import datetime
import json
import shutil
import hashlib
from flask import Flask, render_template, request, jsonify, send_file, abort, session, redirect, url_for
#i am pandu
from model import train_model_background, extract_embedding_for_image, extract_all_embeddings, MODEL_PATH

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "attendance.db")
DATASET_DIR = os.path.join(APP_DIR, "dataset")
UNKNOWN_DIR = os.path.join(APP_DIR, "unknown_faces")
STAFF_DATASET_DIR = os.path.join(APP_DIR, "staff_dataset")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(STAFF_DATASET_DIR, exist_ok=True)

TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "face_attendance_secret_2024"

# =============================================================================
# DB helpers
# =============================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    roll TEXT,
                    class TEXT,
                    section TEXT,
                    created_at TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    name TEXT,
                    timestamp TEXT,
                    status TEXT DEFAULT 'present'
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT,
                    details TEXT,
                    timestamp TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'staff',
                    password_hash TEXT NOT NULL,
                    face_enrolled INTEGER DEFAULT 0,
                    created_at TEXT,
                    is_active INTEGER DEFAULT 1
                )""")
    # Seed default admin account
    _default_pw = hashlib.sha256("admin123".encode()).hexdigest()
    _now = datetime.datetime.now().isoformat()
    c.execute("""INSERT OR IGNORE INTO users
                 (username, full_name, role, password_hash, face_enrolled, created_at, is_active)
                 VALUES (?, ?, ?, ?, 0, ?, 1)""",
              ("admin", "System Administrator", "admin", _default_pw, _now))
    # Default settings
    defaults = {
        "late_cutoff_time": "09:15",
        "low_attendance_threshold": "75",
        "recognition_confidence": "0.60",
        "institution_name": "FaceTrack AI",
        "auto_retrain": "1"
    }
    for k, v in defaults.items():
        c.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
    # Add status column if it doesn't exist (migration for existing DBs)
    try:
        c.execute("ALTER TABLE attendance ADD COLUMN status TEXT DEFAULT 'present'")
    except Exception:
        pass
    conn.commit()
    conn.close()

init_db()

def get_setting(key, default=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else default

def set_setting(key, value):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def log_audit(action, details=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.datetime.now().isoformat()
    c.execute("INSERT INTO audit_log (action, details, timestamp) VALUES (?, ?, ?)", (action, details, ts))
    conn.commit()
    conn.close()

# =============================================================================
# Train status helpers
# =============================================================================
def write_train_status(status_dict):
    with open(TRAIN_STATUS_FILE, "w") as f:
        json.dump(status_dict, f)

def read_train_status():
    if not os.path.exists(TRAIN_STATUS_FILE):
        return {"running": False, "progress": 0, "message": "Not trained"}
    with open(TRAIN_STATUS_FILE, "r") as f:
        return json.load(f)

try:
    _initial_status = read_train_status()
except Exception:
    _initial_status = {}
if _initial_status.get("running"):
    write_train_status({"running": False, "progress": 0, "message": "Training interrupted by server restart."})

# =============================================================================
# Auth helpers
# =============================================================================
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def get_current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return {"id": uid, "role": session.get("user_role", ""),
            "name": session.get("user_name", ""), "username": session.get("user_username", "")}

def require_admin():
    u = get_current_user()
    if not u or u["role"] != "admin":
        session.clear()
        return redirect(url_for("login"))
    return None

def require_staff_or_admin():
    if not session.get("user_id"):
        return redirect(url_for("login"))
    return None

# backward compat alias used by existing routes
require_login = require_staff_or_admin

@app.context_processor
def inject_user():
    return {"current_user": get_current_user()}

# =============================================================================
# Auth Routes
# =============================================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if session.get("user_id"):
            u = get_current_user()
            return redirect(url_for("index") if u["role"] == "admin" else url_for("mark_attendance_page"))
        return render_template("login.html")
    # POST — AJAX
    username = request.form.get("username", "").strip().lower()
    password = request.form.get("password", "").strip()
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,username,full_name,role,password_hash,face_enrolled,is_active FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    if not user or hash_password(password) != user[4]:
        return jsonify({"error": "Invalid username or password"}), 401
    if not user[6]:
        return jsonify({"error": "Account deactivated. Contact admin."}), 403
    uid, uname, fname, role, _, face_enrolled, _ = user
    if role == "admin":
        session.clear()
        session["user_id"] = uid; session["user_username"] = uname
        session["user_name"] = fname; session["user_role"] = role
        log_audit("login", f"Admin '{uname}' logged in")
        return jsonify({"success": True, "role": "admin", "redirect": url_for("index")})
    else:
        if face_enrolled:
            session["staff_pending_id"] = uid
            session["staff_pending_name"] = fname
            return jsonify({"success": True, "role": "staff", "face_required": True, "name": fname})
        else:
            # Face not enrolled — block login entirely
            return jsonify({"error": "⚠️ Your face is not enrolled yet. Please contact the admin to enroll your face before logging in."}), 403

@app.route("/logout")
def logout():
    uname = session.get("user_username", "")
    if uname: log_audit("logout", f"'{uname}' logged out")
    session.clear()
    return redirect(url_for("login"))

@app.route("/staff_face_login")
def staff_face_login():
    if not session.get("staff_pending_id"):
        return redirect(url_for("login"))
    return render_template("staff_face_login.html", staff_name=session.get("staff_pending_name", "Staff"))

@app.route("/staff_face_verify", methods=["POST"])
def staff_face_verify():
    pending_id = session.get("staff_pending_id")
    if not pending_id:
        return jsonify({"verified": False, "error": "Session expired. Please log in again."}), 400
    import numpy as np
    data = request.get_json(force=True)

    # Accept either a single face sample (legacy) or a list of samples for multi-frame voting
    raw_faces = data.get("faces")  # preferred: list of landmark arrays
    if not raw_faces:
        single = data.get("face")
        if single:
            raw_faces = [single]
    if not raw_faces:
        return jsonify({"verified": False, "error": "No face detected"}), 400

    # Build query embeddings for each submitted sample
    query_embeddings = []
    for face_lms in raw_faces:
        pts = np.array([[lm["x"], lm["y"], lm["z"]] for lm in face_lms], dtype="float32")
        if pts.shape[0] < 468:
            continue
        pts = pts[:468]   # 468 pts × 3 = 1404-d, matches trained model
        mean = pts.mean(axis=0)
        pts = pts - mean
        max_dist = np.max(np.abs(pts[:, :2]))
        if max_dist > 0:
            pts = pts / max_dist
        query_embeddings.append(pts.flatten())

    if not query_embeddings:
        return jsonify({"verified": False, "error": "Invalid face data"}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,username,full_name,role,is_active FROM users WHERE id=?", (pending_id,))
    user = c.fetchone()
    conn.close()
    if not user or not user[4]:
        session.pop("staff_pending_id", None); session.pop("staff_pending_name", None)
        return jsonify({"verified": False, "error": "Account inactive"}), 403

    safe_u = "".join([ch for ch in user[1] if ch.isalnum() or ch in ('-','_')])
    staff_folder = os.path.join(STAFF_DATASET_DIR, safe_u)
    from model import verify_face_for_user

    # --- Multi-sample voting ---
    # Threshold raised to 0.92 for tighter person-specific matching.
    # Require at least ceil(len/2)+1 samples to pass (majority) with a minimum of 3.
    THRESHOLD = 0.80
    MIN_VOTES_REQUIRED = max(3, (len(query_embeddings) // 2) + 1)
    passed = 0
    best_sim = 0.0
    worst_sim = 1.0

    for emb in query_embeddings:
        ok, sim = verify_face_for_user(emb, staff_folder, threshold=THRESHOLD)
        if sim > best_sim:
            best_sim = sim
        if sim < worst_sim:
            worst_sim = sim
        # Hard reject: if any sample scores very low the face is clearly different
        if sim < 0.60:
            log_audit("face_verify_fail", f"Staff '{user[1]}' hard-rejected (sim={sim:.2f})")
            return jsonify({"verified": False,
                            "similarity": round(sim * 100, 1),
                            "error": "Face not recognized. Please try again."})
        if ok:
            passed += 1

    avg_sim = best_sim  # report best similarity to frontend
    if passed >= MIN_VOTES_REQUIRED:
        session.pop("staff_pending_id", None); session.pop("staff_pending_name", None)
        session["user_id"] = user[0]; session["user_username"] = user[1]
        session["user_name"] = user[2]; session["user_role"] = user[3]
        log_audit("login", f"Staff '{user[1]}' face-verified ({passed}/{len(query_embeddings)} samples, best_sim={best_sim:.2f})")
        return jsonify({"verified": True, "similarity": round(avg_sim * 100, 1),
                        "redirect": url_for("mark_attendance_page")})

    log_audit("face_verify_fail", f"Staff '{user[1]}' failed ({passed}/{len(query_embeddings)} samples passed, best={best_sim:.2f})")
    return jsonify({"verified": False, "similarity": round(avg_sim * 100, 1),
                    "error": "Face not recognized. Please try again."})

@app.route("/staff_face_verify_image", methods=["POST"])
def staff_face_verify_image():
    """
    Cosine-based staff face verification.
    Receives JPEG frames from browser canvas, compares against enrolled photos
    using face mesh topology embeddings and cosine similarity.
    """
    pending_id = session.get("staff_pending_id")
    if not pending_id:
        return jsonify({"verified": False, "error": "Session expired. Please log in again."}), 400

    images = request.files.getlist("frames[]")
    if not images:
        return jsonify({"verified": False, "error": "No image frames received"}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,username,full_name,role,is_active FROM users WHERE id=?", (pending_id,))
    user = c.fetchone()
    conn.close()
    if not user or not user[4]:
        session.pop("staff_pending_id", None); session.pop("staff_pending_name", None)
        return jsonify({"verified": False, "error": "Account inactive"}), 403

    safe_u = "".join([ch for ch in user[1] if ch.isalnum() or ch in ('-','_')])
    staff_folder = os.path.join(STAFF_DATASET_DIR, safe_u)

    from model import verify_staff_with_sface

    passed = 0
    total_valid = 0
    best_sim = 0.0
    all_sims = []

    for img_file in images:
        try:
            image_bytes = img_file.read()
            # SFace uses Cosine similarity threshold (min_score=0.363 recommended)
            # Default is 0.363, adjust min_score if necessary
            verified, sim_pct, face_detected = verify_staff_with_sface(image_bytes, staff_folder, max_train=30, min_score=0.363)
            if not face_detected:
                # No face detected in this frame — skip (not penalise)
                continue
            total_valid += 1
            all_sims.append(sim_pct)
            if sim_pct > best_sim:
                best_sim = sim_pct
            if verified:
                passed += 1
        except Exception as e:
            app.logger.warning("staff_face_verify_image Cosine error: %s", e)
            continue

    if total_valid == 0:
        return jsonify({"verified": False, "similarity": 0,
                        "error": "No face detected in any frame. Please look at the camera."})

    avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0.0
    # Require majority of valid frames to pass Cosine check
    # Fix: If total_valid is 1, MIN_VOTES should be 1, not 2.
    MIN_VOTES = max(1, (total_valid // 2) + 1)

    app.logger.info(f"Cosine verify: user={user[1]} passed={passed}/{total_valid} avg_sim={avg_sim:.1f} best={best_sim:.1f}")

    if passed >= MIN_VOTES:
        session.pop("staff_pending_id", None); session.pop("staff_pending_name", None)
        session["user_id"] = user[0]; session["user_username"] = user[1]
        session["user_name"] = user[2]; session["user_role"] = user[3]
        log_audit("login", f"Staff '{user[1]}' Cosine-verified ({passed}/{total_valid} frames, sim={avg_sim:.1f})")
        return jsonify({"verified": True, "similarity": round(avg_sim, 1),
                        "redirect": url_for("mark_attendance_page")})

    log_audit("face_verify_fail", f"Staff '{user[1]}' Cosine-failed ({passed}/{total_valid} frames, sim={avg_sim:.1f})")
    return jsonify({"verified": False, "similarity": round(avg_sim, 1),
                    "error": "Face not recognized. Please try again."})

# =============================================================================
# Main Routes (auth protected)
# =============================================================================
@app.route("/")
def index():
    redir = require_login()
    if redir: return redir
    institution_name = get_setting("institution_name", "Digital Attendance System")
    return render_template("index.html", institution_name=institution_name)

# Dashboard simple API for attendance stats (last 30 days)
@app.route("/attendance_stats")
def attendance_stats():
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT timestamp FROM attendance", conn)
    conn.close()
    if df.empty:
        last_30 = [(datetime.date.today() - datetime.timedelta(days=i)) for i in range(29, -1, -1)]
        days = [d.strftime("%d-%b") for d in last_30]
        return jsonify({"dates": days, "counts": [0]*30})
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    last_30 = [(datetime.date.today() - datetime.timedelta(days=i)) for i in range(29, -1, -1)]
    counts = [int(df[df['date'] == d].shape[0]) for d in last_30]
    dates = [d.strftime("%d-%b") for d in last_30]
    return jsonify({"dates": dates, "counts": counts})

# -------- Add student (form) --------
@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    redir = require_admin()
    if redir: return redir
    if request.method == "GET":
        return render_template("add_student.html")
    data = request.form
    name = data.get("name", "").strip()
    roll = data.get("roll", "").strip()
    cls = data.get("class", "").strip()
    sec = data.get("sec", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if roll:
        c.execute("SELECT id FROM students WHERE roll=?", (roll,))
        if c.fetchone():
            conn.close()
            return jsonify({"error": f"Student with Roll No '{roll}' already exists"}), 400
    now = datetime.datetime.now().isoformat()
    c.execute("INSERT INTO students (name, roll, class, section, created_at) VALUES (?, ?, ?, ?, ?)",
              (name, roll, cls, sec, now))
    sid = c.lastrowid
    conn.commit()
    conn.close()
    safe_roll = "".join([ch for ch in roll if ch.isalnum() or ch in ('-', '_')])
    if not safe_roll:
        safe_roll = str(sid)
    new_path = os.path.join(DATASET_DIR, safe_roll)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    log_audit("add_student", f"Added student '{name}' (Roll: {roll})")
    return jsonify({"status": "success", "student_id": sid, "roll": roll})

# -------- Upload face images --------
@app.route("/upload_face", methods=["POST"])
def upload_face():
    student_id = request.form.get("student_id")
    if not student_id:
        return jsonify({"error": "student_id required"}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT roll FROM students WHERE id=?", (student_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Student not found"}), 404
    roll = row[0]
    safe_roll = "".join([ch for ch in roll if ch.isalnum() or ch in ('-', '_')])
    if not safe_roll:
        safe_roll = str(student_id)
    files = request.files.getlist("images[]")
    saved = 0
    folder = os.path.join(DATASET_DIR, safe_roll)
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    for f in files:
        try:
            fname = f"{datetime.datetime.now().timestamp():.6f}_{saved}.jpg"
            path = os.path.join(folder, fname)
            f.save(path)
            saved += 1
        except Exception as e:
            app.logger.error("save error: %s", e)
    # Auto-retrain if enabled
    if get_setting("auto_retrain", "1") == "1":
        status = read_train_status()
        if not status.get("running"):
            _start_training_thread()
    return jsonify({"saved": saved})

# -------- Duplicate face check (called during registration) --------
@app.route("/check_face_duplicate", methods=["POST"])
def check_face_duplicate():
    """
    Accepts a JPEG image and checks whether the face already belongs to
    a registered student using Cosine similarity face recognition.
    Compares face mesh embeddings generated from the live image against enrolled photos.
    Returns {duplicate: true, matched_student, similarity} or {duplicate: false}.
    """
    if "image" not in request.files:
        return jsonify({"duplicate": False, "reason": "no_image"}), 400

    # Skip check if no student folders exist yet
    if not os.path.isdir(DATASET_DIR) or not os.listdir(DATASET_DIR):
        return jsonify({"duplicate": False, "reason": "no_dataset"})

    img_file = request.files["image"]
    similarity_pct = 0.0
    try:
        from model import find_duplicate_lbph

        image_bytes = img_file.read()
        matched_folder, min_distance, similarity_pct = find_duplicate_lbph(
            image_bytes, DATASET_DIR, max_ref_per_student=10, max_distance=45.0
        )

        if matched_folder:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT id, name, roll, class, section FROM students WHERE roll=?", (matched_folder,))
            row = c.fetchone()
            if not row:
                c.execute("SELECT id, name, roll, class, section FROM students")
                all_students = c.fetchall()
                for s in all_students:
                    safe = "".join([ch for ch in (s[2] or "") if ch.isalnum() or ch in ('-', '_')])
                    if safe == matched_folder:
                        row = s
                        break
            conn.close()

            if row:
                app.logger.info(f"Duplicate detected: {row[1]} (LBPH min_distance={min_distance:.2f}, sim={similarity_pct:.1f}%)")
                return jsonify({
                    "duplicate": True,
                    "similarity": round(similarity_pct, 1),
                    "matched_student": {
                        "id": row[0],
                        "name": row[1],
                        "roll": row[2] or matched_folder,
                        "class": row[3] or "",
                        "section": row[4] or ""
                    }
                })

        return jsonify({"duplicate": False, "similarity": round(similarity_pct, 1)})

    except Exception as e:
        app.logger.exception("check_face_duplicate error")
        return jsonify({"duplicate": False, "reason": str(e)})


# -------- Cancel/rollback a partially-created student --------
@app.route("/cancel_student/<int:sid>", methods=["DELETE"])
def cancel_student(sid):
    """Delete a student record + folder that was created but then rejected (e.g. duplicate face)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT roll, name FROM students WHERE id=?", (sid,))
    row = c.fetchone()
    if row:
        roll, name = row
        c.execute("DELETE FROM students WHERE id=?", (sid,))
        c.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
        conn.commit()
        # Delete dataset folder
        safe_roll = "".join([ch for ch in (roll or "") if ch.isalnum() or ch in ('-', '_')])
        for folder_name in [safe_roll, str(sid)]:
            if folder_name:
                folder = os.path.join(DATASET_DIR, folder_name)
                if os.path.exists(folder):
                    shutil.rmtree(folder, ignore_errors=True)
        log_audit("duplicate_rejected", f"Registration cancelled for '{name}' (id={sid}) — duplicate face detected")
    conn.close()
    return jsonify({"cancelled": True})


# -------- Train model --------
def _start_training_thread():
    def update_status(p, m):
        running = True
        if p >= 100:
            running = False
        write_train_status({"running": running, "progress": p, "message": m})
        if not running:
            pass
    def run_training():
        try:
            train_model_background(DATASET_DIR, update_status)
        except Exception as e:
            app.logger.exception("Training failed")
            write_train_status({"running": False, "progress": 0, "message": f"Error: {str(e)}"})
    t = threading.Thread(target=run_training)
    t.daemon = True
    t.start()

@app.route("/train_model", methods=["GET"])
def train_model_route():
    status = read_train_status()
    if status.get("running"):
        return jsonify({"status": "already_running"}), 202
    _start_training_thread()
    return jsonify({"status": "started"}), 202

@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())

# -------- Mark attendance page --------
@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    redir = require_login()
    if redir: return redir
    return render_template("mark_attendance.html")


def _get_late_status(ts_str):
    """Returns 'late' if time is after cutoff, else 'present'."""
    try:
        cutoff_str = get_setting("late_cutoff_time", "09:15")
        ts = datetime.datetime.fromisoformat(ts_str)
        cutoff_h, cutoff_m = map(int, cutoff_str.split(":"))
        cutoff_time = ts.replace(hour=cutoff_h, minute=cutoff_m, second=0, microsecond=0)
        return "late" if ts > cutoff_time else "present"
    except Exception:
        return "present"

# -------- Recognize face endpoint (POST image) --------
@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    if "image" not in request.files:
        return jsonify({"recognized": False, "error": "no image"}), 400
        
    img_file = request.files["image"]
    
    try:
        from model import load_student_sface_embeddings, get_all_sface_embeddings, get_sface_recognizer
        import numpy as np
        import cv2
        
        arr = np.frombuffer(img_file.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"recognized": False, "error": "invalid image"}), 400
            
        live_embs = get_all_sface_embeddings(img)
        if not live_embs:
            return jsonify({"recognized": False, "error": "no face detected"}), 200
            
        db_embeddings = load_student_sface_embeddings()
        if not db_embeddings:
            return jsonify({"recognized": False, "error": "model not trained"}), 200

        recognizer = get_sface_recognizer()
        
        final_results = []
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        today_local = datetime.datetime.now().date().isoformat()
        ts = datetime.datetime.now().isoformat()
        att_status = _get_late_status(ts)

        for live_emb in live_embs:
            best_score = 0.0
            best_roll = None
            
            # SFace gives high robustness. Cosine score >= 0.363 is identical person.
            for roll, embs in db_embeddings.items():
                for ref_emb in embs:
                    score = recognizer.match(live_emb, ref_emb, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > best_score:
                        best_score = score
                        best_roll = roll
            
            # Min threshold logic.
            min_score = 0.363
            similarity_pct = max(0.0, min(100.0, best_score * 100.0))
            
            app.logger.info(f"SFace Predict: Roll {best_roll} -> score: {best_score:.3f}, sim: {similarity_pct:.1f}%")

            if best_score < min_score or not best_roll:
                final_results.append({
                    "recognized": False, 
                    "confidence": similarity_pct / 100.0, 
                    "status": "unknown_student"
                })
                continue

            pred_roll = best_roll
            c.execute("SELECT id, name FROM students WHERE roll=?", (pred_roll,))
            row = c.fetchone()
            if not row:
                final_results.append({
                    "recognized": False, 
                    "confidence": similarity_pct / 100.0, 
                    "status": "unknown_student"
                })
                continue
                
            student_id, name = row

            c.execute("SELECT id, timestamp FROM attendance WHERE student_id=? AND date(timestamp)=? ORDER BY id ASC LIMIT 1",
                      (student_id, today_local))
            existing = c.fetchone()
            
            result_payload = {
                "recognized": True, 
                "student_id": student_id, 
                "roll": pred_roll, 
                "name": name,
                "confidence": similarity_pct / 100.0
            }

            if existing:
                result_payload.update({"status": "already_marked", "marked_at": existing[1]})
            else:
                c.execute("INSERT INTO attendance (student_id, name, timestamp, status) VALUES (?, ?, ?, ?)",
                          (student_id, name, ts, att_status))
                result_payload.update({"status": "marked", "marked_at": ts, "att_status": att_status})
                
            final_results.append(result_payload)
            
        conn.commit()
        conn.close()
        
        return jsonify({"recognized": True, "results": final_results}), 200

    except Exception as e:
        app.logger.exception("recognize error")
        return jsonify({"recognized": False, "error": str(e)}), 500


# -------- Dashboard summary stats --------
@app.route("/dashboard_summary")
def dashboard_summary():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM students")
    total_students = c.fetchone()[0]
    today_utc = datetime.date.today().isoformat()
    c.execute("SELECT COUNT(*) FROM attendance WHERE date(timestamp) = ?", (today_utc,))
    today_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM attendance")
    total_records = c.fetchone()[0]
    # Absentee count
    threshold = float(get_setting("low_attendance_threshold", "75"))
    c.execute("SELECT COUNT(DISTINCT date(timestamp)) FROM attendance")
    total_days = c.fetchone()[0] or 1
    c.execute("SELECT student_id, COUNT(*) FROM attendance GROUP BY student_id")
    rows = c.fetchall()
    absentee_count = sum(1 for r in rows if (r[1] / total_days * 100) < threshold)
    conn.close()
    return jsonify({"total_students": total_students, "today_count": today_count,
                    "total_records": total_records, "absentee_count": absentee_count})

# -------- Per-student attendance percentage --------
@app.route("/student_attendance_percent")
def student_attendance_percent():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT date(timestamp)) FROM attendance")
    total_days = c.fetchone()[0] or 1
    c.execute("SELECT student_id, COUNT(*) FROM attendance GROUP BY student_id")
    rows = c.fetchall()
    conn.close()
    data = {str(r[0]): round(min(100, (r[1] / total_days) * 100), 1) for r in rows}
    return jsonify(data)

# -------- Absentee alerts --------
@app.route("/absentee_alerts")
def absentee_alerts():
    threshold = float(get_setting("low_attendance_threshold", "75"))
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT date(timestamp)) FROM attendance")
    total_days = c.fetchone()[0] or 1
    c.execute("""SELECT s.id, s.name, s.roll, s.class, s.section, COUNT(a.id) as att_count
                 FROM students s LEFT JOIN attendance a ON s.id = a.student_id
                 GROUP BY s.id""")
    rows = c.fetchall()
    conn.close()
    alerts = []
    for r in rows:
        sid, name, roll, cls, sec, cnt = r
        pct = round(min(100, (cnt / total_days * 100)), 1)
        if pct < threshold:
            alerts.append({"id": sid, "name": name, "roll": roll, "class": cls,
                           "section": sec, "percentage": pct, "days_present": cnt,
                           "total_days": total_days})
    return jsonify({"threshold": threshold, "alerts": alerts})

# -------- Per-student report --------
@app.route("/student_report/<int:sid>")
def student_report(sid):
    redir = require_login()
    if redir: return redir
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, roll, class, section, created_at FROM students WHERE id=?", (sid,))
    student = c.fetchone()
    if not student:
        conn.close()
        return abort(404)
    c.execute("SELECT id, timestamp, status FROM attendance WHERE student_id=? ORDER BY timestamp DESC", (sid,))
    records = c.fetchall()
    c.execute("SELECT COUNT(DISTINCT date(timestamp)) FROM attendance")
    total_days = c.fetchone()[0] or 1
    conn.close()
    # Build attendance by date for calendar
    att_dates = {}
    for r in records:
        d = r[1][:10] if r[1] else ""
        if d:
            att_dates[d] = r[2] or "present"
    pct = round(min(100, len(att_dates) / total_days * 100), 1)
    return render_template("student_report.html", student=student, records=records,
                           att_dates=att_dates, pct=pct, total_days=total_days)

# -------- Settings page --------
@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    redir = require_admin()
    if redir: return redir
    if request.method == "POST":
        keys = ["late_cutoff_time", "low_attendance_threshold", "recognition_confidence",
                "institution_name", "auto_retrain"]
        for k in keys:
            v = request.form.get(k, "").strip()
            if v:
                set_setting(k, v)
        log_audit("settings_changed", "Admin updated system settings")
        return redirect(url_for("settings_page") + "?saved=1")
    settings = {}
    for k in ["late_cutoff_time", "low_attendance_threshold", "recognition_confidence",
               "institution_name", "auto_retrain"]:
        settings[k] = get_setting(k, "")
    saved = request.args.get("saved", "0") == "1"
    return render_template("settings.html", settings=settings, saved=saved)

# -------- Manual Attendance Override --------
@app.route("/manual_attendance", methods=["POST"])
def manual_attendance():
    redir = require_login()
    if redir: return jsonify({"error": "not logged in"}), 401
    data = request.get_json(force=True)
    student_id = data.get("student_id")
    date_str = data.get("date")  # YYYY-MM-DD
    reason = data.get("reason", "Manual override")
    if not student_id or not date_str:
        return jsonify({"error": "student_id and date required"}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name FROM students WHERE id=?", (student_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Student not found"}), 404
    name = row[1]
    # Check if already marked for that date
    c.execute("SELECT id FROM attendance WHERE student_id=? AND date(timestamp)=?", (student_id, date_str))
    if c.fetchone():
        conn.close()
        return jsonify({"error": f"Attendance already marked for {date_str}"}), 409
    ts = f"{date_str}T09:00:00"
    c.execute("INSERT INTO attendance (student_id, name, timestamp, status) VALUES (?, ?, ?, ?)",
              (student_id, name, ts, "manual"))
    conn.commit()
    conn.close()
    log_audit("manual_attendance", f"Manual attendance for student_id={student_id} on {date_str}. Reason: {reason}")
    return jsonify({"success": True, "name": name, "date": date_str})

# -------- Attendance records & filters --------
@app.route("/attendance_record", methods=["GET"])
def attendance_record():
    redir = require_login()
    if redir: return redir
    period = request.args.get("period", "all")
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    q = "SELECT a.id, s.roll, a.name, a.timestamp, COALESCE(a.status,'present') FROM attendance a LEFT JOIN students s ON a.student_id = s.id"
    params = ()
    if start_date and end_date:
        q += " WHERE date(a.timestamp) BETWEEN ? AND ?"
        params = (start_date, end_date)
    elif start_date:
        q += " WHERE date(a.timestamp) >= ?"
        params = (start_date,)
    elif period == "daily":
        today = datetime.date.today().isoformat()
        q += " WHERE date(a.timestamp) = ?"
        params = (today,)
    elif period == "weekly":
        start = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
        q += " WHERE date(a.timestamp) >= ?"
        params = (start,)
    elif period == "monthly":
        start = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
        q += " WHERE date(a.timestamp) >= ?"
        params = (start,)
    q += " ORDER BY a.timestamp DESC LIMIT 5000"
    c.execute(q, params)
    rows = c.fetchall()
    # Get students list for manual attendance modal
    c.execute("SELECT id, name, roll FROM students ORDER BY name")
    all_students = c.fetchall()
    conn.close()
    today_date = datetime.date.today().isoformat()
    return render_template("attendance_record.html", records=rows, period=period,
                           start_date=start_date, end_date=end_date, all_students=all_students,
                           today_date=today_date)

# -------- Clear all attendance --------
@app.route("/clear_all_attendance", methods=["DELETE"])
def clear_all_attendance():
    redir = require_admin()
    if redir: return jsonify({"error": "not authorized"}), 401
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()
    log_audit("clear_all_attendance", "All attendance records cleared")
    return jsonify({"cleared": True})

# -------- CSV download --------
@app.route("/download_csv", methods=["GET"])
def download_csv():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT a.id, s.roll, a.name, a.timestamp, COALESCE(a.status,'present') FROM attendance a LEFT JOIN students s ON a.student_id = s.id ORDER BY a.timestamp DESC")
    rows = c.fetchall()
    conn.close()
    output = io.StringIO()
    output.write("id,roll_no,name,timestamp,status\n")
    for r in rows:
        output.write(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n')
    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="attendance.csv", mimetype="text/csv")

# -------- PDF download --------
@app.route("/download_pdf", methods=["GET"])
def download_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        return jsonify({"error": "reportlab not installed. Run: pip install reportlab"}), 500

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT a.id, s.roll, a.name, a.timestamp, COALESCE(a.status,'present') FROM attendance a LEFT JOIN students s ON a.student_id = s.id ORDER BY a.timestamp DESC LIMIT 1000")
    rows = c.fetchall()
    conn.close()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    elements = []
    inst_name = get_setting("institution_name", "Digital Attendance System")
    elements.append(Paragraph(inst_name, styles["Title"]))
    elements.append(Paragraph(f"Attendance Report — Generated: {datetime.datetime.now().strftime('%d %b %Y %I:%M %p')}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    data = [["#", "Roll No", "Name", "Date & Time", "Status"]]
    for i, r in enumerate(rows, 1):
        ts = r[3][:16].replace("T", " ") if r[3] else ""
        data.append([str(i), r[1] or "—", r[2], ts, (r[4] or "present").capitalize()])

    t = Table(data, colWidths=[30, 70, 130, 140, 70])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(t)
    doc.build(elements)
    buf.seek(0)
    fname = f"attendance_{datetime.date.today().isoformat()}.pdf"
    return send_file(buf, as_attachment=True, download_name=fname, mimetype="application/pdf")

# -------- Students API for listing/editing --------
@app.route("/students", methods=["GET"])
def students_list():
    redir = require_login()
    if redir: return jsonify({"error": "not authorized"}), 401
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, roll, class, section, created_at FROM students ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    data = [{"id": r[0], "name": r[1], "roll": r[2], "class": r[3], "section": r[4], "created_at": r[5]} for r in rows]
    return jsonify({"students": data})

@app.route("/students/<int:sid>", methods=["PUT"])
def edit_student(sid):
    redir = require_admin()
    if redir: return jsonify({"error": "not authorized"}), 401
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    roll = data.get("roll", "").strip()
    cls = data.get("class", "").strip()
    sec = data.get("section", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Check duplicate roll for other students
    if roll:
        c.execute("SELECT id FROM students WHERE roll=? AND id!=?", (roll, sid))
        if c.fetchone():
            conn.close()
            return jsonify({"error": f"Roll No '{roll}' already used by another student"}), 400
    # Get old roll to rename dataset folder
    c.execute("SELECT roll FROM students WHERE id=?", (sid,))
    row = c.fetchone()
    old_roll = row[0] if row else None
    c.execute("UPDATE students SET name=?, roll=?, class=?, section=? WHERE id=?", (name, roll, cls, sec, sid))
    conn.commit()
    conn.close()
    # Rename dataset folder if roll changed
    if old_roll and roll and old_roll != roll:
        old_safe = "".join([ch for ch in old_roll if ch.isalnum() or ch in ('-', '_')])
        new_safe = "".join([ch for ch in roll if ch.isalnum() or ch in ('-', '_')])
        old_folder = os.path.join(DATASET_DIR, old_safe) if old_safe else None
        new_folder = os.path.join(DATASET_DIR, new_safe) if new_safe else None
        if old_folder and new_folder and os.path.exists(old_folder) and not os.path.exists(new_folder):
            os.rename(old_folder, new_folder)
    log_audit("edit_student", f"Edited student id={sid}: name='{name}', roll='{roll}'")
    return jsonify({"updated": True})

# -------- Manage Students Page --------
@app.route("/manage_students", methods=["GET"])
def manage_students():
    redir = require_admin()
    if redir: return redir
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, roll, class, section, created_at FROM students ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    threshold = float(get_setting("low_attendance_threshold", "75"))
    return render_template("manage_students.html", students=rows, threshold=threshold)

@app.route("/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT roll, name FROM students WHERE id=?", (sid,))
    row = c.fetchone()
    roll = row[0] if row else None
    name = row[1] if row else "Unknown"
    c.execute("DELETE FROM students WHERE id=?", (sid,))
    c.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
    conn.commit()
    conn.close()
    if roll:
        safe_roll = "".join([ch for ch in roll if ch.isalnum() or ch in ('-', '_')])
        if not safe_roll: safe_roll = str(sid)
        folder = os.path.join(DATASET_DIR, safe_roll)
        if os.path.exists(folder):
            shutil.rmtree(folder, ignore_errors=True)
    folder_legacy = os.path.join(DATASET_DIR, str(sid))
    if os.path.exists(folder_legacy):
        shutil.rmtree(folder_legacy, ignore_errors=True)
    log_audit("delete_student", f"Deleted student id={sid}, name='{name}'")
    return jsonify({"deleted": True})

@app.route("/delete_attendance/<int:aid>", methods=["DELETE"])
def delete_attendance(aid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM attendance WHERE id=?", (aid,))
    conn.commit()
    conn.close()
    return jsonify({"deleted": True})

@app.route("/student_image/<int:sid>")
def student_image(sid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT roll FROM students WHERE id=?", (sid,))
    row = c.fetchone()
    conn.close()
    roll = row[0] if row else None
    folder = None
    if roll:
        safe_roll = "".join([ch for ch in roll if ch.isalnum() or ch in ('-', '_')])
        if not safe_roll: safe_roll = str(sid)
        path = os.path.join(DATASET_DIR, safe_roll)
        if os.path.isdir(path):
            folder = path
    if not folder:
        path = os.path.join(DATASET_DIR, str(sid))
        if os.path.isdir(path):
            folder = path
    if not folder:
        return abort(404)
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        return abort(404)
    return send_file(os.path.join(folder, files[0]))

# -------- Unknown Faces --------
@app.route("/unknown_faces")
def unknown_faces():
    files = []
    for f in os.listdir(UNKNOWN_DIR):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(f)
    return jsonify({"files": sorted(files, reverse=True)})

# -------- Audit Log --------
@app.route("/audit_log")
def audit_log():
    redir = require_admin()
    if redir: return redir
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, action, details, timestamp FROM audit_log ORDER BY id DESC LIMIT 200")
    rows = c.fetchall()
    conn.close()
    return render_template("audit_log.html", logs=rows)

# ---------------- run ------------------------

# =============================================================================
# Admin — Staff Management Routes
# =============================================================================
@app.route("/admin/staff")
def admin_staff_list():
    redir = require_admin()
    if redir: return redir
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,username,full_name,role,face_enrolled,created_at,is_active FROM users ORDER BY role DESC, id ASC")
    users = c.fetchall()
    conn.close()
    return render_template("admin_staff.html", staff_users=users)

@app.route("/admin/staff/add", methods=["POST"])
def admin_staff_add():
    redir = require_admin()
    if redir: return jsonify({"error": "not authorized"}), 401
    data = request.get_json(force=True)
    username = data.get("username", "").strip().lower()
    full_name = data.get("full_name", "").strip()
    role = data.get("role", "staff")
    password = data.get("password", "").strip()
    if not username or not full_name or not password:
        return jsonify({"error": "username, full_name and password required"}), 400
    if role not in ("admin", "staff"):
        return jsonify({"error": "role must be admin or staff"}), 400
    now = datetime.datetime.now().isoformat()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (username,full_name,role,password_hash,face_enrolled,created_at,is_active) VALUES (?,?,?,?,0,?,1)",
                  (username, full_name, role, hash_password(password), now))
        uid = c.lastrowid
        conn.commit(); conn.close()
        safe_u = "".join([ch for ch in username if ch.isalnum() or ch in ('-','_')])
        os.makedirs(os.path.join(STAFF_DATASET_DIR, safe_u), exist_ok=True)
        log_audit("add_user", f"Added user '{username}' (role:{role})")
        return jsonify({"success": True, "id": uid})
    except sqlite3.IntegrityError:
        return jsonify({"error": f"Username '{username}' already exists"}), 400

@app.route("/admin/staff/<int:uid>", methods=["PUT"])
def admin_staff_edit(uid):
    redir = require_admin()
    if redir: return jsonify({"error": "not authorized"}), 401
    data = request.get_json(force=True)
    full_name = data.get("full_name", "").strip()
    is_active = int(data.get("is_active", 1))
    password = data.get("password", "").strip()
    if not full_name: return jsonify({"error": "full_name required"}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if password:
        c.execute("UPDATE users SET full_name=?,is_active=?,password_hash=? WHERE id=?",
                  (full_name, is_active, hash_password(password), uid))
    else:
        c.execute("UPDATE users SET full_name=?,is_active=? WHERE id=?", (full_name, is_active, uid))
    conn.commit(); conn.close()
    log_audit("edit_user", f"Edited user id={uid}")
    return jsonify({"updated": True})

@app.route("/admin/staff/<int:uid>", methods=["DELETE"])
def admin_staff_delete(uid):
    redir = require_admin()
    if redir: return jsonify({"error": "not authorized"}), 401
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username,role FROM users WHERE id=?", (uid,))
    row = c.fetchone()
    if not row:
        conn.close(); return jsonify({"error": "User not found"}), 404
    if row[1] == "admin":
        c.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND is_active=1")
        if c.fetchone()[0] <= 1:
            conn.close(); return jsonify({"error": "Cannot delete the last admin"}), 400
    c.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit(); conn.close()
    safe_u = "".join([ch for ch in row[0] if ch.isalnum() or ch in ('-','_')])
    folder = os.path.join(STAFF_DATASET_DIR, safe_u)
    if os.path.exists(folder): shutil.rmtree(folder, ignore_errors=True)
    log_audit("delete_user", f"Deleted user '{row[0]}'")
    return jsonify({"deleted": True})

@app.route("/admin/staff/<int:uid>/enroll_face")
def admin_staff_enroll_face(uid):
    redir = require_admin()
    if redir: return redir
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,username,full_name,role FROM users WHERE id=?", (uid,))
    user = c.fetchone()
    conn.close()
    if not user: return abort(404)
    return render_template("staff_enroll_face.html", staff_user=user)

@app.route("/upload_staff_face", methods=["POST"])
def upload_staff_face():
    redir = require_admin()
    if redir: return jsonify({"error": "not authorized"}), 401
    staff_id = request.form.get("staff_id")
    if not staff_id: return jsonify({"error": "staff_id required"}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE id=?", (staff_id,))
    row = c.fetchone()
    conn.close()
    if not row: return jsonify({"error": "User not found"}), 404
    safe_u = "".join([ch for ch in row[0] if ch.isalnum() or ch in ('-','_')])
    folder = os.path.join(STAFF_DATASET_DIR, safe_u)
    # Clear old enrollment data so LBPH model is rebuilt fresh
    if os.path.isdir(folder):
        for old_file in os.listdir(folder):
            old_path = os.path.join(folder, old_file)
            try:
                os.remove(old_path)
            except Exception:
                pass
    os.makedirs(folder, exist_ok=True)
    files = request.files.getlist("images[]")
    saved = 0
    for f in files:
        try:
            f.save(os.path.join(folder, f"{datetime.datetime.now().timestamp():.6f}_{saved}.jpg"))
            saved += 1
        except Exception as e:
            app.logger.error("staff face save error: %s", e)
    if saved > 0:
        # Invalidate any stale LBPH cache so the next verification retrains on fresh photos
        from model import invalidate_lbph_cache
        invalidate_lbph_cache(folder)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE users SET face_enrolled=1 WHERE id=?", (staff_id,))
        conn.commit(); conn.close()
        log_audit("enroll_face", f"Enrolled face for user id={staff_id} ({saved} photos)")
    return jsonify({"saved": saved})

if __name__ == "__main__":
    app.run(debug=True)