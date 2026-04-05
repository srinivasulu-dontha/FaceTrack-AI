import os
import cv2
import threading
import numpy as np
import pickle
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model.pkl"

# ---- Utility: extract FaceMesh landmark embedding ----
# Uses 468 3D landmarks (x, y, z) = 1404 floats, normalized to face bounding box.
# This is far more discriminative than a small grayscale pixel vector.

def _extract_landmark_embedding(bgr_image, face_mesh):
    """
    Given a BGR image and a MediaPipe FaceMesh object,
    returns a normalized landmark embedding (1404-d float32) or None.
    """
    h, w = bgr_image.shape[:2]
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    # Get the first face's landmarks
    landmarks = results.multi_face_landmarks[0].landmark

    # Build raw coordinate array
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)  # (468, 3)

    # Normalize: mean center and scale by max absolute distance (X and Y only for stability)
    mean = pts.mean(axis=0)
    pts = pts - mean
    # Robust normalization: use only X and Y range to define scale. 
    # Z-axis in MediaPipe is sensitive to different versions/platforms.
    max_dist = np.max(np.abs(pts[:, :2])) 
    if max_dist > 0:
        pts = pts / max_dist

    return pts.flatten()  # 1404-d vector


# Use thread-local storage so each thread gets its own MediaPipe instances
_thread_local = threading.local()


# ---- OpenCV Deep Learning Models (SFace & YuNet) ----
def get_yunet_detector(shape=(320, 320)):
    """Return a per-thread YuNet DL Face Detector."""
    if not hasattr(_thread_local, 'yunet'):
        _thread_local.yunet = cv2.FaceDetectorYN_create(
            "face_detection_yunet_2023mar.onnx",
            "",
            shape, # This shape must be updated per-image using setInputSize
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )
    return _thread_local.yunet

def get_sface_recognizer():
    """Return a per-thread SFace DL Face Recognizer."""
    if not hasattr(_thread_local, 'sface'):
        _thread_local.sface = cv2.FaceRecognizerSF_create(
            "face_recognition_sface_2021dec.onnx",
            ""
        )
    return _thread_local.sface

def get_sface_embedding(bgr_image):
    """
    Detect face via YuNet and extract 128-d SFace embedding.
    Returns the normalized 128-d numpy array, or None.
    """
    h, w, _ = bgr_image.shape
    detector = get_yunet_detector((w, h))
    detector.setInputSize((w, h))
    
    _, faces = detector.detect(bgr_image)
    if faces is None or len(faces) == 0:
        return None
        
    # Get largest face bbox
    best_face = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    
    recognizer = get_sface_recognizer()
    aligned_face = recognizer.alignCrop(bgr_image, best_face)
    feature = recognizer.feature(aligned_face)
    return feature[0] # Returns a 128-d vector


def get_all_sface_embeddings(bgr_image):
    """
    Detect ALL faces via YuNet and extract 128-d SFace embeddings for each.
    Returns a list of normalized 128-d numpy arrays, or an empty list.
    """
    h, w, _ = bgr_image.shape
    detector = get_yunet_detector((w, h))
    detector.setInputSize((w, h))
    
    _, faces = detector.detect(bgr_image)
    if faces is None or len(faces) == 0:
        return []
        
    recognizer = get_sface_recognizer()
    embeddings = []
    
    for face in faces:
        try:
            aligned_face = recognizer.alignCrop(bgr_image, face)
            feature = recognizer.feature(aligned_face)
            embeddings.append(feature[0])
        except Exception:
            continue
            
    return embeddings


def extract_all_embeddings(stream_or_bytes):
    """
    Extract landmark embeddings for ALL faces in the image.
    Returns a list of embedding arrays (one per detected face).
    """
    data = stream_or_bytes.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    face_mesh = _get_face_mesh()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return []

    embeddings = []
    for face_lms in results.multi_face_landmarks:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in face_lms.landmark], dtype=np.float32)
        mean = pts.mean(axis=0)
        pts = pts - mean
        max_dist = np.max(np.abs(pts[:, :2]))
        if max_dist > 0:
            pts = pts / max_dist
        embeddings.append(pts.flatten())

    return embeddings


def extract_embedding_for_image(stream_or_bytes):
    """Compatibility wrapper for single-face usage (e.g. add_student)."""
    embs = extract_all_embeddings(stream_or_bytes)
    return embs[0] if embs else None


# ---- Load model helpers ----
def load_student_sface_embeddings():
    if not os.path.exists(SFACE_MODEL_PATH):
        return None
    with open(SFACE_MODEL_PATH, "rb") as f:
        return pickle.load(f)

def load_model_if_exists():
    """DEPRECATED: Used for the old RandomForest model."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    if hasattr(clf, 'n_features_in_') and clf.n_features_in_ != 1404:
        return None
    return clf


# Removed obsolete predict_with_model and cosine_verify_against_folder

def compute_embedding_from_bytes(image_bytes):
    """
    Given raw JPEG/PNG bytes, extract the face landmark embedding.
    Returns a 1404-d numpy array or None if no face found.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    face_mesh = _get_face_mesh()
    return _extract_landmark_embedding(img, face_mesh)


def cosine_similarity(a, b):
    """Returns cosine similarity in [0, 1] between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def find_duplicate_in_dataset(query_embedding, dataset_dir, max_ref_per_student=5, threshold=0.90):
    """
    DEPRECATED stub — kept for import compatibility.
    Actual duplicate detection now uses find_duplicate_cosine() with image bytes.
    """
    return None, 0.0


def invalidate_lbph_cache(folder_path):
    """
    Delete the cached lbph_model.yml from a staff folder so it is retrained
    fresh on the next verification call.  Call this whenever new photos are
    saved to the folder (e.g. after re-enrollment).
    """
    cache_path = os.path.join(folder_path, "lbph_model.yml")
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except Exception:
            pass


def _get_cached_lbph(folder_path, max_images):
    """
    Checks for `lbph_model.yml` in a given folder. If it exists, loads and returns the recognizer.
    If not, processes up to `max_images` JPEGs from the folder, extracts grayscale ROIs,
    trains a new LBPH recognizer, saves it, and returns it.
    """
    cache_path = os.path.join(folder_path, "lbph_model.yml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists(cache_path):
        try:
            recognizer.read(cache_path)
            return recognizer
        except Exception:
            pass  # Corrupt cache, rebuild it

    # Rebuild cache
    image_files = sorted([f for f in os.listdir(folder_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        return None

    # Use more images for better LBPH training (up to 30)
    max_images = max(max_images, 30)
    step = max(1, len(image_files) // max_images)
    sampled = image_files[::step][:max_images]

    faces = []
    labels = []

    for fn in sampled:
        img = cv2.imread(os.path.join(folder_path, fn))
        if img is None: continue
        roi = _extract_face_roi(img)
        if roi is not None:
            faces.append(roi)
            labels.append(1)

    if faces:
        recognizer.train(faces, np.array(labels, dtype=np.int32))
        recognizer.write(cache_path)
        return recognizer
    return None


def find_duplicate_lbph(live_image_bytes, dataset_dir, max_ref_per_student=10, max_distance=45.0):
    """
    LBPH-based duplicate detection across all registered student folders.
    Extracts face ROI from live image, compares against registered cached LBPH models.
    Returns:
        (student_folder_name, min_distance, similarity_pct) if duplicate, else (None, 0.0, 0.0)
    """
    arr = np.frombuffer(live_image_bytes, np.uint8)
    live_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if live_img is None:
        return None, 0.0, 0.0

    live_roi = _extract_face_roi(live_img)
    if live_roi is None:
        return None, 0.0, 0.0

    best_folder = None
    min_dist = float('inf')

    for folder_name in sorted(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        recognizer = _get_cached_lbph(folder_path, max_ref_per_student)
        if recognizer is None:
            continue
        
        try:
            label, dist = recognizer.predict(live_roi)
            if dist < min_dist:
                min_dist = dist
                best_folder = folder_name
        except Exception:
            pass

    similarity_pct = max(0.0, 100.0 - min_dist)
    if best_folder is not None and min_dist <= max_distance:
        return best_folder, min_dist, similarity_pct

    return None, min_dist if best_folder else 0.0, similarity_pct



def verify_face_for_user(query_embedding, user_folder, max_ref=8, threshold=0.92):
    """
    Legacy landmark-based verification (kept for backward compat).
    New code should use verify_staff_with_cosine instead.
    """
    if not os.path.isdir(user_folder):
        return False, 0.0

    image_files = [f for f in os.listdir(user_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        return False, 0.0

    face_mesh_inst = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.5
    )

    step = max(1, len(image_files) // max_ref)
    sampled = image_files[::step][:max_ref]
    best_sim = 0.0

    for fn in sampled:
        img = cv2.imread(os.path.join(user_folder, fn))
        if img is None:
            continue
        ref_emb = _extract_landmark_embedding(img, face_mesh_inst)
        if ref_emb is None:
            continue
        sim = cosine_similarity(query_embedding, ref_emb)
        if sim > best_sim:
            best_sim = sim

    face_mesh_inst.close()
    return best_sim >= threshold, best_sim


# ---- Haar cascade (thread-safe, loaded once) ----
_face_cascade = None
_face_cascade_lock = threading.Lock()

def _get_face_cascade():
    global _face_cascade
    with _face_cascade_lock:
        if _face_cascade is None:
            _face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    return _face_cascade


def _extract_face_roi(bgr_image, target_size=(100, 100)):
    """
    Detect and crop the face ROI from a BGR image using MediaPipe, resize to target_size.
    Returns a grayscale face ROI or None if no face detected.
    """
    detector = _get_face_detector()
    h, w = bgr_image.shape[:2]
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if not results.detections:
        return None

    # Pick the largest detection (use bbox area since proportions is not always reliable)
    best_det = sorted(results.detections, 
                      key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height, 
                      reverse=True)[0]
    bbox = best_det.location_data.relative_bounding_box
    
    xmin = max(0, int(bbox.xmin * w))
    ymin = max(0, int(bbox.ymin * h))
    width = int(bbox.width * w)
    height = int(bbox.height * h)
    
    xmax = min(w, xmin + width)
    ymax = min(h, ymin + height)

    if xmax <= xmin or ymax <= ymin:
        return None

    roi = bgr_image[ymin:ymax, xmin:xmax]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for lighting normalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    return cv2.resize(gray, target_size)


def verify_staff_with_sface(live_image_bytes, user_folder, max_train=5, min_score=0.363):
    """
    SFace-based staff face verification.
    Computes SFace embedding for the live image and compares it to stored staff photos via Cosine Distance.
    OpenCV's standard SFace threshold is 0.363 for Cosine matching.
    Returns (verified, similarity_pct, face_detected)
    """
    if not os.path.isdir(user_folder):
        return False, 0.0, False

    arr = np.frombuffer(live_image_bytes, np.uint8)
    live_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if live_img is None:
        return False, 0.0, False

    live_emb = get_sface_embedding(live_img)
    if live_emb is None:
        return False, 0.0, False

    # Extract reference embeddings on the fly (SFace is fast)
    image_files = sorted([f for f in os.listdir(user_folder)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        return False, 0.0, False
        
    best_score = 0.0
    for fn in image_files[:max_train]:
        img_path = os.path.join(user_folder, fn)
        ref_img = cv2.imread(img_path)
        if ref_img is None: continue
        ref_emb = get_sface_embedding(ref_img)
        if ref_emb is None: continue
        
        recognizer = get_sface_recognizer()
        score = recognizer.match(live_emb, ref_emb, cv2.FaceRecognizerSF_FR_COSINE)
        if score > best_score:
            best_score = score
            
    # OpenCV Zoo recommends threshold 0.363 for Cosine match.
    similarity_pct = max(0.0, min(100.0, best_score * 100.0))
    verified = best_score >= min_score
    
    return verified, similarity_pct, True





SFACE_MODEL_PATH = os.path.join(os.path.dirname(MODEL_PATH) or ".", "sface_student_embeddings.pkl")

# ---- Training function used in background ----
def train_model_background(dataset_dir, progress_callback=None):
    """
    dataset_dir/
        <roll_no>/
            img1.jpg
            img2.jpg
            ...
    Extracts 128-d SFace embeddings for each student and saves them to
    sface_student_embeddings.pkl.
    """
    student_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    total_students = max(1, len(student_dirs))
    processed = 0
    student_sample_counts = {}

    MAX_SAMPLES_PER_STUDENT = 10  # SFace is powerful, we only need ~10 good varied angles
    
    # Structure: {"roll_no": [emb1, emb2, ...]}
    db_embeddings = {}

    for sid in student_dirs:
        folder = os.path.join(dataset_dir, sid)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        if len(files) > MAX_SAMPLES_PER_STUDENT:
            step = len(files) // MAX_SAMPLES_PER_STUDENT
            files = files[::step][:MAX_SAMPLES_PER_STUDENT]
            
        count = 0
        embs = []
        for fn in files:
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is None:
                continue
                
            emb = get_sface_embedding(img)
            if emb is not None:
                embs.append(emb)
                count += 1
                
        if len(embs) > 0:
            db_embeddings[sid] = embs
            
        student_sample_counts[sid] = count
        processed += 1
        if progress_callback:
            pct = int((processed / total_students) * 80)
            progress_callback(pct, f"Extracting embeddings: {processed}/{total_students} students…")

    if len(db_embeddings) == 0:
        if progress_callback:
            progress_callback(100, "No valid faces found")
        return

    if progress_callback:
        progress_callback(85, "Saving SFace Database...")

    with open(SFACE_MODEL_PATH, "wb") as f:
        pickle.dump(db_embeddings, f)

    if progress_callback:
        total_samples = sum(student_sample_counts.values())
        progress_callback(100, f"Training complete! {total_students} student(s), {total_samples} faces.")

