import hashlib
import io
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

APP_TITLE = "FaceMatch — Facial Recognition"
DB_PATH = Path("faces.db")
MAX_UPLOAD_MB = 8
FACE_SIZE = (160, 160)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")

# --- Database ---------------------------------------------------------------
def connect_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL COLLATE NOCASE UNIQUE,
            face_crop BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

def list_people():
    with connect_db() as conn:
        return conn.execute(
            "SELECT id, name, created_at FROM people ORDER BY name COLLATE NOCASE"
        ).fetchall()

def get_training_rows():
    with connect_db() as conn:
        return conn.execute(
            "SELECT id, name, face_crop FROM people ORDER BY id"
        ).fetchall()

def insert_person(name, face_crop):
    with connect_db() as conn:
        conn.execute(
            "INSERT INTO people(name, face_crop, created_at) VALUES (?, ?, ?)",
            (name.strip(), sqlite3.Binary(face_crop), datetime.now(timezone.utc).isoformat())
        )
        conn.commit()

def delete_person(person_id):
    with connect_db() as conn:
        conn.execute("DELETE FROM people WHERE id = ?", (int(person_id),))
        conn.commit()

def delete_all_people():
    with connect_db() as conn:
        conn.execute("DELETE FROM people")
        conn.commit()

# --- Image processing -------------------------------------------------------
def load_image(uploaded_file):
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Image is too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")
    try:
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width * image.height > 20_000_000:
            raise ValueError("Image dimensions are too large. Please use an image under 20 megapixels.")
        return np.array(image)
    except (UnidentifiedImageError, OSError):
        raise ValueError("The uploaded file could not be read as an image.")

def detect_faces(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("OpenCV face detector could not be loaded.")
    boxes = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(45, 45)
    )
    # Sort from left to right for predictable labels
    return sorted([(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes],
                  key=lambda b: (b[0], b[1]))

def crop_face(rgb_image, box):
    x, y, w, h = box
    # Expand crop modestly to include facial context; clamp to image boundaries
    H, W = rgb_image.shape[:2]
    pad_x, pad_y = int(w * 0.18), int(h * 0.18)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(W, x + w + pad_x), min(H, y + h + pad_y)
    crop = rgb_image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    # Normalize contrast for somewhat more stable LBPH comparisons
    gray = cv2.equalizeHist(gray)
    return gray

def encode_crop(gray):
    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("Could not encode face crop.")
    return encoded.tobytes()

def decode_crop(blob):
    arr = np.frombuffer(blob, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return cv2.resize(gray, FACE_SIZE)

def build_recognizer():
    rows = get_training_rows()
    if not rows:
        return None, {}
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images, labels, label_to_name = [], [], {}
    for person_id, name, blob in rows:
        crop = decode_crop(blob)
        if crop is None:
            continue
        images.append(crop)
        labels.append(int(person_id))
        label_to_name[int(person_id)] = name
    if not images:
        return None, {}
    recognizer.train(images, np.asarray(labels, dtype=np.int32))
    return recognizer, label_to_name

def annotate_faces(rgb_image, boxes):
    out = rgb_image.copy()
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 180, 90), 3)
        cv2.putText(out, f"Face {idx}", (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 90), 2, cv2.LINE_AA)
    return out

# --- Admin access -----------------------------------------------------------
def get_admin_password():
    try:
        return str(st.secrets.get("ADMIN_PASSWORD", "")).strip()
    except Exception:
        return ""

def admin_unlocked():
    expected = get_admin_password()
    if not expected:
        st.warning(
            "Registration and deletion are disabled until you set `ADMIN_PASSWORD` "
            "in Streamlit Secrets. Recognition of existing enrolled profiles remains available."
        )
        return False
    with st.sidebar:
        st.subheader("Administrator")
        supplied = st.text_input("Admin password", type="password", key="admin_password")
    if supplied:
        # Hash before compare to avoid a direct plain-text comparison in app logic.
        good = hashlib.sha256(supplied.encode()).digest()
        want = hashlib.sha256(expected.encode()).digest()
        import hmac
        if hmac.compare_digest(good, want):
            st.sidebar.success("Admin access enabled for this session.")
            return True
        st.sidebar.error("Incorrect admin password.")
    return False

# --- UI ---------------------------------------------------------------------
st.title("🔎 FaceMatch")
st.caption("A local-demo facial detection and enrolled-face matching application")
st.info(
    "Privacy notice: only upload images you have permission to use. Facial crops are biometric data. "
    "This prototype stores face crops in a local SQLite database; do not use it for consequential "
    "decisions, surveillance, access control, or identity verification."
)

with st.sidebar:
    st.header("About")
    st.write("**Detection:** OpenCV Haar cascade")
    st.write("**Recognition:** OpenCV LBPH")
    st.write(f"**Upload limit:** {MAX_UPLOAD_MB} MB per image")
    st.caption("LBPH is a basic classroom/demo recognizer, not a modern or security-grade face model.")

# Render the admin password widget once; reuse the result across tabs.
admin_ok = admin_unlocked()

tab_detect, tab_register, tab_recognize, tab_manage = st.tabs(
    ["Face Detection", "Register Face", "Recognize Face", "Manage Database"]
)

with tab_detect:
    st.subheader("Detect faces in an uploaded image")
    detect_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], key="detect_upload"
    )
    if detect_file:
        try:
            image_rgb = load_image(detect_file)
            boxes = detect_faces(image_rgb)
            left, right = st.columns(2)
            with left:
                st.image(image_rgb, caption="Original image", use_container_width=True)
            with right:
                st.image(annotate_faces(image_rgb, boxes),
                         caption="Detected faces", use_container_width=True)
            if boxes:
                st.success(f"Detected {len(boxes)} face(s).")
            else:
                st.warning("No face was detected. Try a clear, front-facing image with good lighting.")
        except Exception as exc:
            st.error(str(exc))

with tab_register:
    st.subheader("Register an authorized person")
    can_admin = admin_ok
    st.warning("Registration stores a normalized grayscale face crop in SQLite, not the original photo.")
    with st.form("register_form", clear_on_submit=True):
        person_name = st.text_input("Person's display name", max_chars=80)
        reg_file = st.file_uploader(
            "Reference photograph (one person only)", type=["jpg", "jpeg", "png"], key="register_upload"
        )
        consent = st.checkbox(
            "I confirm that I have the person's permission to enroll this face and store its biometric crop."
        )
        submitted = st.form_submit_button("Register face", disabled=not can_admin)
    if submitted:
        if not can_admin:
            st.error("Administrator access is required.")
        elif not person_name.strip():
            st.error("Enter a display name.")
        elif not consent:
            st.error("Consent confirmation is required.")
        elif reg_file is None:
            st.error("Upload a reference photograph.")
        else:
            try:
                rgb = load_image(reg_file)
                boxes = detect_faces(rgb)
                if len(boxes) != 1:
                    st.error(f"Registration requires exactly one detected face; found {len(boxes)}.")
                else:
                    crop = crop_face(rgb, boxes[0])
                    insert_person(person_name.strip(), encode_crop(crop))
                    st.success(f"Registered '{person_name.strip()}'. The original uploaded file is not saved by this app.")
            except sqlite3.IntegrityError:
                st.error("That name is already registered. Use a unique display name.")
            except Exception as exc:
                st.error(str(exc))

with tab_recognize:
    st.subheader("Compare a photograph against enrolled faces")
    st.caption("This is a similarity estimate. Results can be wrong, especially with poor lighting, pose, image quality, or different cameras.")
    rec_file = st.file_uploader(
        "Upload a query image", type=["jpg", "jpeg", "png"], key="recognize_upload"
    )
    threshold = st.slider(
        "LBPH distance threshold (lower is stricter)", min_value=25, max_value=100,
        value=55, step=5,
        help="Starting point only. Calibrate with consented test images; the score is not a probability."
    )
    if rec_file:
        try:
            rgb = load_image(rec_file)
            boxes = detect_faces(rgb)
            st.image(annotate_faces(rgb, boxes), caption="Detected faces", use_container_width=True)
            if not boxes:
                st.warning("No face detected.")
            else:
                recognizer, label_to_name = build_recognizer()
                if recognizer is None:
                    st.warning("No registered faces yet. Register a person first.")
                else:
                    output = rgb.copy()
                    for i, box in enumerate(boxes, start=1):
                        crop = crop_face(rgb, box)
                        label, distance = recognizer.predict(crop)
                        x, y, w, h = box
                        if label in label_to_name and distance <= threshold:
                            result = f"{label_to_name[label]} (distance {distance:.1f})"
                            color = (0, 180, 90)
                        else:
                            result = f"Unknown (distance {distance:.1f})"
                            color = (220, 70, 50)
                        cv2.rectangle(output, (x, y), (x + w, y + h), color, 3)
                        cv2.putText(output, result[:48], (x, max(25, y - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                        st.write(f"**Face {i}:** {result}")
                    st.image(output, caption="Recognition result", use_container_width=True)
                    st.caption("Do not treat a name match as proof of identity. Check results manually.")
        except Exception as exc:
            st.error(str(exc))

with tab_manage:
    st.subheader("Enrolled profiles")
    profiles = list_people()
    if profiles:
        st.dataframe(
            [{"ID": row[0], "Name": row[1], "Created (UTC)": row[2]} for row in profiles],
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No profiles are registered.")

    st.divider()
    st.subheader("Delete biometric data")
    st.caption("Deleting a profile removes its stored face crop from this SQLite database.")
    can_delete = admin_ok
    if profiles:
        choices = {f"{row[1]} (ID {row[0]})": row[0] for row in profiles}
        selected = st.selectbox("Choose profile to delete", list(choices.keys()))
        confirm_one = st.checkbox("I understand this permanently deletes the selected profile.", key="confirm_one")
        if st.button("Delete selected profile", disabled=not (can_delete and confirm_one), type="secondary"):
            delete_person(choices[selected])
            st.success("Profile deleted.")
            st.rerun()

    confirm_all = st.checkbox("I understand this permanently deletes every enrolled profile.", key="confirm_all")
    if st.button("Delete ALL biometric profiles", disabled=not (can_delete and confirm_all), type="primary"):
        delete_all_people()
        st.success("All enrolled profiles deleted.")
        st.rerun()

st.divider()
st.caption(
    "Prototype notice: OpenCV Haar detection and LBPH recognition are intended for learning and demonstration. "
    "Performance is not validated for different demographic groups or real-world identity decisions."
)
