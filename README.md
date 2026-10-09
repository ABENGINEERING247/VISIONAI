# FaceMatch — Streamlit Facial Detection & Recognition

A Python/Streamlit educational prototype that detects faces in uploaded photographs and compares them with faces enrolled by an administrator. It uses OpenCV Haar-cascade face detection and OpenCV LBPH recognition, with SQLite storage.

> **Important limitation:** LBPH is a basic, traditional face-recognition method, not a modern embedding model and not security-grade. It can produce false matches and missed matches. Do not use this app for surveillance, consequential decisions, access control, attendance penalties, or proof of identity.

## Features

- Upload JPG/JPEG/PNG photographs for face detection.
- Enroll one authorized person per reference photograph.
- Compare uploaded images against enrolled face crops.
- SQLite database stores names, timestamps, and normalized grayscale face crops.
- Administrator password gate for registration and deletion.
- Consent confirmation before enrollment.
- Delete an individual profile or all biometric profiles.
- Upload size and image dimension limits.
- Original uploaded photographs are processed in memory and are not deliberately saved by the app.

## Repository structure

```text
facial-recognition-system/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

`faces.db` is created automatically when the app runs. **Do not commit `faces.db` or any biometric data to GitHub.**

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
ADMIN_PASSWORD = "replace-this-with-a-long-unique-password"
```

Then start the app:

```bash
streamlit run app.py
```

Keep `secrets.toml` private and never commit it.

## GitHub setup

1. Create a GitHub repository named `facial-recognition-system`.
2. Add `app.py`, `requirements.txt`, `README.md`, and `.gitignore`.
3. Keep the repository private if your project code or configuration needs to be restricted.
4. Never upload face photographs, `faces.db`, `.streamlit/secrets.toml`, or exports containing biometric data.

Suggested `.gitignore`:

```gitignore
.venv/
__pycache__/
*.py[cod]
.streamlit/secrets.toml
faces.db
*.db
.DS_Store
```

## Deploy to Streamlit Community Cloud

1. Push the four project files to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app** and select your repository, branch, and `app.py`.
4. In the app's **Settings → Secrets**, add:

   ```toml
   ADMIN_PASSWORD = "use-a-long-unique-password"
   ```

5. Deploy and wait for dependencies to install.
6. Open the deployed app and test detection with a non-sensitive sample image.
7. Register only faces whose owners have given informed permission.

### Deployment and persistence warning

Streamlit Community Cloud's local filesystem is not durable storage. The SQLite database may be lost when the app is restarted, rebuilt, or moved. This implementation is suitable for a demo, not a production biometric database. For a real deployment, use a properly secured managed database and encrypted private object storage, authentication/authorization, audit logs, retention policies, backups, and a documented deletion process. Do not expose a public enrollment/deletion interface without access controls.

The administrator password protects app actions, but it is not a complete user authentication system. For multi-user or public deployments, integrate a proper identity provider and role-based access control.

## How it works

1. OpenCV's Haar cascade detects face rectangles.
2. The face crop is converted to grayscale, resized to 160×160, and contrast-normalized.
3. The normalized crop is stored as a PNG byte string in SQLite.
4. On recognition, the app trains an LBPH recognizer from stored crops and predicts the closest enrolled label.
5. A configurable LBPH distance threshold decides whether to display a possible match or `Unknown`.

The LBPH distance is **not a probability or confidence percentage**. A lower distance generally indicates a closer match within this model, but the threshold must be evaluated using representative, consented images. The app does not implement liveness detection and can be fooled by photographs or screens.

## Privacy checklist

- Obtain informed consent before enrolling a person's face.
- Use only images you have permission to process.
- Limit access to the application and its database.
- Do not commit images, database files, secrets, or biometric exports to GitHub.
- Delete profiles when consent is withdrawn or the data is no longer needed.
- Tell users what is collected, why it is collected, where it is stored, and when it is deleted.
- Use this as a learning demo, not as the sole basis for identity-related decisions.
