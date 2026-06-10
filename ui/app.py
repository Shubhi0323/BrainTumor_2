"""
Brain Tumor Analysis Dashboard
================================
Streamlit UI for visualising pipeline outputs.

Fixes applied (2026-06-10):
  - Removed unused stage_uploaded_dicoms(), INPUT_FORMAT constant, is_processed()
  - Fixed discover_all_patients() to support NIfTI/BraTS (was DICOM-only)
  - Fixed render_image_safe() invalid width="stretch" argument
  - Moved _label() helper to module level (was re-defined inside sidebar loop)
  - Removed redundant job/job_status re-fetch after sidebar block
  - Removed unused col_r variable in Processing Status tab
  - Added "no jobs yet" message in Processing Status tab
"""
import os
import re
import json
import glob
import io
import zipfile
import subprocess
import time
import sys
import warnings
from datetime import datetime
import numpy as np
import streamlit as st
import pydicom
from PIL import Image
from PIL import ImageFile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Allow large images (segmentation overlays) without decompression-bomb errors.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
JOBS_DIR = os.path.join(OUTPUT_DIR, ".jobs")
UPLOADS_DIR = os.path.join(OUTPUT_DIR, "uploaded_data")
HITL_REVIEWS_PATH = os.path.join(JOBS_DIR, "hitl_reviews.json")
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


# ── Job tracking ──────────────────────────────────────────────
def job_path(patient_id):
    return os.path.join(JOBS_DIR, f"{patient_id}.json")


def get_job_status(patient_id):
    path = job_path(patient_id)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def set_job_status(patient_id, status, **extra):
    data = {
        "patient_id": patient_id,
        "status": status,
        "updated_at": datetime.now().isoformat(),
        **extra,
    }
    with open(job_path(patient_id), "w") as f:
        json.dump(data, f)


def get_all_jobs():
    """Return all job records sorted by patient ID."""
    jobs = []
    for f in sorted(glob.glob(os.path.join(JOBS_DIR, "*.json"))):
        try:
            with open(f) as fh:
                jobs.append(json.load(fh))
        except Exception:
            pass
    return jobs


# ── HITL review persistence ────────────────────────────────────
def load_hitl_reviews():
    if os.path.exists(HITL_REVIEWS_PATH):
        try:
            with open(HITL_REVIEWS_PATH) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def get_hitl_review(patient_id):
    return load_hitl_reviews().get(patient_id, {})


def save_hitl_review(patient_id, reviewer, approved, notes):
    reviews = load_hitl_reviews()
    reviews[patient_id] = {
        "patient_id": patient_id,
        "reviewer": (reviewer or "").strip(),
        "approved": bool(approved),
        "notes": (notes or "").strip(),
        "updated_at": datetime.now().isoformat(),
    }
    with open(HITL_REVIEWS_PATH, "w") as f:
        json.dump(reviews, f, indent=2)


# ── Pipeline launcher ──────────────────────────────────────────
def start_pipeline_job(patient_id, data_dir, input_format="nifti"):
    """Launch main.py as a detached background process."""
    run_data_dir = (data_dir or "").strip()
    run_fmt = (input_format or "nifti").lower()
    if not run_data_dir or not os.path.isdir(run_data_dir):
        print(f"[ERROR] Invalid data directory for {patient_id}: {run_data_dir}")
        return False

    log_path = os.path.join(JOBS_DIR, f"{patient_id}.log")
    main_py = os.path.join(PROJECT_ROOT, "main.py")
    run_env = os.environ.copy()
    existing_pp = run_env.get("PYTHONPATH", "")
    run_env["PYTHONPATH"] = (
        PROJECT_ROOT if not existing_pp else f"{PROJECT_ROOT}:{existing_pp}"
    )

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [
                "python", main_py,
                "--data_dir", run_data_dir,
                "--format", run_fmt,
                "--phase", "all",
                "--output_dir", OUTPUT_DIR,
                "--skip_hitl",
                "--patient_id", patient_id,
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=run_env,
        )

    set_job_status(
        patient_id, "running",
        started_at=datetime.now().isoformat(),
        log_file=log_path,
        pid=proc.pid,
        data_dir=run_data_dir,
        input_format=run_fmt,
    )
    return True


# ── DICOM ZIP staging ─────────────────────────────────────────
def _safe_token(text):
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
    return token.strip("._-") or "uploaded_patient"


def stage_dicom_zip(uploaded_zip, patient_id):
    """Extract and stage all readable DICOM files from an uploaded ZIP."""
    if uploaded_zip is None:
        return None, None, 0, 0, "Please upload a ZIP file first."

    normalized_patient = _safe_token(patient_id)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_root = os.path.join(UPLOADS_DIR, f"session_{stamp}")
    patient_root = os.path.join(data_root, normalized_patient)

    try:
        zf = zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue()))
    except Exception:
        return None, None, 0, 0, "Uploaded file is not a valid ZIP archive."

    valid_count = 0
    invalid_count = 0
    for idx, member in enumerate(sorted(zf.namelist()), start=1):
        if member.endswith("/"):
            continue
        try:
            blob = zf.read(member)
            if not blob:
                continue
            ds = pydicom.dcmread(io.BytesIO(blob), stop_before_pixels=True, force=True)
            series_uid  = _safe_token(str(getattr(ds, "SeriesInstanceUID", "series_unknown")))
            member_dir  = os.path.dirname(member).strip()
            series_hint = _safe_token(os.path.basename(member_dir)) if member_dir else ""
            instance_no = getattr(ds, "InstanceNumber", idx)
            series_name = f"series_{series_hint}_{series_uid}" if series_hint else f"series_{series_uid}"
            series_dir  = os.path.join(patient_root, series_name)
            os.makedirs(series_dir, exist_ok=True)
            orig_name = _safe_token(os.path.basename(member))
            try:
                instance_token = f"{int(instance_no):05d}"
            except Exception:
                instance_token = f"{idx:05d}"
            with open(os.path.join(series_dir, f"{instance_token}_{orig_name}.dcm"), "wb") as f:
                f.write(blob)
            valid_count += 1
        except Exception:
            invalid_count += 1

    if valid_count == 0:
        return None, None, 0, invalid_count, "No readable DICOM files found in ZIP."
    return data_root, normalized_patient, valid_count, invalid_count, ""


# ── Process liveness check ────────────────────────────────────
def _is_pid_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def check_running_jobs():
    """Update job statuses: detect completion or crash of running jobs."""
    for fpath in glob.glob(os.path.join(JOBS_DIR, "*.json")):
        try:
            with open(fpath) as fh:
                job = json.load(fh)
        except Exception:
            continue

        source_meta = {
            "data_dir": job.get("data_dir", ""),
            "input_format": job.get("input_format", "nifti"),
        }
        status      = job.get("status", "")
        patient_id  = job.get("patient_id", "")
        log_path    = job.get("log_file", "")
        proc_pid    = job.get("pid")
        report_path = os.path.join(OUTPUT_DIR, "reports", f"{patient_id}_report.json")

        # Auto-recover: status=failed but report now exists
        if status == "failed" and os.path.exists(report_path):
            report = load_json(report_path)
            errors = (report or {}).get("pipeline_errors", [])
            has_errors = (report or {}).get("report_metadata", {}).get("has_errors", False)
            new_status = "completed_with_errors" if (has_errors or errors) else "completed"
            set_job_status(patient_id, new_status,
                           started_at=job.get("started_at"),
                           finished_at=datetime.now().isoformat(),
                           log_file=log_path, errors=errors, **source_meta)
            continue

        if status != "running":
            continue

        if os.path.exists(report_path):
            report = load_json(report_path)
            errors = (report or {}).get("pipeline_errors", [])
            has_errors = (report or {}).get("report_metadata", {}).get("has_errors", False)
            new_status = "completed_with_errors" if (has_errors or errors) else "completed"
            set_job_status(patient_id, new_status,
                           started_at=job.get("started_at"),
                           finished_at=datetime.now().isoformat(),
                           log_file=log_path, errors=errors, **source_meta)
        elif proc_pid and not _is_pid_alive(proc_pid):
            set_job_status(patient_id, "failed",
                           started_at=job.get("started_at"),
                           finished_at=datetime.now().isoformat(),
                           log_file=log_path, **source_meta)
        elif log_path and os.path.exists(log_path):
            if time.time() - os.path.getmtime(log_path) > 600:
                set_job_status(patient_id, "failed",
                               started_at=job.get("started_at"),
                               finished_at=datetime.now().isoformat(),
                               log_file=log_path, **source_meta)


# ── Patient / data discovery ───────────────────────────────────
def discover_all_patients(format_name, data_dir):
    """Discover patient IDs from a dataset directory.

    - NIfTI/BraTS: each immediate sub-folder is a patient.
    - DICOM: uses dicom_adapter; falls back to sub-folder scan.
    """
    if not data_dir or not os.path.isdir(data_dir):
        return []

    patients = set()

    if format_name == "nifti":
        for name in sorted(os.listdir(data_dir)):
            if os.path.isdir(os.path.join(data_dir, name)) and not name.startswith("."):
                patients.add(name)
    else:  # dicom
        try:
            from utils.dicom_adapter import discover_dicom_series
            discovered = discover_dicom_series(data_dir)
            patients.update(discovered.keys())
        except Exception as e:
            print(f"[WARNING] DICOM discovery failed: {e}")
        if not patients:
            for name in sorted(os.listdir(data_dir)):
                if os.path.isdir(os.path.join(data_dir, name)) and not name.startswith("."):
                    patients.add(name)

    return sorted(patients)


def find_processed_patients():
    """Return set of patient IDs that have at least one pipeline output file."""
    processed = set()
    suffixes = {
        "_report.json": os.path.join(OUTPUT_DIR, "reports"),
        "_clinical.json": os.path.join(OUTPUT_DIR, "clinical_features"),
        "_radiomics.json": os.path.join(OUTPUT_DIR, "radiomics"),
    }
    for suffix, folder in suffixes.items():
        for f in glob.glob(os.path.join(folder, f"*{suffix}")):
            processed.add(os.path.basename(f).replace(suffix, ""))
    return processed


# ── Data loaders ───────────────────────────────────────────────
def load_json(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_report(patient_id):
    return load_json(os.path.join(OUTPUT_DIR, "reports", f"{patient_id}_report.json"))


def load_cap(patient_id):
    return load_json(os.path.join(OUTPUT_DIR, "reports", "cap", f"{patient_id}_cap_report.json"))


def load_clinical(patient_id):
    return load_json(os.path.join(OUTPUT_DIR, "clinical_features", f"{patient_id}_clinical.json"))


def load_radiomics(patient_id):
    return load_json(os.path.join(OUTPUT_DIR, "radiomics", f"{patient_id}_radiomics.json"))


def get_viz_images(patient_id):
    viz_dir = os.path.join(OUTPUT_DIR, "visualizations", patient_id)
    images = {}
    if os.path.isdir(viz_dir):
        for f in sorted(os.listdir(viz_dir)):
            if f.endswith(".png"):
                label = (
                    f.replace(f"{patient_id}_", "")
                     .replace(".png", "")
                     .replace("_", " ")
                     .title()
                )
                images[label] = os.path.join(viz_dir, f)
    return images


def get_patient_modalities(data_dir, patient_id):
    """Return {modality: (found, filename)} for a BraTS NIfTI patient folder."""
    MODALITY_PATTERNS = {
        "T1":    ["_t1.nii.gz",    "_t1.nii"],
        "T1ce":  ["_t1ce.nii.gz",  "_t1ce.nii"],
        "T2":    ["_t2.nii.gz",    "_t2.nii"],
        "FLAIR": ["_flair.nii.gz", "_flair.nii"],
        "Seg":   ["_seg.nii.gz",   "_seg.nii"],
    }
    patient_dir = os.path.join(data_dir, patient_id) if data_dir else ""
    result = {}
    for mod, patterns in MODALITY_PATTERNS.items():
        found_name = ""
        if patient_dir and os.path.isdir(patient_dir):
            for fname in os.listdir(patient_dir):
                if any(fname.endswith(p) for p in patterns):
                    found_name = fname
                    break
        result[mod] = (bool(found_name), found_name)
    return result, patient_dir


# ── Rendering helpers ─────────────────────────────────────────
def render_image_safe(path):
    """Display an image without crashing if PIL refuses to load it."""
    try:
        st.image(path, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not display image {os.path.basename(path)}: {e}")


def render_hitl_review_form(patient_id):
    review = get_hitl_review(patient_id)
    with st.expander("🧑‍⚕️ Final Human Review", expanded=(not bool(review))):
        reviewer = st.text_input(
            "Reviewer",
            value=review.get("reviewer", ""),
            key=f"hitl_reviewer_{patient_id}",
        )
        approved = st.checkbox(
            "Approved for final sign-off",
            value=bool(review.get("approved", False)),
            key=f"hitl_approved_{patient_id}",
        )
        notes = st.text_area(
            "Final review notes",
            value=review.get("notes", ""),
            height=100,
            key=f"hitl_notes_{patient_id}",
        )
        if st.button("Save Final Review", key=f"save_hitl_{patient_id}", use_container_width=True):
            save_hitl_review(patient_id, reviewer, approved, notes)
            st.success("Final review saved.")
            st.rerun()
        if review:
            status = "Approved" if review.get("approved") else "Pending approval"
            updated = review.get("updated_at", "")[:19]
            reviewer_name = review.get("reviewer", "Unassigned")
            st.caption(f"Latest review: {status} · Reviewer: {reviewer_name} · Updated: {updated}")


# ── Display helpers ───────────────────────────────────────────
def severity_badge(severity):
    colors = {
        "small":      "badge-green",
        "medium":     "badge-yellow",
        "large":      "badge-red",
        "very_large": "badge-red",
    }
    cls = colors.get(severity, "badge-blue")
    return f'<span class="status-badge {cls}">{severity.upper()}</span>'


def rano_badge(assessment):
    colors = {"CR": "badge-green", "PR": "badge-blue", "SD": "badge-yellow", "PD": "badge-red"}
    names  = {
        "CR": "Complete Response",
        "PR": "Partial Response",
        "SD": "Stable Disease",
        "PD": "Progressive Disease",
    }
    cls  = colors.get(assessment, "badge-blue")
    name = names.get(assessment, assessment)
    return f'<span class="status-badge {cls}">{assessment} — {name}</span>'


def metric_card(title, value):
    return (
        f'<div class="metric-card">'
        f'<h3>{title}</h3>'
        f'<p>{value}</p>'
        f'</div>'
    )


def _patient_label(pid, processed, all_jobs_by_pid):
    """Return a status-emoji-prefixed label for the patient dropdown."""
    job = all_jobs_by_pid.get(pid)
    job_status = job.get("status") if job else None
    if pid in processed and job_status == "completed_with_errors":
        return f"⚠️ {pid}"
    if pid in processed:
        return f"✅ {pid}"
    if job_status == "running":
        return f"⏳ {pid}"
    if job_status == "failed":
        return f"❌ {pid}"
    return f"○  {pid}"


# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroAgent — Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #3d3d5c;
        text-align: center;
    }
    .metric-card h3 {
        color: #a0a0c0;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }
    .metric-card p {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-green  { background: #1a4d2e; color: #4ade80; }
    .badge-yellow { background: #4d3d1a; color: #facc15; }
    .badge-red    { background: #4d1a1a; color: #f87171; }
    .badge-blue   { background: #1a2d4d; color: #60a5fa; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
check_running_jobs()

with st.sidebar:
    st.markdown("## 🧠 NeuroAgent")
    st.markdown("---")

    # ── Three-tab data source ─────────────────────────────────
    # Tab 1: BraTS / NIfTI local directory
    # Tab 2: DICOM local directory
    # Tab 3: DICOM ZIP upload (for Colab / remote)
    brats_tab, dicom_tab, zip_tab = st.tabs([
        "🧬 BraTS (NIfTI)",
        "🗂️ DICOM Dir",
        "📤 Upload ZIP",
    ])

    with brats_tab:
        default_dir = os.environ.get("DATA_DIR", "")
        brats_dir_input = st.text_input(
            "BraTS dataset path",
            value=st.session_state.get("brats_dir", default_dir),
            key="brats_dir_widget",
            help=(
                "Path to folder containing BraTS20_Training_XXX sub-folders. "
                "Each sub-folder = one patient (NIfTI format)."
            ),
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📂 Load", use_container_width=True, key="load_brats"):
                if brats_dir_input and os.path.isdir(brats_dir_input):
                    # Count sub-folders so we can warn immediately if empty
                    sub_dirs = [
                        n for n in os.listdir(brats_dir_input)
                        if os.path.isdir(os.path.join(brats_dir_input, n))
                        and not n.startswith(".")
                    ]
                    if sub_dirs:
                        st.session_state["active_data_dir"] = brats_dir_input
                        st.session_state["active_format"]   = "nifti"
                        st.session_state["brats_dir"]       = brats_dir_input
                        st.session_state.pop("staged_patient_id", None)
                        st.rerun()
                    else:
                        st.error(
                            "No patient sub-folders found in that directory. "
                            "Expected folders like `BraTS20_Training_001`."
                        )
                else:
                    st.error("Directory not found.")
        with col_b:
            if st.button("🔄 Reset", use_container_width=True, key="reset_brats"):
                for k in ["active_data_dir", "active_format", "brats_dir", "staged_patient_id"]:
                    st.session_state.pop(k, None)
                st.rerun()

        # Auto-tip when DATA_DIR env var is set
        if default_dir and not st.session_state.get("active_data_dir"):
            st.info(
                f"DATA_DIR env detected:\n`{default_dir}`\n\n"
                "Click **Load** to use it."
            )

    with dicom_tab:
        dicom_dir_input = st.text_input(
            "DICOM dataset path",
            value=st.session_state.get("dicom_dir", ""),
            key="dicom_dir_widget",
            help=(
                "Path to a folder containing patient sub-folders of DICOM (.dcm) files. "
                "Each immediate sub-folder = one patient."
            ),
        )
        col_c, col_d = st.columns(2)
        with col_c:
            if st.button("📂 Load", use_container_width=True, key="load_dicom"):
                if dicom_dir_input and os.path.isdir(dicom_dir_input):
                    sub_dirs = [
                        n for n in os.listdir(dicom_dir_input)
                        if os.path.isdir(os.path.join(dicom_dir_input, n))
                        and not n.startswith(".")
                    ]
                    if sub_dirs:
                        st.session_state["active_data_dir"] = dicom_dir_input
                        st.session_state["active_format"]   = "dicom"
                        st.session_state["dicom_dir"]       = dicom_dir_input
                        st.session_state.pop("staged_patient_id", None)
                        st.rerun()
                    else:
                        st.error(
                            "No patient sub-folders found. "
                            "Each patient should be a sub-folder containing .dcm files."
                        )
                else:
                    st.error("Directory not found.")
        with col_d:
            if st.button("🔄 Reset", use_container_width=True, key="reset_dicom"):
                for k in ["active_data_dir", "active_format", "dicom_dir", "staged_patient_id"]:
                    st.session_state.pop(k, None)
                st.rerun()

    with zip_tab:
        st.caption("Upload a single patient's DICOM folder as a ZIP file.")
        upload_patient_id = st.text_input(
            "Patient ID", value="uploaded_patient", key="upload_pid"
        )
        uploaded_zip = st.file_uploader(
            "DICOM folder (ZIP)",
            type=["zip"],
            key="zip_upload",
            help="Zip your DICOM series folder and upload here.",
        )
        if st.button("Stage ZIP", use_container_width=True, key="stage_zip"):
            staged_dir, staged_pid, ok, bad, err = stage_dicom_zip(
                uploaded_zip, upload_patient_id
            )
            if staged_dir and staged_pid:
                st.session_state["active_data_dir"]   = staged_dir
                st.session_state["active_format"]     = "dicom"
                st.session_state["staged_patient_id"] = staged_pid
                st.success(f"Staged {ok} DICOM file(s).")
                if bad:
                    st.warning(f"Skipped {bad} non-DICOM file(s).")
                st.rerun()
            else:
                st.error(err or "Could not stage ZIP.")

    # ── Resolve active settings ──────────────────────────────
    active_data_dir = st.session_state.get("active_data_dir", "")
    active_format   = st.session_state.get("active_format", "nifti")
    st.markdown("---")

    # ── Patient discovery ────────────────────────────────────
    all_patients = discover_all_patients(active_format, active_data_dir)
    staged_pid   = st.session_state.get("staged_patient_id")
    if staged_pid and staged_pid not in all_patients:
        all_patients = sorted(set(all_patients + [staged_pid]))
    processed = find_processed_patients()

    if not all_patients:
        if active_data_dir:
            st.warning("No patients found in that directory.")
            if active_format == "nifti":
                st.info(
                    "Expected BraTS sub-folder names like `BraTS20_Training_001`.\n\n"
                    "Make sure you selected the **training root folder** "
                    "(the one that contains the patient folders), not a patient folder itself."
                )
            else:
                st.info(
                    "Each patient must be an immediate sub-folder containing `.dcm` files.\n\n"
                    "If you have a single patient ZIP, use the **Upload ZIP** tab instead."
                )
        else:
            st.info(
                "Pick a data source above:\n"
                "- **BraTS (NIfTI)** — local BraTS2020 dataset\n"
                "- **DICOM Dir** — local folder of DICOM patients\n"
                "- **Upload ZIP** — upload a single patient's DICOM ZIP"
            )
        st.stop()

    # ── Searchable patient selector ──────────────────────────
    search_term = st.text_input(
        "🔍 Search patient", placeholder="Type to filter…", key="patient_search"
    )
    filtered = (
        [p for p in all_patients if search_term.lower() in p.lower()]
        if search_term else all_patients
    )
    if not filtered:
        st.warning("No patients match the search.")
        st.stop()

    # Build label list (single pass over jobs dict for efficiency)
    all_jobs_by_pid = {j["patient_id"]: j for j in get_all_jobs() if "patient_id" in j}
    labels = [_patient_label(p, processed, all_jobs_by_pid) for p in filtered]

    sel_label  = st.selectbox("Select Patient", labels, key="patient_selector")
    patient_id = sel_label.split(" ", 1)[1].strip()

    running_count = sum(1 for j in all_jobs_by_pid.values() if j.get("status") == "running")
    st.caption(
        f"{len(processed)}/{len(all_patients)} processed"
        + (f"  ·  {running_count} running" if running_count else "")
    )

    # ── Run / status controls ────────────────────────────────
    st.markdown("---")
    patient_processed = patient_id in processed
    job        = all_jobs_by_pid.get(patient_id)
    job_status = job.get("status") if job else None

    if job_status == "running":
        st.info("⏳ Analysis running…")
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_btn"):
            st.rerun()

    elif job_status == "failed" and not patient_processed:
        st.error("❌ Pipeline failed")
        log_path = (job or {}).get("log_file", "")
        if log_path and os.path.exists(log_path):
            with open(log_path) as lf:
                last = "".join(lf.readlines()[-8:])
            with st.expander("Last log lines"):
                st.code(last, language="text")
        if st.button("🔄 Retry Analysis", type="primary", use_container_width=True, key="retry_btn"):
            retry_dir = (job or {}).get("data_dir", active_data_dir)
            retry_fmt = (job or {}).get("input_format", active_format)
            if start_pipeline_job(patient_id, retry_dir, retry_fmt):
                st.success("Restarted!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Cannot retry: data directory not available.")

    elif not patient_processed:
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True, key="run_btn"):
            if active_data_dir and os.path.isdir(active_data_dir):
                if start_pipeline_job(patient_id, active_data_dir, active_format):
                    st.success("Analysis started!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to start pipeline.")
            else:
                st.error("Load a valid data directory first.")
    else:
        report_meta = (load_report(patient_id) or {}).get("report_metadata", {})
        if report_meta.get("has_errors", False):
            st.warning("⚠️ Completed with errors")
        else:
            st.success("✅ Analysis complete")
        st.caption(f"Generated: {report_meta.get('generated_at', '')[:19]}")
        if st.button("🔄 Reprocess", use_container_width=True, key="reprocess_btn"):
            if start_pipeline_job(patient_id, active_data_dir, active_format):
                st.rerun()
            else:
                st.error("Cannot reprocess: data directory not available.")


# ══════════════════════════════════════════════════════════════
# LOAD PATIENT DATA
# ══════════════════════════════════════════════════════════════
if patient_processed:
    report    = load_report(patient_id)
    cap       = load_cap(patient_id)
    clinical  = load_clinical(patient_id)
    radiomics = load_radiomics(patient_id)
    viz_images = get_viz_images(patient_id)
else:
    report = cap = clinical = radiomics = None
    viz_images = {}


# ══════════════════════════════════════════════════════════════
# MAIN CONTENT AREA
# ══════════════════════════════════════════════════════════════

if not patient_processed:
    # ── Pre-analysis states ───────────────────────────────────
    if job_status == "running":
        st.title(f"⏳ Analysing: {patient_id}")
        st.info(
            "The pipeline is running in the background. "
            "This page auto-refreshes every 5 seconds."
        )
        st.progress(0.0, text="Pipeline running…")
        log_path = (job or {}).get("log_file", "")
        if log_path and os.path.exists(log_path):
            with open(log_path) as lf:
                lines = lf.readlines()
            with st.expander("📋 Live Log (last 25 lines)", expanded=True):
                st.code(
                    "".join(lines[-25:]) if lines else "Waiting for output…",
                    language="text",
                )
        time.sleep(5)
        st.rerun()

    elif job_status == "failed":
        st.title(f"❌ Analysis Failed: {patient_id}")
        st.error(
            "The pipeline exited with an error. "
            "Check the log below, then click **Retry Analysis** in the sidebar."
        )
        log_path = (job or {}).get("log_file", "")
        if log_path and os.path.exists(log_path):
            with open(log_path) as lf:
                content = lf.read()
            with st.expander("📋 Full Error Log", expanded=True):
                st.code(content[-4000:], language="text")

    else:
        # ── Patient detail card ───────────────────────────────
        st.title(f"🧬 {patient_id}")
        st.caption(f"Format: {active_format.upper()}  ·  Directory: {active_data_dir}")
        st.markdown("---")

        st.markdown("### Available Modalities")
        modalities, patient_dir = get_patient_modalities(active_data_dir, patient_id)
        mod_cols = st.columns(len(modalities))
        for i, (mod_name, (found, fname)) in enumerate(modalities.items()):
            with mod_cols[i]:
                if found:
                    st.success(f"✅ **{mod_name}**")
                    if fname:
                        st.caption(fname)
                else:
                    st.error(f"❌ **{mod_name}**")
                    st.caption("Not found")

        st.markdown("---")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### What the Pipeline Will Do")
            for icon, phase, desc in [
                ("🔬", "Phase 1 — Imaging",
                 "N4 bias correction → SegResNet segmentation → PyRadiomics (214 features)"),
                ("🧠", "Phase 2 — Intelligence",
                 "WHO CNS5 classification → BioClinicalBERT embeddings → Similar case retrieval → Llama 3 reasoning"),
                ("📋", "Phase 3 — Clinical",
                 "RANO assessment → CAP structured report → Patient memory (ChromaDB)"),
            ]:
                st.markdown(f"**{icon} {phase}**")
                st.caption(desc)
                st.markdown("")

        with c2:
            st.markdown("### Folder Contents")
            if patient_dir and os.path.isdir(patient_dir):
                for fname in sorted(os.listdir(patient_dir)):
                    fpath = os.path.join(patient_dir, fname)
                    size_mb = os.path.getsize(fpath) / 1024 / 1024
                    st.text(f"📄 {fname}  ({size_mb:.1f} MB)")
            else:
                st.warning("Patient folder not found.")

        st.markdown("---")
        st.info("👈 Click **🚀 Run Analysis** in the sidebar to start the pipeline.")

else:
    # ══════════════════════════════════════════════════════════
    # RESULTS DASHBOARD
    # ══════════════════════════════════════════════════════════
    st.title(f"📊 Results: {patient_id}")

    (tab_overview, tab_mri, tab_rad, tab_class,
     tab_reason, tab_cap, tab_status) = st.tabs([
        "📊 Overview",
        "🧠 MRI Visualization",
        "📈 Radiomics",
        "🏷️ Classification",
        "🩺 Clinical Reasoning",
        "📄 CAP Report",
        "⚙️ Processing Status",
    ])

    # ── Overview ──────────────────────────────────────────────
    with tab_overview:
        if report:
            tumor = report.get("tumor_summary", {})
            morph = tumor.get("morphology", {})
            who   = report.get("who_classification", {})
            rano  = report.get("rano_assessment", {})

            vol     = morph.get("tumor_volume", "N/A")
            vol_str = f"{vol:,} mm³" if isinstance(vol, (int, float)) else f"{vol} mm³"
            cols = st.columns(5)
            with cols[0]:
                st.markdown(metric_card("Volume", vol_str), unsafe_allow_html=True)
            with cols[1]:
                diam = morph.get("max_diameter", 0)
                st.markdown(metric_card("Max Diameter", f"{float(diam):.1f} mm"), unsafe_allow_html=True)
            with cols[2]:
                sph = morph.get("sphericity", 0)
                st.markdown(metric_card("Sphericity", f"{float(sph):.3f}"), unsafe_allow_html=True)
            with cols[3]:
                st.markdown(metric_card("WHO Grade", who.get("who_grade", "N/A")), unsafe_allow_html=True)
            with cols[4]:
                conf = who.get("confidence", 0)
                st.markdown(metric_card("Confidence", f"{float(conf):.0%}"), unsafe_allow_html=True)

            st.markdown("")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Tumor Classification")
                st.markdown(f"**Type:** {who.get('full_name', 'N/A')}")
                st.markdown(
                    f"**Severity:** {severity_badge(tumor.get('volume_severity', 'unknown'))}",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Location:** {', '.join(tumor.get('location', ['unknown']))}")
                st.markdown(f"**Prognosis:** {who.get('prognosis', 'N/A')}")

            with col2:
                st.markdown("### Response Assessment")
                st.markdown(
                    f"**RANO:** {rano_badge(rano.get('assessment', 'N/A'))}",
                    unsafe_allow_html=True,
                )
                for r in rano.get("reasoning", []):
                    st.markdown(f"- {r}")
                prog = report.get("tumor_progression", {})
                st.markdown(f"**Progression:** {prog.get('state', 'unknown').title()}")
                if prog.get("growth_rate") is not None:
                    st.markdown(f"**Growth Rate:** {prog['growth_rate']:.2f} mm³/day")

            st.markdown("### Inferred Symptoms")
            symptoms = tumor.get("inferred_symptoms", [])
            if symptoms:
                scols = st.columns(min(len(symptoms), 4))
                for i, s in enumerate(symptoms):
                    with scols[i % len(scols)]:
                        st.info(s.title())
            else:
                st.markdown("_No symptoms inferred_")

            diff = who.get("differential_diagnosis", [])
            if diff:
                st.markdown("### Differential Diagnosis")
                for d in diff:
                    with st.expander(f"{d['type'].replace('_', ' ').title()} (score: {d['score']})"):
                        for r in d.get("reasoning", []):
                            st.markdown(f"- {r}")
        elif clinical:
            st.info("Full report not available — showing clinical profile only.")
            st.json(clinical)
        else:
            st.warning("No data available for this patient.")

    # ── MRI Visualization ─────────────────────────────────────
    with tab_mri:
        st.markdown("## MRI Visualization")
        if viz_images:
            primary   = {k: v for k, v in viz_images.items() if k in ["Tumor Overlay", "Mri Slices"]}
            secondary = {k: v for k, v in viz_images.items() if k not in primary}
            if primary:
                cols = st.columns(len(primary))
                for i, (label, path) in enumerate(primary.items()):
                    with cols[i]:
                        st.markdown(f"#### {label}")
                        render_image_safe(path)
            if secondary:
                st.markdown("---")
                cols = st.columns(min(len(secondary), 3))
                for i, (label, path) in enumerate(secondary.items()):
                    with cols[i % 3]:
                        st.markdown(f"#### {label}")
                        render_image_safe(path)
        else:
            st.warning("No visualisation images found. Run the pipeline to generate them.")

        seg_path  = os.path.join(OUTPUT_DIR, "segmentation", f"{patient_id}_seg.nii.gz")
        prep_path = os.path.join(OUTPUT_DIR, "preprocessed", f"{patient_id}_preprocessed.npy")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Output Files")
            st.markdown(f"- Segmentation mask: {'✅' if os.path.exists(seg_path) else '❌'}")
            st.markdown(f"- Preprocessed volume: {'✅' if os.path.exists(prep_path) else '❌'}")
        with col2:
            if os.path.exists(prep_path):
                vol_arr = np.load(prep_path)
                st.markdown("#### Volume Info")
                st.markdown(f"- Shape: `{vol_arr.shape}`")
                st.markdown(f"- Dtype: `{vol_arr.dtype}`")
                st.markdown(f"- Range: `[{vol_arr.min():.3f}, {vol_arr.max():.3f}]`")

    # ── Radiomics ─────────────────────────────────────────────
    with tab_rad:
        st.markdown("## Radiomics Features")
        if radiomics:
            shape_f     = {k: v for k, v in radiomics.items() if k.startswith("shape_")}
            intensity_f = {k: v for k, v in radiomics.items() if k.startswith("intensity_")}
            texture_f   = {
                k: v for k, v in radiomics.items()
                if not k.startswith("shape_") and not k.startswith("intensity_")
            }
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Shape Features")
                for k, v in shape_f.items():
                    label = k.replace("shape_", "").replace("_", " ").title()
                    st.metric(label, f"{v:,.4f}" if isinstance(v, float) else f"{v:,}")
            with col2:
                st.markdown("### Intensity Features")
                for k, v in intensity_f.items():
                    label = k.replace("intensity_", "").replace("_", " ").title()
                    st.metric(label, f"{v:.6f}")
            if texture_f:
                st.markdown("### Texture Features (GLCM)")
                st.json(texture_f)
            st.markdown("### Intensity Distribution")
            chart_keys = ("intensity_mean", "intensity_std", "intensity_min",
                          "intensity_max", "intensity_median")
            chart_data = {k.replace("intensity_", ""): v
                          for k, v in intensity_f.items() if k in chart_keys}
            if chart_data:
                st.bar_chart(chart_data)
        else:
            st.warning("No radiomics data found. Run the pipeline to generate features.")

    # ── Classification ────────────────────────────────────────
    with tab_class:
        st.markdown("## Tumor Classification")
        if report:
            who = report.get("who_classification", {})
            st.markdown(f"## {who.get('full_name', 'Unknown')}")
            st.markdown(
                f"**WHO Grade:** {who.get('who_grade', 'N/A')}  ·  "
                f"**Confidence:** {who.get('confidence', 0):.0%}  ·  "
                f"**Type:** `{who.get('classified_as', 'N/A')}`"
            )
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Description")
                st.markdown(who.get("description", "N/A"))
                st.markdown("### Reasoning")
                for r in who.get("reasoning", []):
                    st.markdown(f"- {r}")
            with col2:
                st.markdown("### Prognosis")
                st.info(who.get("prognosis", "N/A"))
                st.markdown("### Standard Treatment")
                st.success(who.get("standard_treatment", "N/A"))

            st.markdown("---")
            rano = report.get("rano_assessment", {})
            st.markdown("## RANO Assessment")
            st.markdown(rano_badge(rano.get("assessment", "N/A")), unsafe_allow_html=True)
            for r in rano.get("reasoning", []):
                st.markdown(f"- {r}")

            st.markdown("---")
            similar = report.get("similar_cases", [])
            st.markdown(f"## Similar Cases ({len(similar)} retrieved)")
            if similar:
                for case in similar:
                    with st.expander(
                        f"Patient {case.get('patient_id', '?')} — "
                        f"Similarity: {case.get('similarity_score', 0):.2%}"
                    ):
                        st.markdown(f"- **Location:** {', '.join(case.get('tumor_location', []))}")
                        st.markdown(f"- **Severity:** {case.get('volume_severity', 'N/A')}")
            else:
                st.markdown("_No similar cases retrieved._")
        else:
            st.warning("No report data found for this patient.")

    # ── Clinical Reasoning ────────────────────────────────────
    with tab_reason:
        st.markdown("## AI Clinical Reasoning")
        if report:
            reasoning_text = report.get("ai_clinical_reasoning", "")
            if reasoning_text:
                section_map = {
                    "DIAGNOSIS":      ("🔬", st.markdown),
                    "TREATMENT":      ("💊", st.success),
                    "PROGNOSIS":      ("📊", st.info),
                    "PROGRESSION":    ("📈", st.markdown),
                    "RANO":           ("📋", st.markdown),
                    "SYMPTOM":        ("🩺", st.warning),
                    "RECOMMENDATION": ("✅", st.success),
                }
                for block in reasoning_text.split("\n\n"):
                    if ":" in block:
                        title, body = block.split(":", 1)
                        title = title.strip()
                        body  = body.strip()
                        matched = False
                        for kw, (icon, fn) in section_map.items():
                            if kw in title.upper():
                                st.markdown(f"### {icon} {title}")
                                fn(body)
                                matched = True
                                break
                        if not matched:
                            st.markdown(f"### {title}")
                            st.markdown(body)
                    else:
                        st.markdown(block)
            else:
                st.markdown("_No clinical reasoning generated._")

            errors = report.get("pipeline_errors", [])
            if errors:
                st.markdown("---")
                st.markdown("### ⚠️ Pipeline Errors")
                for e in errors:
                    st.error(e)
        else:
            st.warning("No report data found.")

    # ── CAP Report ────────────────────────────────────────────
    with tab_cap:
        st.markdown("## CAP Structured Pathology Report")
        if cap:
            section_names = {
                "section_1_patient_information":    "📋 Patient Information",
                "section_2_mri_study_information":  "🔬 MRI Study Information",
                "section_3_tumor_characteristics":  "🧠 Tumor Characteristics",
                "section_4_radiomics_summary":      "📊 Radiomics Summary",
                "section_5_rano_classification":    "📈 RANO Classification",
                "section_6_who_classification":     "🏥 WHO Classification",
                "section_7_similar_tumor_cases":    "🔍 Similar Tumor Cases",
                "section_8_clinical_interpretation":"💡 Clinical Interpretation",
                "section_9_physician_notes":        "👨‍⚕️ Physician Notes",
            }
            for key, title in section_names.items():
                data = cap.get(key, {})
                if not data:
                    continue
                expanded = key in ("section_1_patient_information",
                                   "section_3_tumor_characteristics")
                with st.expander(title, expanded=expanded):
                    if isinstance(data, dict):
                        for k, v in data.items():
                            lbl = k.replace("_", " ").title()
                            if isinstance(v, list):
                                st.markdown(f"**{lbl}:**")
                                for item in v:
                                    if isinstance(item, dict):
                                        st.json(item)
                                    else:
                                        st.markdown(f"  - {item}")
                            elif isinstance(v, dict):
                                st.markdown(f"**{lbl}:**")
                                st.json(v)
                            elif isinstance(v, float):
                                st.markdown(f"**{lbl}:** {v:,.4f}")
                            elif isinstance(v, bool):
                                st.markdown(f"**{lbl}:** {'Yes' if v else 'No'}")
                            else:
                                st.markdown(f"**{lbl}:** {v}")
                    else:
                        st.write(data)
                    if key == "section_9_physician_notes":
                        st.markdown("---")
                        render_hitl_review_form(patient_id)
        else:
            st.warning("No CAP report found. Run the full pipeline (Phase 3) to generate it.")

    # ── Processing Status ─────────────────────────────────────
    with tab_status:
        st.markdown("## Processing Status")
        if st.button("🔄 Refresh", use_container_width=False, key="status_refresh"):
            st.rerun()

        jobs           = get_all_jobs()
        running_jobs   = [j for j in jobs if j.get("status") == "running"]
        completed_jobs = [j for j in jobs if j.get("status") == "completed"]
        error_jobs     = [j for j in jobs if j.get("status") in ("completed_with_errors", "failed")]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Jobs",    len(jobs))
        c2.metric("Running",       len(running_jobs))
        c3.metric("Completed",     len(completed_jobs))
        c4.metric("Errors/Failed", len(error_jobs))

        if not jobs:
            st.info("No processing jobs yet. Select a patient and click **🚀 Run Analysis**.")

        if running_jobs:
            st.markdown("### ⏳ Currently Running")
            for j in running_jobs:
                with st.container(border=True):
                    a, b = st.columns([3, 1])
                    with a:
                        st.markdown(f"**{j['patient_id']}**")
                        st.caption(f"Started: {j.get('started_at', '')[:19]}")
                    with b:
                        lp = j.get("log_file", "")
                        if lp and os.path.exists(lp):
                            with open(lp) as lf:
                                tail = "".join(lf.readlines()[-5:])
                            with st.expander("Live log"):
                                st.code(tail)

        if error_jobs:
            st.markdown("### ⚠️ Errors & Failures")
            for j in error_jobs:
                with st.container(border=True):
                    a, b = st.columns([3, 1])
                    with a:
                        lbl = ("Completed with errors"
                               if j["status"] == "completed_with_errors" else "Failed")
                        st.markdown(f"**{j['patient_id']}** — {lbl}")
                        for e in (j.get("errors") or [])[:3]:
                            st.warning(e)
                        st.caption(f"Finished: {j.get('finished_at', '')[:19]}")
                    with b:
                        if st.button("🔄 Reprocess",
                                     key=f"err_reprocess_{j['patient_id']}",
                                     use_container_width=True):
                            if start_pipeline_job(j["patient_id"],
                                                  j.get("data_dir", ""),
                                                  j.get("input_format", "nifti")):
                                st.rerun()
                        lp = j.get("log_file", "")
                        if lp and os.path.exists(lp):
                            with st.expander("Full log"):
                                with open(lp) as lf:
                                    st.code(lf.read()[-3000:])

        if completed_jobs:
            st.markdown("### ✅ Completed")
            for j in completed_jobs:
                with st.container(border=True):
                    a, b = st.columns([3, 1])
                    with a:
                        st.markdown(f"**{j['patient_id']}**")
                        st.caption(
                            f"Started: {j.get('started_at', '')[:19]}  ·  "
                            f"Finished: {j.get('finished_at', '')[:19]}"
                        )
                    with b:
                        if st.button("🔄 Reprocess",
                                     key=f"done_reprocess_{j['patient_id']}",
                                     use_container_width=True):
                            if start_pipeline_job(j["patient_id"],
                                                  j.get("data_dir", ""),
                                                  j.get("input_format", "nifti")):
                                st.rerun()
                            else:
                                st.info("Data directory no longer available.")
