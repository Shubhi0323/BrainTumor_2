"""
Brain Tumor Analysis Dashboard
================================
Streamlit UI for visualizing pipeline outputs.
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

# Output images are generated locally by this app; allow large images to render
# instead of triggering PIL decompression-bomb protection.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
JOBS_DIR = os.path.join(OUTPUT_DIR, ".jobs")
UPLOADS_DIR = os.path.join(OUTPUT_DIR, "uploaded_data")
HITL_REVIEWS_PATH = os.path.join(JOBS_DIR, "hitl_reviews.json")
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


# ── Job tracking ─────────────────────────────────────────────
def job_path(patient_id):
    return os.path.join(JOBS_DIR, f"{patient_id}.json")


def get_job_status(patient_id):
    path = job_path(patient_id)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def set_job_status(patient_id, status, **extra):
    data = {"patient_id": patient_id, "status": status, "updated_at": datetime.now().isoformat(), **extra}
    with open(job_path(patient_id), "w") as f:
        json.dump(data, f)


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


def start_pipeline_job(patient_id, data_dir, input_format="dicom"):
    """Launch pipeline as a background process."""
    run_data_dir = (data_dir or "").strip()
    run_input_format = (input_format or "dicom").lower()
    if not run_data_dir or not os.path.isdir(run_data_dir):
        print(f"[ERROR] Invalid data directory for {patient_id}: {run_data_dir}")
        return False

    log_path = os.path.join(JOBS_DIR, f"{patient_id}.log")
    set_job_status(patient_id, "running", started_at=datetime.now().isoformat(),
                   log_file=log_path, data_dir=run_data_dir, input_format=run_input_format)
    main_py = os.path.join(PROJECT_ROOT, "main.py")
    run_env = os.environ.copy()
    existing_pythonpath = run_env.get("PYTHONPATH", "")
    run_env["PYTHONPATH"] = PROJECT_ROOT if not existing_pythonpath else f"{PROJECT_ROOT}:{existing_pythonpath}"
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [
                "python", main_py,
                "--data_dir", run_data_dir,
                "--format", run_input_format,
                "--phase", "all",
                "--output_dir", OUTPUT_DIR,
                "--skip_hitl",
                "--patient_id", patient_id,
            ],
            stdout=log_file, stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=run_env,
        )
    # Store PID for process liveness checking
    set_job_status(patient_id, "running",
                   started_at=datetime.now().isoformat(),
                   log_file=log_path, pid=proc.pid,
                   data_dir=run_data_dir, input_format=run_input_format)
    return True


def _safe_token(text):
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
    return token.strip("._-") or "uploaded_patient"


def stage_uploaded_dicoms(uploaded_files, patient_id):
    """Persist uploaded DICOM files into a temporary dataset layout."""
    if not uploaded_files:
        return None, None, 0, 0

    normalized_patient = _safe_token(patient_id)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_root = os.path.join(UPLOADS_DIR, f"session_{stamp}")
    patient_root = os.path.join(data_root, normalized_patient)

    valid_count = 0
    invalid_count = 0
    for idx, up in enumerate(uploaded_files, start=1):
        blob = up.getvalue()
        try:
            ds = pydicom.dcmread(io.BytesIO(blob), stop_before_pixels=True, force=True)
            series_uid = _safe_token(str(getattr(ds, "SeriesInstanceUID", "series_unknown")))
            instance_no = getattr(ds, "InstanceNumber", idx)
            series_dir = os.path.join(patient_root, f"series_{series_uid}")
            os.makedirs(series_dir, exist_ok=True)

            orig_name = _safe_token(os.path.basename(up.name))
            try:
                instance_token = f"{int(instance_no):05d}"
            except Exception:
                instance_token = f"{idx:05d}"
            fname = f"{instance_token}_{orig_name}.dcm"
            with open(os.path.join(series_dir, fname), "wb") as f:
                f.write(blob)
            valid_count += 1
        except Exception:
            invalid_count += 1

    if valid_count == 0:
        return None, None, 0, invalid_count

    return data_root, normalized_patient, valid_count, invalid_count


def stage_dicom_zip(uploaded_zip, patient_id):
    """Stage all readable DICOM files from an uploaded ZIP folder export."""
    if uploaded_zip is None:
        return None, None, 0, 0, "Please upload a ZIP file first."

    normalized_patient = _safe_token(patient_id)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_root = os.path.join(UPLOADS_DIR, f"session_{stamp}")
    patient_root = os.path.join(data_root, normalized_patient)

    valid_count = 0
    invalid_count = 0
    try:
        zf = zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue()))
    except Exception:
        return None, None, 0, 0, "Uploaded file is not a valid ZIP archive."

    for idx, member in enumerate(sorted(zf.namelist()), start=1):
        if member.endswith("/"):
            continue
        try:
            blob = zf.read(member)
            if not blob:
                continue

            ds = pydicom.dcmread(io.BytesIO(blob), stop_before_pixels=True, force=True)
            series_uid = _safe_token(str(getattr(ds, "SeriesInstanceUID", "series_unknown")))
            member_dir = os.path.dirname(member).strip()
            series_hint = _safe_token(os.path.basename(member_dir)) if member_dir else ""
            instance_no = getattr(ds, "InstanceNumber", idx)
            series_name = f"series_{series_hint}_{series_uid}" if series_hint else f"series_{series_uid}"
            series_dir = os.path.join(patient_root, series_name)
            os.makedirs(series_dir, exist_ok=True)

            orig_name = _safe_token(os.path.basename(member))
            try:
                instance_token = f"{int(instance_no):05d}"
            except Exception:
                instance_token = f"{idx:05d}"
            dst_path = os.path.join(series_dir, f"{instance_token}_{orig_name}.dcm")
            with open(dst_path, "wb") as f:
                f.write(blob)
            valid_count += 1
        except Exception:
            invalid_count += 1

    if valid_count == 0:
        return None, None, 0, invalid_count, "No readable DICOM files found in ZIP."

    return data_root, normalized_patient, valid_count, invalid_count, ""


def _is_pid_alive(pid):
    """Check if a process with given PID is still running."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def check_running_jobs():
    """Check if running jobs have completed by looking for output files or process exit."""
    for f in glob.glob(os.path.join(JOBS_DIR, "*.json")):
        try:
            with open(f) as fh:
                job = json.load(fh)
        except Exception:
            continue
        source_meta = {
            "data_dir": job.get("data_dir", ""),
            "input_format": job.get("input_format", "dicom"),
        }
        status = job.get("status", "")
        pid_val = job.get("patient_id", "")
        log_path = job.get("log_file", "")
        proc_pid = job.get("pid")
        report_path = os.path.join(OUTPUT_DIR, "reports", f"{pid_val}_report.json")

        # Recovery: if status is "failed" but report now exists, fix the status
        if status == "failed" and os.path.exists(report_path):
            report = load_json(report_path)
            has_errors = report.get("report_metadata", {}).get("has_errors", False) if report else False
            errors = report.get("pipeline_errors", []) if report else []
            if has_errors or errors:
                set_job_status(pid_val, "completed_with_errors",
                               started_at=job.get("started_at"),
                               finished_at=datetime.now().isoformat(),
                               log_file=log_path, errors=errors, **source_meta)
            else:
                set_job_status(pid_val, "completed",
                               started_at=job.get("started_at"),
                               finished_at=datetime.now().isoformat(),
                               log_file=log_path, **source_meta)
            continue

        if status != "running":
            continue

        # Check if report was generated (pipeline completed)
        if os.path.exists(report_path):
            report = load_json(report_path)
            has_errors = report.get("report_metadata", {}).get("has_errors", False) if report else False
            errors = report.get("pipeline_errors", []) if report else []
            if has_errors or errors:
                set_job_status(pid_val, "completed_with_errors",
                               started_at=job.get("started_at"),
                               finished_at=datetime.now().isoformat(),
                               log_file=log_path,
                               errors=errors,
                               **source_meta)
            else:
                set_job_status(pid_val, "completed",
                               started_at=job.get("started_at"),
                               finished_at=datetime.now().isoformat(),
                               log_file=log_path,
                               **source_meta)
        elif proc_pid and not _is_pid_alive(proc_pid):
            # Process exited without producing a report = failed
            set_job_status(pid_val, "failed",
                           started_at=job.get("started_at"),
                           finished_at=datetime.now().isoformat(),
                           log_file=log_path,
                           **source_meta)
        elif log_path and os.path.exists(log_path):
            # Fallback: check log staleness (no PID available)
            mtime = os.path.getmtime(log_path)
            if time.time() - mtime > 600:
                # Log hasn't been updated in 10 min — likely failed
                set_job_status(pid_val, "failed",
                               started_at=job.get("started_at"),
                               finished_at=datetime.now().isoformat(),
                               log_file=log_path,
                               **source_meta)


def get_all_jobs():
    """Return all job statuses."""
    jobs = []
    for f in sorted(glob.glob(os.path.join(JOBS_DIR, "*.json"))):
        try:
            with open(f) as fh:
                jobs.append(json.load(fh))
        except Exception:
            pass
    return jobs


def _find_brats_patient_root(data_dir):
    """Return the directory that directly contains BraTS patient folders.

    Handles both flat layouts and Kaggle's nested layout:
      flat:   <data_dir>/BraTS20_Training_001/  <-- data_dir returned as-is
      Kaggle: <data_dir>/BraTS2020_TrainingData/MICCAI_BraTS2020_Tr.../BraTS20_Training_001/
    """
    def _list_subdirs(path):
        try:
            return [
                n for n in os.listdir(path)
                if os.path.isdir(os.path.join(path, n)) and not n.startswith(".")
            ]
        except Exception:
            return []

    # Check if patient folders exist directly at data_dir
    direct_children = _list_subdirs(data_dir)
    brats_direct = [n for n in direct_children if n.startswith("BraTS")]
    if brats_direct:
        return data_dir

    # Walk one level deeper (handles Kaggle top-level wrapper like BraTS2020_TrainingData/)
    for child in direct_children:
        child_path = os.path.join(data_dir, child)
        grandchildren = _list_subdirs(child_path)
        brats_nested = [n for n in grandchildren if n.startswith("BraTS")]
        if brats_nested:
            return child_path

        # Two levels deep (BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_XXX)
        for gc in grandchildren:
            gc_path = os.path.join(child_path, gc)
            ggchildren = _list_subdirs(gc_path)
            brats_deep = [n for n in ggchildren if n.startswith("BraTS")]
            if brats_deep:
                return gc_path

    # Fallback: return original directory even if no BraTS folders detected
    return data_dir


def discover_all_patients(format_name, data_dir):
    """Discover all patients from the selected input format.

    Supports both NIfTI/BraTS directories (sub-folder per patient) and
    DICOM directories (via dicom_adapter with sub-folder fallback).
    For BraTS / NIfTI, automatically handles Kaggle's nested folder structure.
    """
    patients = set()

    if not data_dir or not os.path.isdir(data_dir):
        return sorted(patients)

    if format_name == "nifti":
        # Auto-resolve Kaggle nested layout to the real patient root
        patient_root = _find_brats_patient_root(data_dir)
        for name in sorted(os.listdir(patient_root)):
            if os.path.isdir(os.path.join(patient_root, name)) and not name.startswith("."):
                patients.add(name)
    else:
        # DICOM: try adapter first, fall back to sub-folder scan
        try:
            from utils.dicom_adapter import discover_dicom_series
            discovered = discover_dicom_series(data_dir)
            patients.update(discovered.keys())
        except Exception as e:
            print(f"[WARNING] DICOM discovery failed: {e}")

        # Fallback for staged uploads: include patient folders even when
        # modality inference cannot classify all series yet.
        if not patients:
            for name in sorted(os.listdir(data_dir)):
                full = os.path.join(data_dir, name)
                if os.path.isdir(full) and not name.startswith("."):
                    patients.add(name)

    return sorted(patients)


def find_processed_patients():
    """Find patients that have pipeline outputs."""
    processed = set()
    for pattern_path in [
        os.path.join(OUTPUT_DIR, "reports", "*_report.json"),
        os.path.join(OUTPUT_DIR, "clinical_features", "*_clinical.json"),
        os.path.join(OUTPUT_DIR, "radiomics", "*_radiomics.json"),
    ]:
        for f in glob.glob(pattern_path):
            name = os.path.basename(f)
            for suffix in ("_report.json", "_clinical.json", "_radiomics.json"):
                if name.endswith(suffix):
                    processed.add(name.replace(suffix, ""))
    return processed


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
                label = f.replace(f"{patient_id}_", "").replace(".png", "").replace("_", " ").title()
                images[label] = os.path.join(viz_dir, f)
    return images


def render_image_safe(path):
    """Render image without crashing the page on PIL size guard errors."""
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


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Sidebar background ── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #111827 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption p {
        color: #9ca3af !important;
    }
    [data-testid="stSidebar"] .stMarkdown strong { color: #e5e7eb !important; }

    /* ── Sidebar title ── */
    .sb-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f9fafb;
        letter-spacing: -0.01em;
        margin-bottom: 0;
        padding: 0.25rem 0;
    }

    /* ── Metric cards (main area) ── */
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

    /* ── Status badges ── */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-green  { background: #052e16; color: #4ade80; }
    .badge-yellow { background: #422006; color: #facc15; }
    .badge-red    { background: #450a0a; color: #f87171; }
    .badge-blue   { background: #0c1a2e; color: #60a5fa; }

    /* ── Pipeline status boxes ── */
    .pipe-success {
        background: rgba(74,222,128,0.08);
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        color: #4ade80;
        font-weight: 600;
        font-size: 0.88rem;
        text-align: center;
        margin: 0.4rem 0;
    }
    .pipe-error {
        background: rgba(248,113,113,0.08);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        color: #f87171;
        font-weight: 600;
        font-size: 0.88rem;
        text-align: center;
        margin: 0.4rem 0;
    }
    .pipe-running {
        background: rgba(96,165,250,0.08);
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        color: #60a5fa;
        font-weight: 600;
        font-size: 0.88rem;
        text-align: center;
        margin: 0.4rem 0;
    }
    .pipe-pending {
        background: rgba(250,204,21,0.07);
        border: 1px solid #eab308;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        color: #facc15;
        font-weight: 600;
        font-size: 0.88rem;
        text-align: center;
        margin: 0.4rem 0;
    }

    /* ── Nav radio buttons — red filled dot ── */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #d1d5db !important;
        font-size: 0.92rem !important;
        padding: 0.15rem 0 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        color: #f9fafb !important;
    }
    /* unselected ring */
    [data-testid="stSidebar"] [role="radiogroup"] [data-baseweb="radio"] div:first-child {
        border-color: #4b5563 !important;
        background-color: transparent !important;
        width: 16px !important;
        height: 16px !important;
    }
    /* selected — red fill */
    [data-testid="stSidebar"] [role="radiogroup"] [aria-checked="true"] [data-baseweb="radio"] div:first-child {
        border-color: #dc2626 !important;
        background-color: #dc2626 !important;
    }

    /* ── Sidebar horizontal divider ── */
    [data-testid="stSidebar"] hr {
        border-color: #1f2937;
        margin: 0.75rem 0;
    }

    /* ── Section header (main area) ── */
    .section-header {
        border-bottom: 2px solid #3d3d5c;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def severity_badge(severity):
    colors = {
        "small": "badge-green",
        "medium": "badge-yellow",
        "large": "badge-red",
        "very_large": "badge-red",
    }
    cls = colors.get(severity, "badge-blue")
    return f'<span class="status-badge {cls}">{severity.upper()}</span>'


def rano_badge(assessment):
    colors = {
        "CR": "badge-green",
        "PR": "badge-blue",
        "SD": "badge-yellow",
        "PD": "badge-red",
    }
    cls = colors.get(assessment, "badge-blue")
    names = {"CR": "Complete Response", "PR": "Partial Response", "SD": "Stable Disease", "PD": "Progressive Disease"}
    name = names.get(assessment, assessment)
    return f'<span class="status-badge {cls}">{assessment} — {name}</span>'


def metric_card(title, value):
    return f'<div class="metric-card"><h3>{title}</h3><p>{value}</p></div>'


# ── Sidebar ──────────────────────────────────────────────────
# Check running jobs on every page load
check_running_jobs()

with st.sidebar:
    st.markdown('<p class="sb-title">🧠 Brain Tumor Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── 2-tab data source selector ────────────────────────────
    brats_tab, upload_tab = st.tabs(["🧬 BraTS Dataset", "📁 Upload Folder"])

    # ── Tab 1: BraTS / NIfTI dataset (Kaggle or local) ───────
    with brats_tab:
        default_dir = os.environ.get("DATA_DIR", "")
        brats_path = st.text_input(
            "BraTS dataset path",
            value=st.session_state.get("brats_dir", default_dir),
            key="brats_dir_widget",
            placeholder="/path/to/BraTS2020_TrainingData",
            help=(
                "Point to ANY level of the Kaggle BraTS dataset — "
                "the app will automatically find the BraTS20_Training_XXX patient folders "
                "even if they are nested inside BraTS2020_TrainingData/MICCAI_BraTS2020_Tr…/."
            ),
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📂 Load", use_container_width=True, key="load_brats"):
                if brats_path and os.path.isdir(brats_path):
                    # Resolve actual patient root (handles Kaggle nesting)
                    resolved = _find_brats_patient_root(brats_path)
                    sub_dirs = [
                        n for n in os.listdir(resolved)
                        if os.path.isdir(os.path.join(resolved, n)) and not n.startswith(".")
                    ]
                    if sub_dirs:
                        st.session_state["uploaded_data_dir"] = resolved
                        st.session_state["active_format"] = "nifti"
                        st.session_state["brats_dir"] = brats_path
                        st.session_state.pop("uploaded_patient_id", None)
                        if resolved != brats_path:
                            st.info(f"Auto-detected patient root:\n`{resolved}`")
                        st.rerun()
                    else:
                        st.error(
                            "No patient sub-folders found. "
                            "Make sure the path contains (or leads to) BraTS20_Training_XXX folders."
                        )
                else:
                    st.error("Directory not found.")
        with col_b:
            if st.button("🔄 Reset", use_container_width=True, key="reset_brats"):
                for k in ["uploaded_data_dir", "active_format", "brats_dir", "uploaded_patient_id"]:
                    st.session_state.pop(k, None)
                st.rerun()
        if default_dir and not st.session_state.get("uploaded_data_dir"):
            st.info(f"DATA_DIR env detected:\n`{default_dir}`\n\nClick **Load** to use it.")

    # ── Tab 2: Upload a DICOM patient folder from your system ─
    with upload_tab:
        st.caption(
            "Select **all files** inside your patient's DICOM folder "
            "(open the folder in the dialog, then press **Ctrl+A** / **Cmd+A** to select all)."
        )
        upload_patient_id_folder = st.text_input(
            "Patient ID",
            value="my_patient",
            key="folder_patient_id",
            help="A name to identify this patient in the dashboard.",
        )
        uploaded_folder_files = st.file_uploader(
            "DICOM files from patient folder",
            accept_multiple_files=True,
            help=(
                "Navigate into your patient folder, press Ctrl+A (Windows) or "
                "Cmd+A (Mac) to select all DICOM files, then click Open."
            ),
            key="folder_uploader",
        )
        if uploaded_folder_files:
            st.caption(f"{len(uploaded_folder_files)} file(s) selected.")
        if st.button("⬆️ Stage & Load", use_container_width=True, key="stage_folder_btn", type="primary"):
            if not uploaded_folder_files:
                st.error("Please select at least one DICOM file first.")
            else:
                with st.spinner(f"Staging {len(uploaded_folder_files)} file(s)…"):
                    staged_dir, staged_pid, ok_count, bad_count = stage_uploaded_dicoms(
                        uploaded_folder_files, upload_patient_id_folder
                    )
                if staged_dir and staged_pid:
                    st.session_state["uploaded_data_dir"] = staged_dir
                    st.session_state["uploaded_patient_id"] = staged_pid
                    st.session_state["active_format"] = "dicom"
                    st.success(f"✅ Staged {ok_count} DICOM file(s) for '{staged_pid}'.")
                    if bad_count:
                        st.warning(f"Skipped {bad_count} non-DICOM file(s).")
                    st.rerun()
                else:
                    st.error(
                        "No readable DICOM files found. "
                        "Make sure the files you selected are .dcm or standard DICOM format."
                    )

    # ── Resolve active data source ────────────────────────────
    active_data_dir = st.session_state.get("uploaded_data_dir", "")
    active_format   = st.session_state.get("active_format", "dicom")
    staged_pid_single = st.session_state.get("uploaded_patient_id")
    if staged_pid_single:
        st.caption(f"Staged patient: **{staged_pid_single}**")
    if active_data_dir:
        st.caption(f"Dataset: `{active_data_dir}`")

    all_patients = discover_all_patients(active_format, active_data_dir)
    if staged_pid_single and staged_pid_single not in all_patients:
        all_patients = sorted(set(all_patients + [staged_pid_single]))
    processed = find_processed_patients()

    if not all_patients:
        st.warning("No patients found. Load a dataset or upload files above.")
        if active_data_dir:
            st.code(f"Dataset dir: {os.path.abspath(active_data_dir)}")
        st.stop()

    # ── Search ────────────────────────────────────────────────
    search_term = st.text_input("🔍 Search patient", placeholder="Type to filter…", key="patient_search")
    filtered_patients = (
        [p for p in all_patients if search_term.lower() in p.lower()]
        if search_term else all_patients
    )
    if not filtered_patients:
        st.warning("No patients match the search.")
        st.stop()

    # Build display labels showing status
    labels = []
    for pid in filtered_patients:
        job = get_job_status(pid)
        if pid in processed and job and job.get("status") == "completed_with_errors":
            labels.append(f"⚠️ {pid}")
        elif pid in processed:
            labels.append(f"✅ {pid}")
        elif job and job.get("status") == "running":
            labels.append(f"⏳ {pid}")
        elif job and job.get("status") == "failed":
            labels.append(f"❌ {pid}")
        else:
            labels.append(f"⬚ {pid}")

    selected_label = st.selectbox("Select Patient", labels, key="patient_selector")
    patient_id = selected_label.split(" ", 1)[1].strip()

    running_count = sum(1 for j in get_all_jobs() if j.get("status") == "running")
    st.caption(
        f"{len(processed)}/{len(all_patients)} processed"
        + (f" · {running_count} running" if running_count else "")
    )
    st.markdown("---")

    patient_processed = patient_id in processed
    job = get_job_status(patient_id)
    job_status = job.get("status") if job else None

    # ── Pipeline status & controls ───────────────────────────
    if job_status == "running":
        st.markdown('<div class="pipe-running">⏳ Pipeline is running…</div>', unsafe_allow_html=True)
        st.markdown("")
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_running"):
            st.rerun()

    elif job_status == "failed" and not patient_processed:
        st.markdown('<div class="pipe-error">❌ Pipeline failed</div>', unsafe_allow_html=True)
        st.markdown("")
        log_path = job.get("log_file", "") if job else ""
        if log_path and os.path.exists(log_path):
            with open(log_path) as lf:
                lines = lf.readlines()
            last_lines = "".join(lines[-10:]) if lines else "No log output"
            with st.expander("Last log lines"):
                st.code(last_lines, language="text")
        if st.button("🔄 Retry Pipeline", type="primary", use_container_width=True, key="retry_btn"):
            retry_data_dir = (job or {}).get("data_dir", active_data_dir)
            retry_format = (job or {}).get("input_format", active_format)
            if start_pipeline_job(patient_id, retry_data_dir, retry_format):
                st.success("Pipeline restarted!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Cannot restart: staged data directory not available.")

    elif not patient_processed:
        st.markdown('<div class="pipe-pending">⬚ Not yet processed</div>', unsafe_allow_html=True)
        st.markdown("")
        if st.button("🚀 Run Pipeline", type="primary", use_container_width=True, key="run_pipeline_btn"):
            if active_data_dir and os.path.isdir(active_data_dir):
                if start_pipeline_job(patient_id, active_data_dir, active_format):
                    st.success("Pipeline started in background!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to start pipeline.")
            else:
                st.error("Load a dataset or stage files first.")

    else:
        # ── Processed: show meta + status box ────────────────
        report = load_report(patient_id)
        if report:
            meta = report.get("report_metadata", {})
            st.markdown(f"**Generated:** {meta.get('generated_at', 'N/A')[:19]}")
            st.markdown(f"**Pipeline:** v{meta.get('pipeline_version', '?')}")
            has_errors = meta.get("has_errors", False)
            errors = report.get("pipeline_errors", [])
            if has_errors or errors:
                st.markdown('<div class="pipe-error">⚠️ Pipeline reported errors</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pipe-success">✅ Pipeline completed successfully</div>', unsafe_allow_html=True)
        st.markdown("")
        if st.button("🔄 Reprocess Patient", use_container_width=True, key="reprocess_btn"):
            if start_pipeline_job(patient_id, active_data_dir, active_format):
                st.success("Reprocessing started in background!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Cannot reprocess: staged data directory not available.")

    st.markdown("---")
    st.markdown("### Navigation")
    section = st.radio(
        "Go to",
        [
            "Overview",
            "MRI Visualization",
            "Radiomics",
            "Classification",
            "Clinical Reasoning",
            "CAP Report",
            "Processing Status",
        ],
        label_visibility="collapsed",
        key="nav_radio",
    )

# Load data for processed patients (needed outside sidebar)
if patient_processed:
    report = load_report(patient_id)
    cap = load_cap(patient_id)
    clinical = load_clinical(patient_id)
    radiomics = load_radiomics(patient_id)
    viz_images = get_viz_images(patient_id)
else:
    report = cap = clinical = radiomics = None
    viz_images = {}


# ── Overview ─────────────────────────────────────────────────
if section == "Overview":
    st.markdown("# Patient Overview")

    if report:
        tumor = report.get("tumor_summary", {})
        morph = tumor.get("morphology", {})
        who = report.get("who_classification", {})
        rano = report.get("rano_assessment", {})

        # Top metrics row
        vol = morph.get('tumor_volume', 'N/A')
        vol_str = f"{vol:,} mm³" if isinstance(vol, (int, float)) else f"{vol} mm³"
        cols = st.columns(5)
        with cols[0]:
            st.markdown(metric_card("Volume", vol_str), unsafe_allow_html=True)
        with cols[1]:
            diam = morph.get('max_diameter', 0)
            st.markdown(metric_card("Max Diameter", f"{float(diam):.1f} mm" if isinstance(diam, (int, float)) else str(diam)), unsafe_allow_html=True)
        with cols[2]:
            sph = morph.get('sphericity', 0)
            st.markdown(metric_card("Sphericity", f"{float(sph):.3f}" if isinstance(sph, (int, float)) else str(sph)), unsafe_allow_html=True)
        with cols[3]:
            st.markdown(metric_card("WHO Grade", who.get("who_grade", "N/A")), unsafe_allow_html=True)
        with cols[4]:
            conf = who.get('confidence', 0)
            st.markdown(metric_card("Confidence", f"{float(conf):.0%}" if isinstance(conf, (int, float)) else str(conf)), unsafe_allow_html=True)

        st.markdown("")

        # Classification & RANO row
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Tumor Classification")
            st.markdown(f"**Type:** {who.get('full_name', 'N/A')}")
            st.markdown(f"**Severity:** {severity_badge(tumor.get('volume_severity', 'unknown'))}", unsafe_allow_html=True)
            st.markdown(f"**Location:** {', '.join(tumor.get('location', ['unknown']))}")
            st.markdown(f"**Prognosis:** {who.get('prognosis', 'N/A')}")

        with col2:
            st.markdown("### Response Assessment")
            st.markdown(f"**RANO:** {rano_badge(rano.get('assessment', 'N/A'))}", unsafe_allow_html=True)
            reasoning = rano.get("reasoning", [])
            for r in reasoning:
                st.markdown(f"- {r}")

            prog = report.get("tumor_progression", {})
            st.markdown(f"**Progression:** {prog.get('state', 'unknown').title()}")
            if prog.get("growth_rate") is not None:
                st.markdown(f"**Growth Rate:** {prog['growth_rate']:.2f} mm³/day")

        # Symptoms
        st.markdown("### Inferred Symptoms")
        symptoms = tumor.get("inferred_symptoms", [])
        if symptoms:
            symptom_cols = st.columns(min(len(symptoms), 4))
            for i, s in enumerate(symptoms):
                with symptom_cols[i % len(symptom_cols)]:
                    st.info(s.title())
        else:
            st.markdown("_No symptoms inferred_")

        # Differential diagnosis
        diff = who.get("differential_diagnosis", [])
        if diff:
            st.markdown("### Differential Diagnosis")
            for d in diff:
                with st.expander(f"{d['type'].replace('_', ' ').title()} (score: {d['score']})"):
                    for r in d.get("reasoning", []):
                        st.markdown(f"- {r}")

    elif clinical:
        st.info("Full report not available. Showing clinical profile only.")
        st.json(clinical)
    else:
        st.warning("No data available for this patient.")


# ── MRI Visualization ───────────────────────────────────────
elif section == "MRI Visualization":
    st.markdown("# MRI Visualization")

    if viz_images:
        # Show tumor overlay and MRI slices side by side if available
        primary = ["Tumor Overlay", "Mri Slices"]
        top_imgs = {k: v for k, v in viz_images.items() if k in primary}
        other_imgs = {k: v for k, v in viz_images.items() if k not in primary}

        if top_imgs:
            cols = st.columns(len(top_imgs))
            for i, (label, path) in enumerate(top_imgs.items()):
                with cols[i]:
                    st.markdown(f"#### {label}")
                    render_image_safe(path)

        if other_imgs:
            st.markdown("---")
            cols = st.columns(min(len(other_imgs), 3))
            for i, (label, path) in enumerate(other_imgs.items()):
                with cols[i % 3]:
                    st.markdown(f"#### {label}")
                    render_image_safe(path)
    else:
        st.warning("No visualization images found. Run the full pipeline to generate them.")

    # Show segmentation info
    seg_path = os.path.join(OUTPUT_DIR, "segmentation", f"{patient_id}_seg.nii.gz")
    prep_path = os.path.join(OUTPUT_DIR, "preprocessed", f"{patient_id}_preprocessed.npy")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Available Data")
        st.markdown(f"- Segmentation mask: {'✅' if os.path.exists(seg_path) else '❌'}")
        st.markdown(f"- Preprocessed volume: {'✅' if os.path.exists(prep_path) else '❌'}")
    with col2:
        if os.path.exists(prep_path):
            vol = np.load(prep_path)
            st.markdown("#### Volume Info")
            st.markdown(f"- Shape: {vol.shape}")
            st.markdown(f"- Dtype: {vol.dtype}")
            st.markdown(f"- Range: [{vol.min():.3f}, {vol.max():.3f}]")


# ── Radiomics ────────────────────────────────────────────────
elif section == "Radiomics":
    st.markdown("# Radiomics Features")

    if radiomics:
        shape_features = {k: v for k, v in radiomics.items() if k.startswith("shape_")}
        intensity_features = {k: v for k, v in radiomics.items() if k.startswith("intensity_")}
        texture_features = {k: v for k, v in radiomics.items() if not k.startswith("shape_") and not k.startswith("intensity_")}

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Shape Features")
            for k, v in shape_features.items():
                label = k.replace("shape_", "").replace("_", " ").title()
                if isinstance(v, float):
                    st.metric(label, f"{v:,.4f}")
                else:
                    st.metric(label, f"{v:,}")

        with col2:
            st.markdown("### Intensity Features")
            for k, v in intensity_features.items():
                label = k.replace("intensity_", "").replace("_", " ").title()
                st.metric(label, f"{v:.6f}")

        if texture_features:
            st.markdown("### Texture Features")
            st.json(texture_features)

        # Bar chart of intensity distribution stats
        st.markdown("### Intensity Distribution")
        chart_data = {
            k.replace("intensity_", ""): v
            for k, v in intensity_features.items()
            if k in ("intensity_mean", "intensity_std", "intensity_min", "intensity_max", "intensity_median")
        }
        if chart_data:
            st.bar_chart(chart_data)

    else:
        st.warning("No radiomics data found for this patient.")


# ── Classification ───────────────────────────────────────────
elif section == "Classification":
    st.markdown("# Tumor Classification Details")

    if report:
        who = report.get("who_classification", {})

        st.markdown(f"## {who.get('full_name', 'Unknown')}")
        st.markdown(f"**WHO Grade:** {who.get('who_grade', 'N/A')}  |  "
                    f"**Confidence:** {who.get('confidence', 0):.0%}  |  "
                    f"**Type:** `{who.get('classified_as', 'N/A')}`")

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

        # RANO
        st.markdown("---")
        rano = report.get("rano_assessment", {})
        st.markdown("## RANO Assessment")
        st.markdown(rano_badge(rano.get("assessment", "N/A")), unsafe_allow_html=True)
        for r in rano.get("reasoning", []):
            st.markdown(f"- {r}")

        # Similar cases
        st.markdown("---")
        similar = report.get("similar_cases", [])
        st.markdown(f"## Similar Cases ({len(similar)} found)")
        if similar:
            for case in similar:
                with st.expander(f"Patient {case.get('patient_id', '?')} — Similarity: {case.get('similarity_score', 0):.2%}"):
                    st.markdown(f"- **Location:** {', '.join(case.get('tumor_location', []))}")
                    st.markdown(f"- **Severity:** {case.get('volume_severity', 'N/A')}")
        else:
            st.markdown("_No similar cases retrieved. Run with Weaviate to enable similarity search._")
    else:
        st.warning("No report data found for this patient.")


# ── Clinical Reasoning ──────────────────────────────────────
elif section == "Clinical Reasoning":
    st.markdown("# AI Clinical Reasoning")

    if report:
        reasoning = report.get("ai_clinical_reasoning", "")
        if reasoning:
            # Parse the reasoning text into sections
            sections = reasoning.split("\n\n")
            for s in sections:
                if ":" in s:
                    title, body = s.split(":", 1)
                    title = title.strip()
                    body = body.strip()
                    if "DIAGNOSIS" in title.upper():
                        st.markdown(f"### 🔬 {title}")
                        st.markdown(body)
                    elif "TREATMENT" in title.upper():
                        st.markdown(f"### 💊 {title}")
                        st.success(body)
                    elif "PROGNOSIS" in title.upper():
                        st.markdown(f"### 📊 {title}")
                        st.info(body)
                    elif "PROGRESSION" in title.upper():
                        st.markdown(f"### 📈 {title}")
                        st.markdown(body)
                    elif "RANO" in title.upper():
                        st.markdown(f"### 📋 {title}")
                        st.markdown(body)
                    elif "SYMPTOM" in title.upper():
                        st.markdown(f"### 🩺 {title}")
                        st.warning(body)
                    elif "RECOMMENDATION" in title.upper():
                        st.markdown(f"### ✅ {title}")
                        st.success(body)
                    else:
                        st.markdown(f"### {title}")
                        st.markdown(body)
                else:
                    st.markdown(s)
        else:
            st.markdown("_No clinical reasoning generated._")

        # Pipeline errors
        errors = report.get("pipeline_errors", [])
        if errors:
            st.markdown("---")
            st.markdown("### ⚠️ Pipeline Errors")
            for e in errors:
                st.error(e)
    else:
        st.warning("No report data found for this patient.")


# ── CAP Report ───────────────────────────────────────────────
elif section == "CAP Report":
    st.markdown("# CAP Structured Report")

    if cap:
        section_names = {
            "section_1_patient_information": "📋 Patient Information",
            "section_2_mri_study_information": "🔬 MRI Study Information",
            "section_3_tumor_characteristics": "🧠 Tumor Characteristics",
            "section_4_radiomics_summary": "📊 Radiomics Summary",
            "section_5_rano_classification": "📈 RANO Classification",
            "section_6_who_classification": "🏥 WHO Classification",
            "section_7_similar_tumor_cases": "🔍 Similar Tumor Cases",
            "section_8_clinical_interpretation": "💡 Clinical Interpretation",
            "section_9_physician_notes": "👨‍⚕️ Physician Notes",
        }

        for key, title in section_names.items():
            data = cap.get(key, {})
            if data:
                with st.expander(title, expanded=(key in ("section_1_patient_information", "section_3_tumor_characteristics"))):
                    if isinstance(data, dict):
                        for k, v in data.items():
                            label = k.replace("_", " ").title()
                            if isinstance(v, list):
                                st.markdown(f"**{label}:**")
                                for item in v:
                                    if isinstance(item, dict):
                                        st.json(item)
                                    else:
                                        st.markdown(f"  - {item}")
                            elif isinstance(v, dict):
                                st.markdown(f"**{label}:**")
                                st.json(v)
                            elif isinstance(v, float):
                                st.markdown(f"**{label}:** {v:,.4f}")
                            elif isinstance(v, bool):
                                st.markdown(f"**{label}:** {'Yes' if v else 'No'}")
                            else:
                                st.markdown(f"**{label}:** {v}")
                    else:
                        st.write(data)

                    if key == "section_9_physician_notes":
                        st.markdown("---")
                        render_hitl_review_form(patient_id)
    else:
        st.warning("No CAP report found. Run the full pipeline (Phase 3) to generate it.")


# ── Processing Status ───────────────────────────────────────
elif section == "Processing Status":
    st.markdown("# Processing Status")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        auto_refresh = st.checkbox("Auto-refresh (10s)")

    jobs = get_all_jobs()
    running_jobs = [j for j in jobs if j.get("status") == "running"]
    completed_jobs = [j for j in jobs if j.get("status") == "completed"]
    error_jobs = [j for j in jobs if j.get("status") in ("completed_with_errors", "failed")]

    # Summary metrics
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Jobs", len(jobs))
    with cols[1]:
        st.metric("Running", len(running_jobs))
    with cols[2]:
        st.metric("Completed", len(completed_jobs))
    with cols[3]:
        st.metric("Errors/Failed", len(error_jobs))

    # Running jobs
    if running_jobs:
        st.markdown("### ⏳ Currently Running")
        for j in running_jobs:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{j['patient_id']}**")
                    started = j.get("started_at", "")[:19]
                    st.caption(f"Started: {started}")
                with c2:
                    log_path = j.get("log_file", "")
                    if log_path and os.path.exists(log_path):
                        with open(log_path) as lf:
                            lines = lf.readlines()
                        last_lines = lines[-5:] if lines else ["(no output yet)"]
                        with st.expander("Live log"):
                            st.code("".join(last_lines))

    # Error / failed jobs
    if error_jobs:
        st.markdown("### ⚠️ Errors & Failures")
        for j in error_jobs:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    status_label = "Completed with errors" if j["status"] == "completed_with_errors" else "Failed"
                    st.markdown(f"**{j['patient_id']}** — {status_label}")
                    errors = j.get("errors", [])
                    if errors:
                        for e in errors[:3]:
                            st.warning(e)
                    finished = j.get("finished_at", "")[:19]
                    st.caption(f"Started: {j.get('started_at', '')[:19]} · Finished: {finished}")
                with c2:
                    if st.button("🔄 Reprocess", key=f"reprocess_{j['patient_id']}", use_container_width=True):
                        if start_pipeline_job(j["patient_id"], j.get("data_dir", ""), j.get("input_format", "dicom")):
                            st.rerun()
                        else:
                            st.info("Cannot reprocess: original staged data directory no longer exists.")
                    log_path = j.get("log_file", "")
                    if log_path and os.path.exists(log_path):
                        with st.expander("Full log"):
                            with open(log_path) as lf:
                                st.code(lf.read()[-3000:])

    # Completed jobs
    if completed_jobs:
        st.markdown("### ✅ Completed")
        for j in completed_jobs:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{j['patient_id']}**")
                    started = j.get("started_at", "")[:19]
                    finished = j.get("finished_at", "")[:19]
                    st.caption(f"Started: {started} · Finished: {finished}")
                with c2:
                    if st.button("🔄 Reprocess", key=f"reprocess_{j['patient_id']}", use_container_width=True):
                        if start_pipeline_job(j["patient_id"], j.get("data_dir", ""), j.get("input_format", "dicom")):
                            st.rerun()
                        else:
                            st.info("Cannot reprocess: original staged data directory no longer exists.")

    if not jobs:
        st.info("No processing jobs yet. Select a patient and click 'Run Pipeline' to start.")

    if auto_refresh:
        time.sleep(10)
        st.rerun()
