#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import os
import platform
import socket
import threading
import time
import zipfile
from pathlib import PurePosixPath
from types import SimpleNamespace

import psutil
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from prometheus_client import (Counter, Gauge, Histogram, Info, Summary,
                               disable_created_metrics, start_http_server)


MODEL_ID = "Salesforce/blip-image-captioning-base"   
MODEL_LABEL = "blip"                                 
METRICS_PORT = 18000
MAX_BULK_IMAGES = 100                                 
MAX_ZIP_UNCOMPRESSED = 500 * 1024 * 1024              
MAX_SIDE = 640                                        
MAX_NEW_TOKENS = 30
APP_VERSION = "2.0.0"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
MODES = ("single", "bulk")
BULK_STATUSES = ("completed", "partial", "failed", "aborted")

INFERENCE_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 21)
PER_IMAGE_BUCKETS = (0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34, 60)
PREPROCESS_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)
WAIT_BUCKETS = (0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 60)
UPLOAD_BUCKETS = (0.5, 1, 2, 3, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800)
ZIP_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
SIZE_BUCKETS = (10e3, 50e3, 100e3, 250e3, 500e3, 1e6, 2.5e6, 5e6, 10e6)


class CaptionError(Exception):
    """An image failed; str(e) is the `reason` label (already counted in metrics)."""


class BulkProgress:
    """Combined progress of all ZIP jobs currently running (thread-safe).

    A single unlabelled gauge is used instead of one series per job, because a
    job-id label would create a new time series for every job (unbounded cardinality).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total = 0
        self.done = 0
        self.last = 0.0

    def start(self, n: int) -> None:
        with self._lock:
            self.total += n

    def step(self) -> None:
        with self._lock:
            self.done += 1

    def finish(self, n_total: int, n_done: int) -> None:
        with self._lock:
            self.total -= n_total
            self.done -= n_done
            if self.total <= 0:
                self.total, self.done, self.last = 0, 0, 1.0   # idle -> show 100 %

    def ratio(self) -> float:
        with self._lock:
            return self.done / self.total if self.total else self.last

# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_metrics() -> SimpleNamespace:
    disable_created_metrics()                   
    m = SimpleNamespace(progress=BulkProgress())

    # ---------------- COUNTERS (only increase; exported with _total) -------
    m.upload_requests = Counter(
        "upload_requests", "Upload operations received (1 per single caption click, 1 per ZIP job)",
        ["mode", "client_ip", "session_id", "browser"])
    m.upload_errors = Counter(
        "upload_errors", "Upload operations that failed as a whole", ["mode", "reason"])
    m.images_uploaded = Counter(
        "images_uploaded", "Images accepted for captioning (single upload, or image file found in a ZIP)",
        ["mode"])
    m.images_processed = Counter(
        "images_processed", "Images captioned successfully", ["mode", "model", "client_ip"])
    m.images_failed = Counter(
        "images_failed", "Images that could not be captioned", ["mode", "stage", "reason"])
    m.bulk_jobs = Counter(
        "bulk_jobs", "Bulk ZIP jobs finished, by outcome", ["status"])
    m.zip_skipped = Counter(
        "zip_entries_skipped", "Files inside ZIPs that were not images or exceeded limits", ["reason"])
    m.model_load_failures = Counter(
        "model_load_failures", "Model initialisation attempts that raised an error", ["model"])
    m.sessions = Counter(
        "sessions_started", "New browser sessions opened", ["client_ip", "browser"])

    # ---------------- GAUGES (current state; go up and down) ---------------
    m.active = Gauge("active_requests", "Upload operations being processed right now", ["mode"])
    m.images_queued = Gauge("images_queued", "Images still waiting inside running ZIP jobs")
    m.progress_ratio = Gauge("bulk_job_progress_ratio",
                             "Progress of running ZIP jobs, 0 to 1 (1 when idle)")
    m.progress_ratio.set_function(m.progress.ratio)          # evaluated at scrape time
    m.zip_discovered = Gauge("zip_images_discovered", "Valid images found in the latest ZIP job")
    m.model_loaded = Gauge("model_loaded", "1 when the model is loaded and ready, else 0", ["model"])
    m.model_load_seconds = Gauge("model_load_duration_seconds",
                                 "Duration of the most recent model load", ["model"])
    m.model_memory = Gauge("model_memory_bytes",
                           "Resident memory added to the process by loading the model", ["model"])
    m.last_success = Gauge("last_successful_inference_timestamp_seconds",
                           "Unix time of the latest successful caption (0 = none yet)")
    m.active_sessions = Gauge("active_sessions", "Browser sessions currently connected")
    m.active_sessions.set_function(_active_sessions)

    # ---------------- HISTOGRAMS (distributions: _bucket, _sum, _count) ----
    m.inference = Histogram("inference_duration_seconds", "Model generate() time for one image",
                            ["mode", "model"], buckets=INFERENCE_BUCKETS)
    m.preprocess = Histogram("image_preprocessing_duration_seconds",
                             "Decode + resize + tensorise time for one image", ["mode"],
                             buckets=PREPROCESS_BUCKETS)
    m.per_image = Histogram("image_processing_duration_seconds",
                            "Total time for one image: preprocessing + model wait + inference + decoding "
                            "(failed images included)", ["mode"], buckets=PER_IMAGE_BUCKETS)
    m.model_wait = Histogram("model_wait_duration_seconds",
                             "Time an image waited for the shared model (concurrent users)", ["mode"],
                             buckets=WAIT_BUCKETS)
    m.upload_seconds = Histogram("upload_processing_duration_seconds",
                                 "End-to-end time of one upload operation (single image, or whole ZIP job)",
                                 ["mode"], buckets=UPLOAD_BUCKETS)
    m.zip_extract = Histogram("zip_extraction_duration_seconds",
                              "Time to open a ZIP and inspect/filter its entries", buckets=ZIP_BUCKETS)
    m.image_size = Histogram("image_size_bytes", "File size of each image", ["mode"],
                             buckets=SIZE_BUCKETS)

    m.upload_size = Summary("upload_size_bytes", "Size of each uploaded payload (image or ZIP)", ["mode"])
    m.caption_words = Summary("caption_length_words", "Words in each generated caption", ["mode"])
    m.bulk_job_size = Summary("bulk_job_size_images", "Valid images in each ZIP job")

    m.build = Info("app_build", "Build information of the captioning app")

  
    for mode in MODES:
        m.active.labels(mode)
        m.images_uploaded.labels(mode)
        m.upload_size.labels(mode)
        m.caption_words.labels(mode)
    for status in BULK_STATUSES:
        m.bulk_jobs.labels(status)
    m.model_loaded.labels(MODEL_LABEL).set(0)
    m.model_load_failures.labels(MODEL_LABEL)

    start_http_server(METRICS_PORT)     
    return m


def _active_sessions() -> int:
    try:
        from streamlit.runtime import get_instance
        return get_instance()._session_mgr.num_active_sessions()
    except Exception:
        return 0



@st.cache_resource(show_spinner="Loading the BLIP model on CPU (first run only)…")
def load_model() -> SimpleNamespace:
    m = get_metrics()
    import torch
    import transformers
    from transformers import BlipForConditionalGeneration, BlipProcessor

    torch.set_num_threads(os.cpu_count() or 1)
    rss_before = psutil.Process().memory_info().rss
    t0 = time.perf_counter()
    try:
        processor = BlipProcessor.from_pretrained(MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).eval()
    except Exception:
        m.model_load_failures.labels(MODEL_LABEL).inc()
        m.model_loaded.labels(MODEL_LABEL).set(0)
        raise
    m.model_load_seconds.labels(MODEL_LABEL).set(time.perf_counter() - t0)
    m.model_memory.labels(MODEL_LABEL).set(max(0, psutil.Process().memory_info().rss - rss_before))
    m.model_loaded.labels(MODEL_LABEL).set(1)
    m.build.info({
        "version": APP_VERSION, "model": MODEL_ID, "device": "cpu",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "host": socket.gethostname(),
    })
    return SimpleNamespace(processor=processor, model=model, torch=torch, lock=threading.Lock())

def _browser_family(ua: str) -> str:
    for key, name in (("Edg/", "edge"), ("OPR/", "opera"), ("Firefox/", "firefox"),
                      ("Chrome/", "chrome"), ("Safari/", "safari"), ("curl", "curl")):
        if key in ua:
            return name
    return "other"


def client_context() -> SimpleNamespace:
    ip, ua, sid = None, "", "unknown"
    try:
        ip = st.context.ip_address                    
    except Exception:
        pass
    try:
        headers = st.context.headers
        ua = headers.get("User-Agent", "") or ""
        if not ip:
            xff = headers.get("X-Forwarded-For")
            ip = xff.split(",")[0].strip() if xff else None
    except Exception:
        pass
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is not None:
            sid = ctx.session_id[:8]                   
    except Exception:
        pass
    return SimpleNamespace(ip=ip or "localhost", browser=_browser_family(ua), session=sid)

def _fail(m: SimpleNamespace, mode: str, stage: str, reason: str) -> CaptionError:
    m.images_failed.labels(mode, stage, reason).inc()
    return CaptionError(reason)


def caption_one(raw: bytes, mode: str, cli: SimpleNamespace, res: SimpleNamespace,
                m: SimpleNamespace) -> tuple[str, float]:
    t_start = time.perf_counter()
    m.image_size.labels(mode).observe(len(raw))
    try:
        t0 = time.perf_counter()
        try:
            img = Image.open(io.BytesIO(raw))
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            inputs = res.processor(images=img, return_tensors="pt")
        except Image.DecompressionBombError as e:          # absurdly large resolution
            raise _fail(m, mode, "preprocessing", "image_too_large") from e
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
            raise _fail(m, mode, "preprocessing", "corrupt_image") from e
        except Exception as e:
            raise _fail(m, mode, "preprocessing", "preprocessing_error") from e
        m.preprocess.labels(mode).observe(time.perf_counter() - t0)

        w0 = time.perf_counter()
        with res.lock:
            m.model_wait.labels(mode).observe(time.perf_counter() - w0)
            t1 = time.perf_counter()
            try:
                with res.torch.inference_mode():
                    out = res.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            except Exception as e:
                raise _fail(m, mode, "inference", "inference_error") from e
            infer_s = time.perf_counter() - t1
        m.inference.labels(mode, MODEL_LABEL).observe(infer_s)

        try:
            text = res.processor.decode(out[0], skip_special_tokens=True).strip()
        except Exception as e:
            raise _fail(m, mode, "decoding", "decoding_error") from e
        if not text:
            raise _fail(m, mode, "decoding", "empty_caption")

        m.caption_words.labels(mode).observe(len(text.split()))
        m.images_processed.labels(mode, MODEL_LABEL, cli.ip).inc()
        m.last_success.set_to_current_time()
        return text, infer_s
    finally:
        m.per_image.labels(mode).observe(time.perf_counter() - t_start)

def single_mode(m: SimpleNamespace, res: SimpleNamespace, cli: SimpleNamespace) -> None:
    up = st.file_uploader("Upload one image", type=sorted(e.lstrip(".") for e in IMAGE_EXTS),
                          key="single_file")
    if up is None:
        return
    raw = up.getvalue()
    key = (up.name, len(raw))
    left, right = st.columns(2)
    try:
        left.image(raw, caption=f"{up.name} · {len(raw) / 1024:.0f} KB")
    except Exception:
        left.warning("Preview not available for this file.")

    if right.button("Generate caption", type="primary", key="single_go"):
        m.upload_requests.labels("single", cli.ip, cli.session, cli.browser).inc()
        m.images_uploaded.labels("single").inc()
        m.upload_size.labels("single").observe(len(raw))
        m.active.labels("single").inc()
        t0 = time.perf_counter()
        try:
            with right, st.spinner("Running BLIP on CPU…"):
                text, dt = caption_one(raw, "single", cli, res, m)
            st.session_state.single_result = (key, text, dt)
        except CaptionError as e:
            m.upload_errors.labels("single", str(e)).inc()
            st.session_state.single_result = None
            right.error(f"Could not caption this image ({e}).")
        finally:
            m.active.labels("single").dec()
            m.upload_seconds.labels("single").observe(time.perf_counter() - t0)

    result = st.session_state.get("single_result")
    if result and result[0] == key:
        right.markdown(f"### “{result[1]}”")
        right.caption(f"Inference time on CPU: {result[2]:.2f} s")


def inspect_zip(raw: bytes, m: SimpleNamespace):
    """Open the ZIP and pick the image entries. Raises zipfile.BadZipFile."""
    t0 = time.perf_counter()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
        members, skipped, total = [], [], 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            p = PurePosixPath(info.filename)
            if "__MACOSX" in p.parts or p.name.startswith("."):
                reason = "hidden_or_metadata"
            elif p.suffix.lower() not in IMAGE_EXTS:
                reason = "not_an_image"
            elif len(members) >= MAX_BULK_IMAGES:
                reason = "over_image_limit"
            elif total + info.file_size > MAX_ZIP_UNCOMPRESSED:
                reason = "over_size_limit"
            else:
                total += info.file_size
                members.append(info)
                continue
            skipped.append((info.filename, reason))
            m.zip_skipped.labels(reason).inc()
        return zf, members, skipped
    finally:
        m.zip_extract.observe(time.perf_counter() - t0)

def bulk_mode(m: SimpleNamespace, res: SimpleNamespace, cli: SimpleNamespace) -> None:
    st.caption(f"Up to {MAX_BULK_IMAGES} images per ZIP are captioned one at a time on the CPU. "
               "Non-image files are skipped; corrupt images are reported, not fatal.")
    up = st.file_uploader("Upload a ZIP of images", type=["zip"], key="bulk_file")
    if up is None:
        return
    raw = up.getvalue()
    key = (up.name, len(raw))

    if st.button("Caption all images", type="primary", key="bulk_go"):
        m.upload_requests.labels("bulk", cli.ip, cli.session, cli.browser).inc()
        m.upload_size.labels("bulk").observe(len(raw))
        m.active.labels("bulk").inc()
        t0 = time.perf_counter()
        
        status, n_total, n_done, remaining = "aborted", 0, 0, 0
        try:
            try:
                zf, members, skipped = inspect_zip(raw, m)
            except (zipfile.BadZipFile, OSError):
                status = "failed"
                m.upload_errors.labels("bulk", "bad_zip").inc()
                st.error("That file is not a valid ZIP archive.")
                return
            if not members:
                status = "failed"
                m.upload_errors.labels("bulk", "no_images").inc()
                st.warning("No images found in the ZIP.")
                return

            n_total = remaining = len(members)
            m.zip_discovered.set(n_total)
            m.bulk_job_size.observe(n_total)
            m.images_uploaded.labels("bulk").inc(n_total)
            m.images_queued.inc(n_total)
            m.progress.start(n_total)

            rows, n_ok = [], 0
            bar = st.progress(0.0, text="Starting…")
            for i, info in enumerate(members, 1):
                try:
                    try:
                        data = zf.read(info)
                    except Exception as e:                 
                        raise _fail(m, "bulk", "read", "read_error") from e
                    text, dt = caption_one(data, "bulk", cli, res, m)
                    rows.append({"file": info.filename, "caption": text,
                                 "inference_s": round(dt, 3), "status": "ok"})
                    n_ok += 1
                except CaptionError as e:
                    rows.append({"file": info.filename, "caption": "", "inference_s": None,
                                 "status": f"error: {e}"})
                finally:
                    remaining -= 1
                    n_done += 1
                    m.images_queued.dec()
                    m.progress.step()
                bar.progress(i / n_total, text=f"{i}/{n_total} · {info.filename}")

            status = "completed" if n_ok == n_total else ("partial" if n_ok else "failed")
            if status == "failed":
                m.upload_errors.labels("bulk", "all_images_failed").inc()
            st.session_state.bulk_result = {"key": key, "rows": rows, "skipped": skipped,
                                            "elapsed": time.perf_counter() - t0}
        finally:
            if remaining:                                   
                m.images_queued.dec(remaining)
            if n_total:
                m.progress.finish(n_total, n_done)
            m.bulk_jobs.labels(status).inc()
            m.active.labels("bulk").dec()
            m.upload_seconds.labels("bulk").observe(time.perf_counter() - t0)

    result = st.session_state.get("bulk_result")
    if result and result["key"] == key:
        rows = result["rows"]
        n_ok = sum(r["status"] == "ok" for r in rows)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Captioned", n_ok)
        c2.metric("Failed", len(rows) - n_ok)
        c3.metric("Skipped files", len(result["skipped"]))
        c4.metric("Throughput", f"{n_ok / result['elapsed']:.2f} img/s")
        st.dataframe(rows, width="stretch", hide_index=True)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["file", "caption", "inference_s", "status"])
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("Download captions (CSV)", buf.getvalue(), "captions.csv", "text/csv")
        if result["skipped"]:
            with st.expander("Skipped files"):
                st.dataframe([{"file": f, "reason": r} for f, r in result["skipped"]],
                             width="stretch", hide_index=True)

def main() -> None:
    st.set_page_config(page_title="CPU Image Captioning", page_icon="🖼️", layout="wide")
    m = get_metrics()
    cli = client_context()
    if "session_counted" not in st.session_state:
        m.sessions.labels(cli.ip, cli.browser).inc()
        st.session_state.session_counted = True

    st.title("CPU Image Captioning")
    st.caption(f"Model: `{MODEL_ID}` · CPU only · metrics on port {METRICS_PORT}")

    try:
        res = load_model()
    except Exception as e:
        st.error(f"Model failed to load: {e}")
        st.stop()

    with st.sidebar:
        st.subheader("Your session (metric labels)")
        st.write(f"**Client IP:** `{cli.ip}`")
        st.write(f"**Session ID:** `{cli.session}`")
        st.write(f"**Browser:** `{cli.browser}`")
        st.divider()
        st.write(f"Metrics: `http://192.168.64.2:{METRICS_PORT}/metrics`")

    tab_single, tab_bulk = st.tabs(["Single image", "Bulk (ZIP)"])
    with tab_single:
        single_mode(m, res, cli)
    with tab_bulk:
        bulk_mode(m, res, cli)


if __name__ == "__main__":
    main()
