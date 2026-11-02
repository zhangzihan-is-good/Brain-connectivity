"""Recover original ABIDE PCP ROI series and compute locked common primitives.

This downloads public CPAC/filt_global data only into the fourth-part inputs.
Every successful file is recorded with SHA-256 and can be resumed safely.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import urllib.request
import zlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
# Bound nested BLAS threads; these variables affect this process only.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")
sys.path.insert(0, str(ROOT / "_runtime"))
import bootstrap  # Existing libraries only; retain Windows DLL handles.

import h5py
import numpy as np

STRATEGY = "filt_global"
ATLASES = ("aal", "cc200", "ez")
RAW = ROOT / "inputs" / "raw_timeseries"
SOURCE_H5 = WORKSPACE / "dataset" / "abide_data" / "04_rgc_intermediate_data" / "rgc_base.hdf5"
ESTIMATOR_PATH = WORKSPACE / "第三部分最终版" / "code" / "estimators.py"
RGIC_PATH = WORKSPACE / "第三部分最终版" / "code" / "rgic.py"
BASE_URL = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac"
SPECTRAL_BAND = (.01, .1)
REQUIRED = ("r0", "W1_rstar", "W1_lag", "W1_tie", "W2_rstar", "W2_lag", "W2_tie",
            "te_h1_d1", "te_h1_d2", "var_p1", "var_p2", "psi", "coherence", "gcmi",
            "phiid_d1", "phiid_d2")
ESTIMATORS = None
PROTOCOL_ROOT = "https://fcon_1000.projects.nitrc.org/indi/abide/scan_params/"
PROTOCOLS = {
    "CMU_a": ("CMU", 2.), "CMU_b": ("CMU", 1.5), "Caltech": ("Caltech", 2.),
    "KKI": ("KKI", 2.5), "Leuven_1": ("Leuven-1", 1.667), "Leuven_2": ("Leuven-2", 1.667),
    "MaxMun_a": ("MaxMun", 3.), "MaxMun_b": ("MaxMun", 3.),
    "MaxMun_c": ("MaxMun", 3.), "MaxMun_d": ("MaxMun", 3.),
    "NYU": ("NYU", 2.), "OHSU": ("OHSU", 2.5), "Olin": ("Olin", 1.5),
    "Pitt": ("Pitt", 1.5), "SBL": ("SBL", 2.2), "SDSU": ("SDSU", 2.),
    "Stanford": ("Stanford", 2.), "Trinity": ("Trinity", 2.),
    "UCLA_1": ("UCLA1", 3.), "UCLA_2": ("UCLA2", 3.),
    "UM_1": ("UM1", 2.), "UM_2": ("UM2", 2.), "USM": ("USM", 2.), "Yale": ("Yale", 2.)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def header_hash(tokens):
    return hashlib.sha256("\t".join(tokens).encode("ascii")).hexdigest()


def estimator_module():
    global ESTIMATORS
    if ESTIMATORS is None:
        spec = importlib.util.spec_from_file_location("abide_locked_estimators", str(ESTIMATOR_PATH))
        ESTIMATORS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ESTIMATORS)
    return ESTIMATORS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def subjects():
    with h5py.File(SOURCE_H5, "r") as handle:
        return sorted(handle["patients"])


def analysis_subjects():
    cohort = json.loads((ROOT / "inputs" / "common_qc_cohort.json").read_text(encoding="utf-8"))
    if cohort["status"] != "ok":
        raise ValueError("Common three-atlas cohort has not passed input QC")
    return cohort["subject_ids"]


def read_series(path):
    x = np.loadtxt(str(path), dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 12 or x.shape[1] < 2:
        raise ValueError("Invalid time-series shape: {}".format(x.shape))
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite values in original series")
    return x


def download_one(atlas, sid, previous):
    directory = RAW / atlas
    directory.mkdir(parents=True, exist_ok=True)
    name = sid + "_rois_" + atlas + ".1D"
    path = directory / name
    url = BASE_URL + "/" + STRATEGY + "/rois_" + atlas + "/" + name
    key = atlas + "/" + name
    old = previous.get(key)
    if path.exists() and old and old.get("sha256") == sha256(path):
        return dict(old, resumed=True)
    probe = ROOT / "inputs" / "source_probe" / (sid + "_" + STRATEGY + "_rois_" + atlas + ".1D")
    last_error = None
    for attempt in range(1, 5):
        try:
            if probe.exists():
                data = probe.read_bytes()
            elif path.exists():
                data = path.read_bytes()
            else:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
            x = np.loadtxt(io.BytesIO(data), dtype=np.float64)
            if x.ndim != 2 or x.shape[0] < 12 or x.shape[1] < 2 or not np.isfinite(x).all():
                raise ValueError("Invalid original time series")
            if not path.exists() or path.read_bytes() != data:
                temporary = path.with_suffix(".1D.partial")
                temporary.write_bytes(data)
                temporary.replace(path)
            roi_ids = data.splitlines()[0].decode("ascii").split()
            return dict(key=key, subject=sid, atlas=atlas, strategy=STRATEGY,
                        url=url, sha256=hashlib.sha256(data).hexdigest(),
                        bytes=len(data), shape=list(x.shape), roi_ids=roi_ids,
                        roi_order_sha256=header_hash(roi_ids),
                        constant_roi_count=int(np.sum(np.ptp(x, axis=0) == 0)),
                        status="ok", resumed=False)
        except Exception as error:
            last_error = repr(error)
            if path.exists() and (not old or old.get("sha256") != sha256(path)):
                # A corrupt cache is preserved for inspection, then fetched again.
                path.replace(path.with_suffix(".1D.invalid"))
            if attempt < 4:
                time.sleep(min(2 ** attempt, 8))
    return dict(key=key, subject=sid, atlas=atlas, strategy=STRATEGY,
                url=url, status="error", error=last_error)


def download_all(atlases=ATLASES, workers=8):
    RAW.mkdir(parents=True, exist_ok=True)
    log = RAW / "download_manifest.jsonl"
    previous = {}
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("status") == "ok":
                    previous[record["key"]] = record
    ids = subjects()
    tasks = [(atlas, sid) for atlas in atlases for sid in ids]
    done = 0
    failed = []
    records = []
    started = time.time()
    print("DOWNLOAD_START subjects={} atlases={} total={} threads={}".format(
        len(ids), ",".join(atlases), len(tasks), workers), flush=True)
    with log.open("a", encoding="utf-8") as writer:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(download_one, atlas, sid, previous) for atlas, sid in tasks]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                records.append(record)
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()
                done += 1
                if record["status"] != "ok":
                    failed.append(record)
                    print("DOWNLOAD_ERROR " + json.dumps(record), flush=True)
                if done % 50 == 0 or done == len(tasks):
                    print("DOWNLOAD_PROGRESS {}/{} failed={} elapsed_s={:.1f}".format(
                        done, len(tasks), len(failed), time.time() - started), flush=True)
    summary = dict(strategy=STRATEGY, subjects=len(ids), atlases=list(atlases),
                   total=len(tasks), success=len(tasks)-len(failed), failed=failed,
                   files=sorted(records, key=lambda record: record["key"]))
    target = RAW / "download_manifest.json"
    temporary = target.with_suffix(".json.partial")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    print("DOWNLOAD_DONE success={} failed={}".format(summary["success"], len(failed)), flush=True)
    if failed:
        raise RuntimeError("Download incomplete: {} files failed".format(len(failed)))
    return summary


def header_one(sid):
    """Read only 4096 compressed bytes of an official functional NIfTI file.

    PCP headers omit the temporal unit bit in xyzt_units. pixdim[4] is expressed
    in seconds, independently cross-checked against the official ABIDE I
    functional scan protocols; CMU_a=2 and CMU_b=1.5 are kept distinct.
    """
    directory = RAW / "_nifti_headers"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (sid + ".nii.gz.head4096")
    url = BASE_URL + "/" + STRATEGY + "/func_preproc/" + sid + "_func_preproc.nii.gz"
    last_error = None
    for attempt in range(1, 5):
        try:
            if path.exists():
                compressed = path.read_bytes()
            else:
                request = urllib.request.Request(url, headers={"Range": "bytes=0-4095"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise ValueError("Server did not honor limited Range request")
                    compressed = response.read(4096)
                path.write_bytes(compressed)
            raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(compressed)
            if len(raw) < 348:
                raise ValueError("Insufficient decompressed NIfTI header")
            endian = "<" if struct.unpack("<I", raw[:4])[0] == 348 else ">"
            if struct.unpack(endian + "I", raw[:4])[0] != 348:
                raise ValueError("Not a NIfTI-1 header")
            dimensions = struct.unpack(endian + "8h", raw[40:56])
            pixdim4 = float(struct.unpack(endian + "8f", raw[76:108])[4])
            temporal_unit = int(raw[123]) & 56
            if temporal_unit not in (0, 8, 16, 24):
                raise ValueError("Unsupported NIfTI temporal unit")
            tr = pixdim4 * {0: 1., 8: 1., 16: .001, 24: .000001}[temporal_unit]
            if not .1 < tr < 10:
                raise ValueError("Implausible TR: {}".format(tr))
            return sid, dict(status="ok", url=url, header_path=str(path),
                             header_sha256=hashlib.sha256(compressed).hexdigest(),
                             header_range="bytes=0-4095", pixdim4=pixdim4, tr_seconds=tr,
                             xyzt_units=int(raw[123]), n_timepoints=int(dimensions[4]),
                             temporal_unit_code=temporal_unit,
                             unit_interpretation="seconds_assumed_for_missing_time_unit" if temporal_unit == 0 else "explicit_nifti_time_unit",
                             protocol_source=PROTOCOL_ROOT+PROTOCOLS[sid.rsplit("_", 1)[0]][0]+"/rest.pdf",
                             protocol_nominal_tr_seconds=PROTOCOLS[sid.rsplit("_", 1)[0]][1],
                             acquisition_prefix=sid.rsplit("_", 1)[0])
        except Exception as error:
            last_error = repr(error)
            if attempt < 4:
                time.sleep(min(2 ** attempt, 8))
    return sid, dict(status="error", url=url, error=last_error)


def recover_tr(workers=8):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "header_tr_manifest.json"
    records = {}
    ids = subjects()
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(header_one, sid) for sid in ids]
        for future in concurrent.futures.as_completed(futures):
            sid, record = future.result()
            records[sid] = record
            if len(records) % 50 == 0 or len(records) == len(ids):
                print("HEADER_PROGRESS {}/{} elapsed_s={:.1f}".format(
                    len(records), len(ids), time.time()-started), flush=True)
    failed = [sid for sid, record in records.items() if record["status"] != "ok"]
    summary = dict(schema_version="abide-header-tr-v1", status="incomplete" if failed else "complete",
                   interpretation="Official PCP pixdim[4], seconds; temporal-unit bit absent. Cross-check: ABIDE I functional scan protocols.",
                   temporal_unit_policy="PCP pixdim4 interpreted as seconds; absent time unit bit",
                   unit_interpretation="seconds_assumed_for_missing_time_unit",
                   protocol_url="https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html",
                   protocol_sources={prefix: dict(url=PROTOCOL_ROOT+value[0]+"/rest.pdf", nominal_tr_seconds=value[1],
                       note="Official actual TR is 1667 ms with shortest setting; retain each original header TR instead of rounding" if prefix.startswith("Leuven") else "Official scan protocol, ms converted to seconds")
                       for prefix, value in PROTOCOLS.items()},
                   failed=failed, subjects={sid: records[sid] for sid in ids})
    write_json(path, summary)
    groups = {}
    for sid, record in records.items():
        if record["status"] == "ok":
            groups.setdefault(record["acquisition_prefix"], set()).add(record["tr_seconds"])
    print("HEADER_DONE " + json.dumps({key: sorted(value) for key, value in sorted(groups.items())}), flush=True)
    if failed:
        raise RuntimeError("Missing exact TR for {} subjects".format(len(failed)))
    return summary


def spectral_required(x, tr):
    """Final-HNU multitaper estimators, ABIDE's prespecified 0.01–0.1 Hz band.

    Exact five-taper point estimates; unused leave-one-taper diagnostics are
    omitted. No temporal refiltering is performed on the downloaded series.
    """
    estimators = estimator_module()
    n, r = x.shape
    i, j = np.triu_indices(r, 1)
    taper = estimators.dpss(n, NW=3., Kmax=5, sym=False, norm=2)
    fft = np.fft.rfft((x-x.mean(axis=0))[None, :, :]*taper[:, :, None], n=n, axis=1)
    frequency = np.fft.rfftfreq(n, d=tr)
    band = (frequency >= SPECTRAL_BAND[0]) & (frequency <= SPECTRAL_BAND[1])
    if np.sum(band) < 2:
        raise ValueError("Insufficient frequency bins in 0.01–0.1 Hz")
    fourier = fft[:, band, :]
    power = np.abs(fourier)**2
    psi, coherence = np.empty(len(i)), np.empty(len(i))
    for lo in range(0, len(i), 256):
        sl = slice(lo, min(lo+256, len(i)))
        cross = fourier[:, :, i[sl]] * np.conj(fourier[:, :, j[sl]])
        cs = cross.mean(axis=0)
        pi = power[:, :, i[sl]].mean(axis=0)
        pj = power[:, :, j[sl]].mean(axis=0)
        coherency = cs/np.sqrt(pi*pj)
        psi[sl] = np.imag(np.sum(np.conj(coherency[:-1])*coherency[1:], axis=0))
        coherence[sl] = np.mean(np.abs(coherency)**2, axis=0)
    return dict(psi=psi, coherence=coherence), dict(frequency_bins=int(np.sum(band)),
                first_frequency=float(frequency[band][0]), last_frequency=float(frequency[band][-1]))


def prediction_required(x):
    """Use the locked covariance/OLS helpers, computing only used h1 / p1,p2.

    Each retained value is identical to final-HNU gaussian_prediction(x);
    unused histories h2/h3 and order p3 are not evaluated.
    """
    estimators = estimator_module()
    r = x.shape[1]
    i, j = np.triu_indices(r, 1)
    e = len(i)
    z = (x-x.mean(axis=0))/x.std(axis=0)
    covariance = {p: estimators._lag_cov(z, p) for p in (1, 2)}
    source = np.column_stack((i, j)).ravel()
    target = np.column_stack((j, i)).ravel()
    out = {}
    diagnostics = {}
    def finite_condition(a, b):
        value = float(max(np.max(a), np.max(b)))
        return value if np.isfinite(value) else None
    for d in (1, 2):
        cov, n = covariance[d]
        own = target[:, None] + r
        predictors = np.column_stack((own, source+r*d))
        restricted, _, cr, _, sr = estimators._regress_cov(cov, own, target, n)
        full, _, cf, _, sf = estimators._regress_cov(cov, predictors, target, n)
        values, tiny, bad = estimators._ratio(restricted, full, .5)
        name = "te_h1_d{}".format(d)
        out[name] = values.reshape(e, 2)
        diagnostics[name] = dict(singular=int(np.sum(sr | sf)), tiny_negative_clipped=int(np.sum(tiny)),
                                  invalid=int(np.sum(bad)), max_condition=finite_condition(cr, cf))
    for p in (1, 2):
        cov, n = covariance[p]
        own = target[:, None] + r*np.arange(1, p+1)[None, :]
        predictors = np.column_stack((own, source[:, None]+r*np.arange(1, p+1)[None, :]))
        restricted, _, cr, _, sr = estimators._regress_cov(cov, own, target, n)
        full, _, cf, _, sf = estimators._regress_cov(cov, predictors, target, n)
        values, tiny, bad = estimators._ratio(restricted, full, 1.)
        name = "var_p{}".format(p)
        out[name] = values.reshape(e, 2)
        diagnostics[name] = dict(singular=int(np.sum(sr | sf)), tiny_negative_clipped=int(np.sum(tiny)),
                                  invalid=int(np.sum(bad)), max_condition=finite_condition(cr, cf))
    return out, diagnostics


def compute_one(atlas, sid, tr):
    try:
        path = RAW / atlas / (sid + "_rois_" + atlas + ".1D")
        x = read_series(path)
        if np.any(np.ptp(x, axis=0) == 0):
            raise ValueError("Common input QC: constant ROI")
        estimators = estimator_module()
        estimators._validate(x)
        out = estimators.correlation_primitives(x)
        prediction, prediction_diagnostics = prediction_required(x)
        out.update(prediction)
        spectral, spectral_diagnostics = spectral_required(x, tr)
        out.update(spectral)
        out.update(estimators.gcmi_primitives(x))
        phiid = estimators.phiid_primitives(x)
        out.update(phiid)
        diagnostics = dict(prediction=prediction_diagnostics, spectral=spectral_diagnostics,
            gcmi_ties=int(np.sum(out["gcmi_ties_per_roi"])),
            phiid_failed_d1=int(np.sum(phiid["phiid_failed_d1"])),
            phiid_failed_d2=int(np.sum(phiid["phiid_failed_d2"])),
            phiid_reconstruction_max=float(max(np.max(np.abs(phiid["phiid_reconstruction_error_d1"])),
                                               np.max(np.abs(phiid["phiid_reconstruction_error_d2"])))) )
        invalid = {key: int(np.sum(~np.isfinite(out[key]))) for key in REQUIRED
                   if np.any(~np.isfinite(out[key]))}
        if invalid:
            raise ValueError("Estimator nonfinite values; no method-specific fill: " + repr(invalid))
        needed = {}
        for key in REQUIRED:
            dtype = np.int8 if key.endswith("_lag") else np.bool_ if key.endswith("_tie") else np.float32
            needed[key] = np.asarray(out[key], dtype=dtype)
        return sid, needed, dict(status="ok", shape=list(x.shape), diagnostics=diagnostics)
    except Exception as error:
        return sid, None, dict(status="error", error=repr(error))


def source_record(atlas, sid, tr_record):
    path = RAW / atlas / (sid + "_rois_" + atlas + ".1D")
    x = read_series(path)
    tokens = path.open("r", encoding="ascii").readline().split()
    if len(tokens) != x.shape[1]:
        raise ValueError("ROI header count mismatch: " + sid)
    return dict(subject_id=sid, path=str(path), sha256=sha256(path),
                url=BASE_URL + "/" + STRATEGY + "/rois_" + atlas + "/" + path.name,
                strategy=STRATEGY, roi_ids=tokens, roi_order_sha256=header_hash(tokens),
                n_timepoints=int(x.shape[0]), n_rois=int(x.shape[1]), tr_seconds=tr_record["tr_seconds"],
                constant_roi_count=int(np.sum(np.ptp(x, axis=0) == 0)))


def cohort_counts(ids):
    with (WORKSPACE / "dataset" / "abide_data" / "01_subject_metadata" / "list-of-subjects.csv").open(
            "r", encoding="utf-8-sig", newline="") as handle:
        metadata = {row["FILE_ID"]: row for row in csv.DictReader(handle)}
    classes = {"ASD": 0, "TC": 0}
    sites = {}
    for sid in ids:
        row = metadata[sid]
        label = "ASD" if row["DX_GROUP"] == "1" else "TC"
        classes[label] += 1
        counts = sites.setdefault(row["SITE_ID"], dict(n=0, ASD=0, TC=0))
        counts["n"] += 1
        counts[label] += 1
    return dict(n=len(ids), classes=classes, sites=sites)


def make_common_cohort():
    """Keep exactly the same valid subjects in every atlas and every method."""
    ids = subjects()
    tr_path = RAW / "header_tr_manifest.json"
    tr_manifest = json.loads(tr_path.read_text(encoding="utf-8"))
    if tr_manifest["status"] != "complete":
        raise ValueError("TR source recovery incomplete")
    excluded = {}
    prechecks, manifests = {}, {}
    for atlas in ATLASES:
        records, checks = [], {}
        reference_header = None
        expected_rois = 200 if atlas == "cc200" else 116
        for sid in ids:
            reasons = []
            try:
                record = source_record(atlas, sid, tr_manifest["subjects"][sid])
                records.append(record)
                if reference_header is None:
                    reference_header = record["roi_order_sha256"]
                if record["n_rois"] != expected_rois:
                    reasons.append("unexpected_roi_count")
                if record["roi_order_sha256"] != reference_header:
                    reasons.append("roi_header_order_differs")
                if record["n_timepoints"] != tr_manifest["subjects"][sid]["n_timepoints"]:
                    reasons.append("timeseries_nifti_timepoint_mismatch")
                if record["constant_roi_count"]:
                    reasons.append("constant_roi_count={}".format(record["constant_roi_count"]))
                checks[sid] = dict(status="excluded" if reasons else "ok", reasons=reasons,
                                   n_rois=record["n_rois"], n_timepoints=record["n_timepoints"],
                                   constant_roi_count=record["constant_roi_count"],
                                   roi_order_sha256=record["roi_order_sha256"])
            except Exception as error:
                reasons.append(repr(error))
                checks[sid] = dict(status="excluded", reasons=reasons)
            if reasons:
                excluded.setdefault(sid, []).append(dict(atlas=atlas, stage="common_input_qc", reasons=reasons))
        raw_path = RAW / (atlas + "_source_manifest.json")
        write_json(raw_path, dict(schema_version="abide-raw-source-v1", atlas=atlas,
                   status="complete" if len(records) == len(ids) else "incomplete", subjects=ids, files=records))
        check_path = RAW / (atlas + "_input_precheck.json")
        write_json(check_path, dict(schema_version="abide-input-qc-v1", atlas=atlas,
                   original_n=len(ids), available=len(records), excluded_n=sum(c["status"] != "ok" for c in checks.values()), subjects=checks))
        prechecks[atlas] = dict(path=str(check_path), sha256=sha256(check_path))
        manifests[atlas] = dict(path=str(raw_path), sha256=sha256(raw_path))
        print("COMMON_QC_ATLAS atlas={} available={} excluded={}".format(
            atlas, len(records), sum(c["status"] != "ok" for c in checks.values())), flush=True)
    retained = [sid for sid in ids if sid not in excluded]
    target = ROOT / "inputs" / "common_qc_cohort.json"
    cohort = dict(schema_version="abide-common-qc-v1", status="ok", original_subject_ids=ids,
        subject_ids=retained, excluded=excluded, n_original=len(ids), n_retained=len(retained),
        n_excluded=len(excluded), original_counts=cohort_counts(ids), retained_counts=cohort_counts(retained),
        excluded_counts=cohort_counts(sorted(excluded)), policy="Exclude a subject from all three atlases and all methods if any atlas has constant/nonfinite/short input or invalid required primitive; preserve every original ROI; no imputation.",
        prechecks=prechecks, raw_manifests=manifests, tr_manifest=dict(path=str(tr_path), sha256=sha256(tr_path)),
        prepare_source=dict(path=str(Path(__file__)), sha256=sha256(__file__)),
        estimator_source=dict(path=str(ESTIMATOR_PATH), sha256=sha256(ESTIMATOR_PATH)),
        rgic_source=dict(path=str(RGIC_PATH), sha256=sha256(RGIC_PATH)), primitive_qc_exclusions=[])
    write_json(target, cohort)
    print("COMMON_QC_DONE original={} retained={} excluded={} classes={}".format(
        len(ids), len(retained), len(excluded), json.dumps(cohort["retained_counts"]["classes"])), flush=True)
    return cohort


def exclude_invalid_primitives(atlas, failures):
    path = ROOT / "inputs" / "common_qc_cohort.json"
    cohort = json.loads(path.read_text(encoding="utf-8"))
    for sid, failure in failures.items():
        cohort["excluded"].setdefault(sid, []).append(dict(atlas=atlas, stage="primitive_qc", reasons=[failure["error"]]))
        cohort["primitive_qc_exclusions"].append(dict(subject_id=sid, atlas=atlas, error=failure["error"]))
    cohort["subject_ids"] = [sid for sid in cohort["original_subject_ids"] if sid not in cohort["excluded"]]
    cohort["n_retained"] = len(cohort["subject_ids"])
    cohort["n_excluded"] = len(cohort["excluded"])
    cohort["retained_counts"] = cohort_counts(cohort["subject_ids"])
    cohort["excluded_counts"] = cohort_counts(sorted(cohort["excluded"]))
    write_json(path, cohort)
    print("COMMON_QC_UPDATED atlas={} removed={} retained={}".format(atlas, len(failures), cohort["n_retained"]), flush=True)


def compute_atlas(atlas, workers=2):
    """Two-worker float64 estimation, on-disk float32 arrays, atomic final NPZ."""
    estimator_module()  # Complete module initialization before worker threads.
    tr_path = RAW / "header_tr_manifest.json"
    tr_manifest = json.loads(tr_path.read_text(encoding="utf-8"))
    if tr_manifest["status"] != "complete":
        raise ValueError("Per-subject official TR recovery is incomplete")
    ids = analysis_subjects()
    original_ids = subjects()
    raw_manifest = RAW / (atlas + "_source_manifest.json")
    original_records = json.loads(raw_manifest.read_text(encoding="utf-8"))["files"]
    selected_ids = set(ids)
    records = [record for record in original_records if record["subject_id"] in selected_ids]
    failed = {}
    roi_orders = {record["roi_order_sha256"]: record["roi_ids"] for record in records}
    if len(roi_orders) != 1:
        failed["atlas_roi_order"] = dict(status="error", error="Original ROI headers differ across subjects")
    qc_path = RAW / (atlas + "_compute_qc.json")
    if failed:
        write_json(qc_path, dict(atlas=atlas, status="incomplete", n_subjects=len(ids), failed=failed))
        raise RuntimeError("Common QC failed for {} entries; no complete NPZ emitted".format(len(failed)))
    source_hashes = dict(prepare_source=dict(path=str(Path(__file__)), sha256=sha256(__file__)),
                         estimator_source=dict(path=str(ESTIMATOR_PATH), sha256=sha256(ESTIMATOR_PATH)),
                         rgic_source=dict(path=str(RGIC_PATH), sha256=sha256(RGIC_PATH)),
                         raw_manifest=dict(path=str(raw_manifest), sha256=sha256(raw_manifest)),
                         tr_manifest=dict(path=str(tr_path), sha256=sha256(tr_path)))
    lock = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    stage = RAW / "_primitives_work" / atlas
    stage.mkdir(parents=True, exist_ok=True)
    checkpoint = stage / "checkpoint.json"
    completed = {}
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if prior.get("lock") == lock:
            completed = prior["completed"]
    n_rois = records[0]["n_rois"]
    n_edges = n_rois*(n_rois-1)//2
    upper = np.triu_indices(n_rois, 1)
    lower = np.tril_indices(n_rois, -1)
    map_matrix = np.empty((n_rois, n_rois), dtype=int)
    map_matrix[lower] = np.arange(n_edges)
    map_matrix[(lower[1], lower[0])] = np.arange(n_edges)
    lower_to_upper = map_matrix[upper]
    arrays = {}
    for key in REQUIRED:
        dtype = np.int8 if key.endswith("_lag") else np.bool_ if key.endswith("_tie") else np.float32
        shape = (len(original_ids), n_edges, 2) if key.startswith(("te_", "var_", "phiid_")) else (len(original_ids), n_edges)
        target = stage / (key + ".npy")
        arrays[key] = np.lib.format.open_memmap(str(target), mode="r+" if completed else "w+", dtype=dtype, shape=shape)
    indices = {sid: index for index, sid in enumerate(original_ids)}
    todo = [sid for sid in ids if sid not in completed]
    print("COMPUTE_START atlas={} completed={} remaining={} workers={}".format(atlas, len(completed), len(todo), workers), flush=True)
    started = time.time()
    with h5py.File(SOURCE_H5, "r") as legacy:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(compute_one, atlas, sid, tr_manifest["subjects"][sid]["tr_seconds"]): sid for sid in todo}
            for future in concurrent.futures.as_completed(futures):
                futures.pop(future)
                sid, needed, qc = future.result()
                if qc["status"] == "ok":
                    group = legacy["patients"][sid][atlas]
                    agreement = {"r0_max_abs": float(np.max(np.abs(needed["r0"]-group["pearson"][:][lower_to_upper])))}
                    for window in ("W1", "W2"):
                        old = group[window.lower()]
                        agreement[window+"_rstar_max_abs"] = float(np.max(np.abs(needed[window+"_rstar"]-old["peak_rho"][:][lower_to_upper])))
                        agreement[window+"_lag_mismatches"] = int(np.sum(needed[window+"_lag"] != old["peak_lag"][:][lower_to_upper]))
                    qc["legacy_agreement"] = agreement
                    if max(agreement[key] for key in ("r0_max_abs", "W1_rstar_max_abs", "W2_rstar_max_abs")) > 1e-6:
                        qc = dict(status="error", error="Official raw input does not reproduce local primitives", legacy_agreement=agreement)
                    else:
                        for key in REQUIRED:
                            arrays[key][indices[sid]] = needed[key]
                        completed[sid] = qc
                if qc["status"] != "ok":
                    failed[sid] = qc
                    print("COMPUTE_ERROR atlas={} subject={} {}".format(atlas, sid, json.dumps(qc)), flush=True)
                count = sum(sid in completed for sid in ids)+len(failed)
                if count % 10 == 0 or count == len(ids):
                    for array in arrays.values():
                        array.flush()
                    write_json(checkpoint, dict(lock=lock, completed=completed))
                    print("COMPUTE_PROGRESS atlas={} {}/{} failed={} elapsed_s={:.1f}".format(
                        atlas, count, len(ids), len(failed), time.time()-started), flush=True)
    if failed:
        exclude_invalid_primitives(atlas, failed)
        ids = analysis_subjects()
        records = [record for record in original_records if record["subject_id"] in set(ids)]
    selected_completed = {sid: completed[sid] for sid in ids if sid in completed}
    write_json(qc_path, dict(atlas=atlas, status="complete" if len(selected_completed) == len(ids) else "incomplete", n_subjects=len(ids),
                            success=len(selected_completed), failed=failed, subjects=selected_completed))
    if len(selected_completed) != len(ids):
        raise RuntimeError("Incomplete common primitive cohort; no complete NPZ emitted")
    target = ROOT / "inputs" / (atlas + "_primitives.npz")
    temporary = target.with_suffix(".npz.partial")
    selected_rows = np.asarray([indices[sid] for sid in ids], dtype=int)
    with zipfile.ZipFile(str(temporary), mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        with archive.open("subject_ids.npy", "w", force_zip64=True) as handle:
            np.lib.format.write_array(handle, np.asarray(ids), allow_pickle=False)
        for key, array in arrays.items():
            with archive.open(key + ".npy", "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, array[selected_rows], allow_pickle=False)
    temporary.replace(target)
    agreement = {}
    for key in ("r0_max_abs", "W1_rstar_max_abs", "W2_rstar_max_abs"):
        agreement[key] = max(record["legacy_agreement"][key] for record in selected_completed.values())
    for key in ("W1_lag_mismatches", "W2_lag_mismatches"):
        agreement[key] = sum(record["legacy_agreement"][key] for record in selected_completed.values())
    manifest = dict(schema_version="abide-common-primitives-v1", status="complete", atlas=atlas,
        subject_ids=ids, n_subjects=len(ids), n_rois=n_rois, n_edges=n_edges,
        canonical_edge_order="np.triu_indices(n_rois,1)", edge_order="np.triu_indices(n_rois,1)",
        spectral_band_hz=list(SPECTRAL_BAND), temporal_refiltering=False,
        npz_file=str(target), npz_sha256=sha256(target),
        arrays=dict({key: dict(shape=[len(ids)]+list(value.shape[1:]), dtype=str(value.dtype)) for key, value in arrays.items()},
                    subject_ids=dict(shape=[len(ids)], dtype=str(np.asarray(ids).dtype))),
        source_files=records, roi_orders=roi_orders, roi_order_hashes=sorted(roi_orders),
        legacy_agreement=agreement, qc_file=str(qc_path), qc_sha256=sha256(qc_path), **source_hashes)
    write_json(ROOT / "inputs" / (atlas + "_primitives_manifest.json"), manifest)
    print("COMPUTE_DONE atlas={} subjects={} npz_bytes={} sha256={}".format(
        atlas, len(ids), target.stat().st_size, manifest["npz_sha256"]), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--headers-only", action="store_true")
    parser.add_argument("--qc-only", action="store_true")
    parser.add_argument("--compute-workers", type=int, default=2)
    parser.add_argument("--atlas", choices=ATLASES, action="append")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    atlases = tuple(args.atlas or ATLASES)
    if args.headers_only:
        recover_tr(args.workers)
        return
    if args.qc_only:
        make_common_cohort()
        return
    if not args.compute_only:
        download_all(atlases, args.workers)
    if not args.download_only:
        if not (RAW / "header_tr_manifest.json").exists():
            recover_tr(args.workers)
        if not (ROOT / "inputs" / "common_qc_cohort.json").exists():
            make_common_cohort()
        for atlas in atlases:
            compute_atlas(atlas, args.compute_workers)
        # Later-atlas numerical QC can shrink the common cohort. Re-export any
        # earlier cache from its unchanged 1035-row checkpoints before scoring.
        final_ids = analysis_subjects()
        for atlas in atlases:
            manifest = json.loads((ROOT / "inputs" / (atlas+"_primitives_manifest.json")).read_text(encoding="utf-8"))
            if manifest["subject_ids"] != final_ids:
                compute_atlas(atlas, args.compute_workers)


if __name__ == "__main__":
    main()
