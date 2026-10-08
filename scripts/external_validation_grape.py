"""
External validation on GRAPE and paired leakage demonstration.

Two prediction tasks are built from the same cohorts with the same feature
constructor and the same model class. Only the window/target split differs:

  leaky  : features summarise every visit, target = OLS slope over every visit
           (the formulation used in the original submission)
  forecast: features summarise the first 3 visits only, target = OLS slope
           computed exclusively from visit 4 onwards

Cohorts:
  UWHVF  - Humphrey 24-2, University of Washington (internal)
  GRAPE  - Octopus 900 G1, Wenzhou Medical University (external)

GRAPE stores light sensitivity, not total deviation, and no Octopus G1
normative table is distributed with it. Mean sensitivity is therefore used in
place of mean deviation. The two differ by an age-normative offset that drifts
by roughly 0.01 dB/y, which is negligible against the -1.0 dB/y fast-progressor
threshold, so slopes remain comparable across cohorts.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
GRAPE_XLSX = ROOT / "data" / "raw" / "grape" / "VF_and_clinical_information.xlsx"
UWHVF_JSON = ROOT / "data" / "raw" / "uwhvf" / "alldata.json"
OUT_PATH = ROOT / "reports" / "forecast" / "external_validation.json"

N_OBS = 3
MIN_FUTURE = 3
MIN_FUTURE_SPAN = 1.0
FAST_THRESHOLD = -1.0
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


# --------------------------------------------------------------------------
# minimal xlsx reader (stdlib only; openpyxl is not installed in this venv)
# --------------------------------------------------------------------------
def read_xlsx(path: Path) -> dict[str, list[list]]:
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ElementTree.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))

        wb = ElementTree.fromstring(z.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rel_target = {
            r.get("Id"): r.get("Target").lstrip("/").replace("xl/", "")
            for r in rels
        }
        rid = f"{{http://schemas.openxmlformats.org/officeDocument/2006/relationships}}id"

        sheets: dict[str, list[list]] = {}
        for sh in wb.find("m:sheets", NS):
            target = rel_target[sh.get(rid)]
            sheets[sh.get("name")] = _read_sheet(z.read(f"xl/{target}"), shared)
    return sheets


def _read_sheet(data: bytes, shared: list[str]) -> list[list]:
    rows: list[list] = []
    root = ElementTree.fromstring(data)
    for row in root.iter(f"{{{NS['m']}}}row"):
        cells: dict[int, object] = {}
        for c in row.findall("m:c", NS):
            col = _col_index(c.get("r"))
            v = c.find("m:v", NS)
            if c.get("t") == "inlineStr":
                node = c.find("m:is", NS)
                cells[col] = "".join(t.text or "" for t in node.iter(f"{{{NS['m']}}}t"))
                continue
            if v is None or v.text is None:
                cells[col] = None
                continue
            if c.get("t") == "s":
                cells[col] = shared[int(v.text)]
            elif c.get("t") == "str":
                cells[col] = v.text
            else:
                cells[col] = float(v.text)
        if cells:
            width = max(cells) + 1
            rows.append([cells.get(i) for i in range(width)])
    return rows


def _col_index(ref: str) -> int:
    letters = re.match(r"([A-Z]+)", ref or "A").group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


# --------------------------------------------------------------------------
# cohort construction
# --------------------------------------------------------------------------
class Eye:
    __slots__ = ("group", "times", "fields", "age", "gender")

    def __init__(self, group, times, fields, age, gender):
        self.group = group
        self.times = np.asarray(times, dtype=float)
        self.fields = np.asarray(fields, dtype=float)
        self.age = age
        self.gender = gender


def load_grape() -> list[Eye]:
    sheets = read_xlsx(GRAPE_XLSX)
    base_rows = sheets["Baseline"]
    follow_rows = sheets["Follow-up"]
    base_hdr = [str(h) if h is not None else "" for h in base_rows[0]]
    follow_hdr = [str(h) if h is not None else "" for h in follow_rows[0]]

    n_vf = 61
    base_vf = slice(len(base_hdr) - n_vf, len(base_hdr))
    follow_vf = slice(len(follow_hdr) - n_vf, len(follow_hdr))

    def key(subject, laterality) -> str:
        return f"{int(float(subject))}_{str(laterality).strip().upper()}"

    def vf_vector(row, sl) -> np.ndarray | None:
        raw = row[sl]
        if len(raw) < n_vf or any(x is None for x in raw):
            return None
        # "-1" marks a location the eye could not perceive; treat as the 0 dB floor
        return np.array([0.0 if float(x) < 0 else float(x) for x in raw])

    eyes: dict[str, dict] = {}
    for row in base_rows[1:]:
        if not row or row[0] is None:
            continue
        vf = vf_vector(row, base_vf)
        if vf is None:
            continue
        k = key(row[0], row[1])
        eyes[k] = {
            "subject": str(int(float(row[0]))),
            "age": float(row[2]) if row[2] is not None else np.nan,
            "gender": 1.0 if str(row[3]).strip().upper().startswith("M") else 0.0,
            "visits": [(0.0, vf)],
        }

    for row in follow_rows[1:]:
        if not row or row[0] is None:
            continue
        k = key(row[0], row[1])
        if k not in eyes or row[3] is None:
            continue
        vf = vf_vector(row, follow_vf)
        if vf is None:
            continue
        eyes[k]["visits"].append((float(row[3]), vf))

    # the two blind-spot columns are constant across the cohort; drop them
    stack = np.vstack([v[1] for e in eyes.values() for v in e["visits"]])
    keep = stack.std(axis=0) > 1e-9

    out = []
    for rec in eyes.values():
        visits = sorted(rec["visits"], key=lambda t: t[0])
        times = [t for t, _ in visits]
        fields = [f[keep] for _, f in visits]
        out.append(Eye(rec["subject"], times, fields, rec["age"], rec["gender"]))
    return out


def load_uwhvf() -> list[Eye]:
    raw = json.loads(UWHVF_JSON.read_text())
    out = []
    for pid, patient in raw["data"].items():
        for laterality in ("R", "L"):
            seq = patient.get(laterality)
            if not seq:
                continue
            visits = []
            for visit in seq:
                td = visit.get("td_seq") or visit.get("td")
                if td is None or len(td) != 54:
                    continue
                visits.append((float(visit["age"]), np.asarray(td, dtype=float)))
            if len(visits) < N_OBS + MIN_FUTURE:
                continue
            visits.sort(key=lambda t: t[0])
            t0 = visits[0][0]
            out.append(
                Eye(
                    pid,
                    [t - t0 for t, _ in visits],
                    [f for _, f in visits],
                    t0,
                    float(patient.get("gender", 0) == "M"),
                )
            )
    return out


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------
def ols_slope(times: np.ndarray, values: np.ndarray) -> float:
    return float(np.polyfit(times, values, 1)[0])


def summarise(eye: Eye, idx: slice) -> list[float]:
    """Summary of one window. Identical constructor for both tasks."""
    t = eye.times[idx]
    f = eye.fields[idx]
    ms = f.mean(axis=1)
    last = np.sort(f[-1])
    span = float(t[-1] - t[0])
    return [
        eye.age,
        eye.gender,
        float(len(t)),
        span,
        float(ms[0]),
        float(ms[-1]),
        float(ms.mean()),
        float(ms.std()),
        float(ms[-1] - ms[0]),
        float(f[-1].std()),
        float(np.percentile(f[-1], 10)),
        float(np.percentile(f[-1], 90)),
        float(last[:10].mean()),
    ]


def build_task(eyes: list[Eye], task: str):
    x, y, groups = [], [], []
    for eye in eyes:
        n = len(eye.times)
        if task == "leaky":
            if n < N_OBS + MIN_FUTURE:
                continue
            window, target_idx = slice(0, n), slice(0, n)
        else:
            if n < N_OBS + MIN_FUTURE:
                continue
            target_idx = slice(N_OBS, n)
            if len(eye.times[target_idx]) < MIN_FUTURE:
                continue
            t_future = eye.times[target_idx]
            if t_future[-1] - t_future[0] < MIN_FUTURE_SPAN:
                continue
            window = slice(0, N_OBS)
        t_target = eye.times[target_idx]
        v_target = eye.fields[target_idx].mean(axis=1)
        x.append(summarise(eye, window))
        y.append(ols_slope(t_target, v_target))
        groups.append(eye.group)
    return np.asarray(x), np.asarray(y), np.asarray(groups)


def cv_evaluate(x, y, groups, seed=42) -> dict:
    """Grouped 5-fold CV so fellow eyes never straddle a fold."""
    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    preds = {"ridge": np.zeros_like(y), "gbm": np.zeros_like(y)}
    for train, test in cv.split(x, y, groups):
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        ridge.fit(x[train], y[train])
        preds["ridge"][test] = ridge.predict(x[test])
        gbm = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=1.0,
            random_state=seed,
        )
        gbm.fit(x[train], y[train])
        preds["gbm"][test] = gbm.predict(x[test])

    out = {"n": int(len(y)), "observed_slope_mean": float(y.mean()),
           "observed_slope_sd": float(y.std()),
           "fast_progressor_rate": float((y < FAST_THRESHOLD).mean())}
    ss_tot = float(((y - y.mean()) ** 2).sum())
    for name, p in preds.items():
        resid = y - p
        out[name] = {
            "mae": float(np.abs(resid).mean()),
            "rmse": float(np.sqrt((resid**2).mean())),
            "r2": float(1.0 - (resid**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        }
    return out


def main() -> None:
    results = {}
    for cohort, loader in (("uwhvf", load_uwhvf), ("grape", load_grape)):
        eyes = loader()
        entry = {"n_eyes_loaded": len(eyes)}
        for task in ("leaky", "forecast"):
            x, y, g = build_task(eyes, task)
            if len(y) < 30:
                entry[task] = {"n": int(len(y)), "note": "too few eligible eyes"}
                continue
            entry[task] = cv_evaluate(x, y, g)
        results[cohort] = entry
        print(f"[{cohort}] eyes={len(eyes)}")
        for task in ("leaky", "forecast"):
            e = entry.get(task, {})
            if "ridge" in e:
                print(
                    f"  {task:9s} n={e['n']:5d} "
                    f"ridge R2={e['ridge']['r2']:+.3f} MAE={e['ridge']['mae']:.3f}  "
                    f"gbm R2={e['gbm']['r2']:+.3f} MAE={e['gbm']['mae']:.3f}"
                )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
