from flask import Flask, render_template, jsonify, request
from pathlib import Path
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(__file__).resolve().parent / "Results"
APP1_DIR = RESULTS_DIR / "Approach1"
APP2_DIR = RESULTS_DIR / "Approach2"

TASK_SHEETS = ["SideToSide", "Pour", "VerticalSip", "VerticalShelf"]

def detect_task_from_excel(excel_path: Path) -> str:
    """
    Returns detected task name for this excel.
    Priority:
      1) any sheet in TASK_SHEETS that exists AND has >=1 row
      2) else: "Unknown"
    """
    try:
        xls = pd.ExcelFile(excel_path, engine="openpyxl")
        sheet_names = set(xls.sheet_names)

        # quick filter by expected task sheet names
        candidates = [t for t in TASK_SHEETS if t in sheet_names]
        if not candidates:
            return "Unknown"

        # ensure the candidate actually has data (not just an empty sheet)
        for task in candidates:
            try:
                df = pd.read_excel(excel_path, sheet_name=task, engine="openpyxl")
                if df is not None and len(df.index) > 0:
                    return task
            except Exception:
                continue

        return candidates[0]  # fallback
    except Exception:
        print(f"[detect_task_from_excel] Failed for {excel_path.name}: {e}")
        return "Unknown"

def detect_activities_from_results_txt(txt_path: Path) -> list[str]:
    """
    Read *_results.txt and return list of activity names found.
    Tries tab-separated column 'activity' first; fallback to BeginTask parsing.
    """
    txt_path = Path(txt_path)

    # A) Try TSV with an 'activity' column
    try:
        df = pd.read_csv(txt_path, sep="\t", comment="#", engine="python")
        cols = {c.strip().lower(): c for c in df.columns}

        if "activity" in cols:
            act_col = cols["activity"]
            d = df

            # optional: if 'is_task' exists, keep only is_task==1
            if "is_task" in cols:
                is_task_col = cols["is_task"]
                d = d.copy()
                d[is_task_col] = pd.to_numeric(d[is_task_col], errors="coerce")
                d = d[d[is_task_col] == 1]

            acts = (
                d[act_col].astype(str).str.strip()
                .replace({"": None, "nan": None, "None": None})
                .dropna().unique().tolist()
            )
            return sorted(set(acts))
    except Exception:
        pass

    # B) Fallback: scan raw lines for BeginTask/FinishTask
    acts = set()
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if "BeginTask" in s or "FinishTask" in s:
                    tokens = s.replace(",", " ").split()
                    for i, tok in enumerate(tokens):
                        if tok in ("BeginTask", "FinishTask") and i + 1 < len(tokens):
                            cand = tokens[i + 1].strip()
                            if cand:
                                acts.add(cand)
    except Exception:
        pass

    return sorted(acts)

    
def safe_sheet_name(s: str) -> str:
    # Excel sheet names are already safe; we just guard anyway
    return "".join(c for c in s if c.isalnum() or c in (" ", "_", "-")).strip()


def segment_by_timestamp(df: pd.DataFrame, ts_col="Timestamp(s)", gap_thresh=0.1) -> list[pd.DataFrame]:
    """
    Split into continuous timestamp segments ("episodes") based on:
      - timestamp reset (dt < 0)
      - big gap (dt > gap_thresh)
    Returns list of segment dataframes (each sorted by timestamp).
    """
    d = df.copy()
    d[ts_col] = pd.to_numeric(d[ts_col], errors="coerce")
    d = d.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    if len(d) < 3:
        return []

    ts = d[ts_col].to_numpy()
    dt = np.diff(ts)

    boundary = np.concatenate([[True], (dt < 0) | (dt > gap_thresh)])
    seg_id = np.cumsum(boundary)

    segments = []
    for _, g in d.groupby(seg_id):
        if len(g) >= 3:
            segments.append(g.reset_index(drop=True))
    return segments


def compute_njs_windows_absolute(
    seg: pd.DataFrame,
    ts_col="Timestamp(s)",
    a_col="Accel Mag (m/s^2)",
    window=1.0,
    eps=1e-9
) -> pd.DataFrame:
    """
    Compute windowed NJS for ONE segment using absolute timestamps.
    NJS proxy:
      jerk j = da/dt (from accel magnitude)
      Ej = ∫ j^2 dt
      Ea = ∫ a^2 dt
      NJS = (Ej * dur^2) / (Ea + eps)

    Returns dataframe: abs_t_mid, njs
    """
    d = seg[[ts_col, a_col]].copy()
    d[ts_col] = pd.to_numeric(d[ts_col], errors="coerce")
    d[a_col] = pd.to_numeric(d[a_col], errors="coerce")
    d = d.dropna().sort_values(ts_col).reset_index(drop=True)

    if len(d) < 3:
        return pd.DataFrame()

    t = d[ts_col].to_numpy()
    a = d[a_col].to_numpy()

    dt = np.diff(t)
    da = np.diff(a)

    valid = dt > 0
    dt = dt[valid]
    da = da[valid]
    if len(dt) < 2:
        return pd.DataFrame()

    j = da / dt
    t_j = (t[:-1] + t[1:])[valid] / 2.0  # absolute timestamps
    a_mid = (a[:-1] + a[1:])[valid] / 2.0

    t_start = float(t[0])
    t_end = float(t[-1])
    if t_end <= t_start:
        return pd.DataFrame()

    nbins = int(math.ceil((t_end - t_start) / window))
    rows = []

    for k in range(nbins):
        t0 = t_start + k * window
        t1 = min(t_start + (k + 1) * window, t_end)

        m = (t_j >= t0) & (t_j < t1)
        if not np.any(m):
            continue

        dur = t1 - t0
        Ej = float(np.sum((j[m] ** 2) * dt[m]))
        Ea = float(np.sum((a_mid[m] ** 2) * dt[m]))
        njs = (Ej * (dur ** 2)) / (Ea + eps)

        rows.append({"abs_t_mid": float((t0 + t1) / 2.0), "njs": float(njs)})

    return pd.DataFrame(rows)


def build_plotly_figure_for_excel(
    excel_path: Path,
    task_name: str,
    gap_thresh: float = 0.1,
    window: float = 1.0
) -> str:
    """
    Returns Plotly figure JSON for one excel (one task).
    X = absolute timestamps.
    Each segment becomes one trace.
    """
    df = pd.read_excel(excel_path, sheet_name=task_name, engine="openpyxl")

    # Minimal required columns
    ts_col = "Timestamp(s)"
    a_col = "Accel Mag (m/s^2)"
    if ts_col not in df.columns or a_col not in df.columns:
        raise ValueError(f"Missing required columns: {ts_col}, {a_col}")

    segments = segment_by_timestamp(df, ts_col=ts_col, gap_thresh=gap_thresh)
    if not segments:
        raise ValueError("No usable segments found.")

    fig = go.Figure()

    for i, seg in enumerate(segments, start=1):
        df_njs = compute_njs_windows_absolute(seg, ts_col=ts_col, a_col=a_col, window=window)
        if df_njs.empty:
            continue

        fig.add_trace(go.Scatter(
            x=df_njs["abs_t_mid"],
            y=df_njs["njs"],
            mode="lines+markers",
            name=f"segment {i}",
        ))

    fig.update_layout(
        title=f"{excel_path.name} | {task_name}",
        xaxis_title="Absolute Timestamp(s)",
        yaxis_title=f"NJS ({window:.2f}s windows)",
        margin=dict(l=60, r=25, t=60, b=60),
        legend=dict(orientation="h"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # ✅ soften grid + axis lines (dark theme)
    fig.update_xaxes(
        tickmode="linear",
        dtick=5,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",  # softer grid
        gridwidth=1,
        zeroline=False,
        linecolor="rgba(255,255,255,0.25)",  # softer axis line
        linewidth=1,
        ticks="outside",
        tickcolor="rgba(255,255,255,0.25)",
        ticklen=6,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        gridwidth=1,
        zeroline=False,
        linecolor="rgba(255,255,255,0.25)",
        linewidth=1,
        ticks="outside",
        tickcolor="rgba(255,255,255,0.25)",
        ticklen=6,
    )

    return fig.to_json()

def read_results_txt(txt_path: Path) -> pd.DataFrame:
    """
    Reads *_results.txt as TSV, skipping comment lines (# ...).
    """
    df = pd.read_csv(txt_path, sep="\t", comment="#", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df

def build_plotly_figure_for_results_txt(
    txt_path: Path,
    activity: str | None = None,
    dtick: float = 5
) -> str:
    """
    Plot windowed NJS across the entire session:
      - ONE continuous line joining all points (time order)
      - base line is dotted (whole session)
      - task segments are overlaid as solid red line (using shapes)
      - X = absolute timestamp (t_mid_abs)
    Optionally filters by activity if column exists and activity is provided.
    """
    df = read_results_txt(txt_path)

    # ---- normalize column names ----
    lower_map = {c.lower().strip(): c for c in df.columns}

    time_col = lower_map.get("t_mid_abs") or lower_map.get("t_mid") or lower_map.get("timestamp(s)")
    njs_col = lower_map.get("njs")
    is_task_col = lower_map.get("is_task")
    act_col = lower_map.get("activity")

    if not time_col:
        raise ValueError("No time column found. Expected 't_mid_abs' (preferred).")
    if not njs_col:
        raise ValueError("No 'njs' column found in results.txt")
    if not is_task_col:
        raise ValueError("No 'is_task' column found (needed for task highlighting).")

    # ---- optional activity filter ----
    if activity and act_col:
        df = df[df[act_col].astype(str).str.strip() == activity]

    # ---- numeric coercion + cleanup ----
    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[njs_col] = pd.to_numeric(df[njs_col], errors="coerce")
    df[is_task_col] = pd.to_numeric(df[is_task_col], errors="coerce")
    df = df.dropna(subset=[time_col, njs_col, is_task_col]).sort_values(time_col)

    if df.empty:
        raise ValueError("No usable rows after filtering/cleaning.")

    # ---- build ONE continuous trace ----
    # marker colors can vary per-point; line style cannot, so line is one style (dotted)
    is_task = (df[is_task_col].astype(float).fillna(0) == 1).to_numpy()
    marker_colors = np.where(is_task, "rgba(255,80,80,1)", "rgba(120,140,255,1)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df[njs_col],
        mode="lines+markers",
        name="NJS",
        line=dict(dash="dot", width=2, color="rgba(120,140,255,0.85)"),
        marker=dict(size=7, color=marker_colors),
        opacity=0.95
    ))

    # ---- overlay solid red segments for task parts using shapes ----
    # We draw a red line between consecutive points ONLY when both endpoints are task.
    x = df[time_col].to_numpy(dtype=float)
    y = df[njs_col].to_numpy(dtype=float)

    shapes = []
    for i in range(len(df) - 1):
        if is_task[i] and is_task[i + 1]:
            shapes.append(dict(
                type="line",
                x0=float(x[i]), y0=float(y[i]),
                x1=float(x[i + 1]), y1=float(y[i + 1]),
                line=dict(color="rgba(255,80,80,1)", width=4),
                layer="above",
            ))

    title = txt_path.stem
    if activity:
        title = f"{title} | {activity}"

    fig.update_layout(
        title=title,
        xaxis_title="Timestamp (s)",
        yaxis_title="IMU NJS (windowed)",
        margin=dict(l=60, r=25, t=60, b=60),
        showlegend=False,  # since we only have one trace now
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        shapes=shapes
    )

    # ✅ soften gridlines (same as your Approach 1 styling)
    fig.update_xaxes(
        tickmode="linear",
        dtick=dtick,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        gridwidth=1,
        zeroline=False,
        linecolor="rgba(255,255,255,0.20)",
        linewidth=1,
        ticks="outside",
        tickcolor="rgba(255,255,255,0.20)",
        ticklen=6,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        gridwidth=1,
        zeroline=False,
        linecolor="rgba(255,255,255,0.20)",
        linewidth=1,
        ticks="outside",
        tickcolor="rgba(255,255,255,0.20)",
        ticklen=6,
    )

    return fig.to_json()



@app.route("/")
def home():
    return render_template("home.html")


@app.route("/approach/<int:approach_id>")
def approach_subjects(approach_id):
    if approach_id == 1:
        base_dir = APP1_DIR
        approach_name = "Approach 1"
    elif approach_id == 2:
        base_dir = APP2_DIR
        approach_name = "Approach 2"
    else:
        return "Invalid approach", 404

    subjects = []
    if base_dir.exists():
        subjects = sorted([p.name for p in base_dir.iterdir() if p.is_dir()])

    return render_template(
        "approach_subjects.html",
        approach_id=approach_id,
        approach_name=approach_name,
        subjects=subjects,
        breadcrumbs=[
            {"label": "Home", "url": "/"},
            {"label": approach_name, "url": None},
        ]
    )


@app.get("/approach/<int:approach_id>/<subject_id>")
def approach_subject(subject_id, approach_id):
    if approach_id == 1:
        approach_dir = APP1_DIR
        approach_name = "Approach 1"
    elif approach_id == 2:
        approach_dir = RESULTS_DIR / "Approach2"
        approach_name = "Approach 2"
    else:
        return "Invalid approach", 404

    subject_dir = approach_dir / subject_id

    hands = []
    if subject_dir.exists():
        hands = sorted(p.name for p in subject_dir.iterdir() if p.is_dir())

    breadcrumbs = [
        {"label": "Home", "url": "/"},
        {"label": approach_name, "url": f"/approach/{approach_id}"},
        {"label": f"Subject {subject_id}", "url": None},
    ]

    return render_template(
        "hands.html",
        approach_id=approach_id,
        approach_name=approach_name,
        subject_id=subject_id,
        hands=hands,
        breadcrumbs=breadcrumbs
    )



@app.get("/approach/<int:approach_id>/<subject_id>/<hand>")
def approach_hand_files(approach_id, subject_id, hand):

    # ---------- decide base directory ----------
    if approach_id == 1:
        base_dir = APP1_DIR
        approach_name = "Approach 1"
    elif approach_id == 2:
        base_dir = APP2_DIR
        approach_name = "Approach 2"
    else:
        return "Invalid approach", 404

    hand_dir = base_dir / subject_id / hand

    items = []

    # ---------- Approach 1: Excel-based ----------
    if approach_id == 1:
        if hand_dir.exists():
            for xlsx in sorted(hand_dir.glob("*.xlsx")):
                task = detect_task_from_excel(xlsx)
                items.append({
                    "file": xlsx.name,
                    "task": task
                })

    # ---------- Approach 2: results.txt-based ----------
    elif approach_id == 2:
        if hand_dir.exists():
            for txt in sorted(hand_dir.glob("*_results.txt")):
                acts = detect_activities_from_results_txt(txt)

                # show at least one tile even if no activity detected
                if not acts:
                    items.append({
                        "file": txt.name,
                        "task": "Unknown"
                    })
                else:
                    for a in acts:
                        items.append({
                            "file": txt.name,
                            "task": a
                        })

    breadcrumbs = [
        {"label": "Home", "url": "/"},
        {"label": approach_name, "url": f"/approach/{approach_id}"},
        {"label": f"Subject {subject_id}", "url": f"/approach/{approach_id}/{subject_id}"},
        {"label": hand, "url": None},
    ]

    return render_template(
        "files.html",
        approach_id=approach_id,
        subject_id=subject_id,
        hand=hand,
        items=items,
        breadcrumbs=breadcrumbs
    )


@app.get("/approach/<int:approach_id>/<subject_id>/<hand>/<path:file_name>/plot")
def approach_plot(approach_id, subject_id, hand, file_name):
    if approach_id == 1:
        base_dir = APP1_DIR
        approach_name = "Approach 1"
    elif approach_id == 2:
        base_dir = APP2_DIR
        approach_name = "Approach 2"
    else:
        return "Invalid approach", 404

    file_path = base_dir / subject_id / hand / file_name
    if not file_path.exists():
        return f"File not found: {file_name}", 404

    # For now:
    # - Approach 1: Excel -> detect task
    # - Approach 2: later you will plot from results txt (skip logic for now if you want)
    task = "Unknown"
    if approach_id == 1:
        task = detect_task_from_excel(file_path)
    else:
        # temporary label until plotting logic is implemented
        # you can also detect activity name from txt here if you want
        task = request.args.get("activity") or "Unknown"

    breadcrumbs = [
        {"label": "Home", "url": "/"},
        {"label": approach_name, "url": f"/approach/{approach_id}"},
        {"label": f"Subject {subject_id}", "url": f"/approach/{approach_id}/{subject_id}"},
        {"label": hand, "url": f"/approach/{approach_id}/{subject_id}/{hand}"},
        {"label": task, "url": None},
    ]

    return render_template(
        "plot.html",
        approach_id=approach_id,
        subject_id=subject_id,
        hand=hand,
        file_name=file_name,     # <-- use this name in template now
        task=task,
        breadcrumbs=breadcrumbs
    )


@app.get("/api/approach/<int:approach_id>/<subject_id>/<hand>/<path:file_name>")
def api_approach_plot(approach_id, subject_id, hand, file_name):
    if approach_id == 1:
        base_dir = APP1_DIR
    elif approach_id == 2:
        base_dir = APP2_DIR
    else:
        return jsonify({"error": "Invalid approach"}), 404

    file_path = base_dir / subject_id / hand / file_name
    if not file_path.exists():
        return jsonify({"error": f"File not found: {file_name}"}), 404

    # ✅ Approach 1: your current Excel plotting logic stays same
    if approach_id == 1:
        task = detect_task_from_excel(file_path)
        if task == "Unknown":
            return jsonify({"error": "No task sheet found in this excel."}), 400

        gap_thresh = 0.1
        window = 1.0

        try:
            fig_json = build_plotly_figure_for_excel(
                file_path,
                task_name=task,
                gap_thresh=gap_thresh,
                window=window
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        return jsonify({
            "task": task,
            "fig_json": fig_json,
            "meta": {"gap_thresh": gap_thresh, "window": window}
        })

    # ✅ Approach 2: results.txt plotting (solid=task, dotted=other)
    if approach_id == 2:
        # optional activity passed as query param: ?activity=SideToSide
        activity = (request.args.get("activity") or "").strip() or None

        try:
            fig_json = build_plotly_figure_for_results_txt(
                file_path,
                activity=activity,
                dtick=5
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        return jsonify({
            "task": activity or "Continuous",
            "fig_json": fig_json,
            "meta": {"dtick": 5, "activity": activity}
        })




if __name__ == "__main__":
    app.run(debug=True)
