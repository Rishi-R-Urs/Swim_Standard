"""
file: Swim_Standard.py
project: Swim_Standard
author: Rishi Urs

Overview
- Coach-facing dashboard comparing student times vs Indian records (CSV only).
- Tabs: Compare (gap bar), Exercises (stroke/bucket table), Records (full table),
        Analytics (record speed vs distance), Pace Predictions (power-law pace curve).
- Robust parsing (mm:ss.xx / mm.ss.cs / ss.xx), stroke normalization, mixed-relay awareness.

Unique points
- Stable offline workflow: `sfi_records.csv` (records) + `swim_exercises.csv` (recommendations).
- Pace-curve model predicts times across distances and highlights underperforming events- Unsupervised ML model
- Clean, legible UI; tuned legends/margins to avoid overlap.
"""

from __future__ import annotations

from pathlib import Path
import math, re
import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
import plotly.express as px

# ---------- Config ----------
HERE = Path(__file__).parent
RECORDS_CSV = HERE / "sfi_records.csv"
EXERCISES_CSV = HERE / "swim_exercises.csv"
PORT = 5010

PLOTLY_FONT = dict(size=14)
PLOTLY_LEGEND_TOP = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
PLOTLY_LEGEND_BOTTOM = dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5)

# ---------- Panel / Theme ----------
pn.extension("tabulator", "plotly", notifications=True, raw_css=[r"""
:root { --brand:#0a2540; --accent:#12a594; --ink:#0b1f32; --muted:#6b7280; }
.bk .bk-header { background:var(--brand)!important; }
.bk .bk-header .bk-typography{ color:#fff!important; font-weight:700; letter-spacing:.3px; }
.bk-tabs-header{ background:#fff!important; border-bottom:1px solid #e5e7eb; }
.bk-tabs-header .bk-tab{ font-weight:700; font-size:14px; padding:10px 14px; color:var(--ink)!important; }
.bk-tabs-header .bk-tab.bk-active{ border-bottom:3px solid var(--accent); }
.card-title{ font-weight:700; color:var(--ink); }
.hint{ color:var(--muted); font-size:12.5px; }
"""])

# ---------- Utilities ----------
VALID_EVENTS = {
    "Freestyle": [50,100,200,400,800,1500],
    "Backstroke": [50,100,200],
    "Breaststroke": [50,100,200],
    "Butterfly": [50,100,200],
    "Individual Medley": [200,400],
}

def canonical_stroke(name: str) -> str:
    n = (name or "").strip().lower()
    if "breast" in n: return "Breaststroke"
    if "back" in n: return "Backstroke"
    if "butterfly" in n: return "Butterfly"
    if "freestyle" in n and "relay" not in n: return "Freestyle"
    if "individual medley" in n or (("medley" in n) and "relay" not in n):
        return "Individual Medley"
    return name.strip().title()

def is_relay(stroke: str) -> bool:
    return "relay" in (stroke or "").lower()

# Parsing supports mm:ss.xx, mm.ss.cs, and ss.xx / ss.xxx
def to_seconds(text: str | float | int | None) -> float | None:
    if text is None or (isinstance(text, float) and math.isnan(text)): return None
    s = str(text).strip()
    m = re.match(r"^(\d{1,2}):(\d{2})(?:\.(\d+))?$", s)
    if m:
        mm, ss = int(m.group(1)), int(m.group(2))
        frac = float("0."+m.group(3)) if m.group(3) else 0.0
        return mm*60 + ss + frac
    m = re.match(r"^(\d{1,2})\.(\d{2})\.(\d{2})$", s)
    if m:
        mm, ss, cs = map(int, m.groups())
        return mm*60 + ss + cs/100.0
    m = re.match(r"^(\d{1,2}\.\d{2,3})$", s)
    if m:
        return float(m.group(1))
    return None

def fmt_seconds(sec: float | None) -> str:
    if sec is None: return "—"
    if sec < 60: return f"{sec:.2f}"
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:05.2f}"

def numeric_distance(event_m: str) -> int | None:
    if event_m is None or str(event_m).strip()=="":
        return None
    s = str(event_m).lower().strip()
    if "x" in s:  # relay
        return None
    m = re.search(r"\d{2,4}", s)
    return int(m.group()) if m else None

def bucket_for_distance(d: int | None) -> str | None:
    if d is None: return None
    if d <= 100: return "Sprint"
    if d <= 400: return "Middle"
    return "Distance"

def gender_norm(g: str, stroke: str, event: str) -> str:
    s = (g or "").lower()
    if "female" in s: return "Female"
    if "male" in s: return "Male"
    if "mixed" in (stroke or "").lower() or "mixed" in (event or "").lower():
        return "Mixed"
    return "Male" if "women" not in s else "Female"

# ---------- Load Data ----------
def load_records(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "time_s" not in df.columns:
        df["time_s"] = df["time_str"].apply(to_seconds)
    df["stroke_clean"] = df["stroke"].apply(canonical_stroke)
    df["is_relay"] = df["stroke"].str.lower().str.contains("relay")
    df["gender_norm"] = df.apply(lambda r: gender_norm(str(r.get("gender","")), str(r.get("stroke","")), str(r.get("event",""))), axis=1)
    if "event_m" not in df.columns:
        df["event_m"] = df["event"].astype(str).str.extract(r"(\d+)")
    df["distance_m"] = df["event_m"].apply(numeric_distance)
    cat_map = {
        "Best Indian Performance":"Best Indian Performance",
        "Seniors":"Seniors",
        "Juniors":"Juniors",
        "Sub Juniors":"Sub Juniors"
    }
    df["category"] = df["category"].map(lambda c: cat_map.get(str(c), str(c)))
    df = df.dropna(subset=["time_s"]).reset_index(drop=True)
    return df

def load_exercises(csv_path: Path) -> pd.DataFrame:
    try:
        ex = pd.read_csv(csv_path)
    except Exception:
        ex = pd.DataFrame()
    if not ex.empty and "stroke" in ex.columns:
        ex["stroke_clean"] = ex["stroke"].apply(canonical_stroke)
    else:
        ex["stroke_clean"] = None
    if "event_bucket" not in ex.columns:
        if "event_m" in ex.columns:
            ex["event_bucket"] = ex["event_m"].apply(lambda v: bucket_for_distance(numeric_distance(v)))
        else:
            ex["event_bucket"] = None
    return ex

records = load_records(RECORDS_CSV)
exercises = load_exercises(EXERCISES_CSV)

# ---------- Widgets ----------
category_opts = ["Best Indian Performance","Seniors","Juniors","Sub Juniors"]
category_select = pn.widgets.Select(name="Category", options=category_opts, value="Seniors")
gender_select   = pn.widgets.Select(name="Gender", options=["Male","Female"], value="Male")
stroke_select   = pn.widgets.Select(name="Stroke", options=list(VALID_EVENTS.keys()), value="Freestyle")

def event_options(stroke: str):
    return VALID_EVENTS.get(stroke, [50,100,200])

event_select   = pn.widgets.Select(name="Event Distance (m)", options=event_options(stroke_select.value), value=50)
student_name   = pn.widgets.TextInput(name="Student Name (optional)", placeholder="Name")
student_time   = pn.widgets.TextInput(name="Student Time", placeholder="mm:ss.xx or ss.xx (e.g., 1:02.50 or 62.50)")
reset_btn      = pn.widgets.Button(name="Reset Filters", button_type="primary")
include_mixed  = pn.widgets.Checkbox(name="Include Mixed Relays in Records table", value=False)

chart_w = pn.widgets.IntSlider(name="Chart Width", start=640, end=1400, step=40, value=920)
chart_h = pn.widgets.IntSlider(name="Chart Height", start=320, end=900, step=20, value=460)
hint_md = pn.pane.Markdown("<div class='hint'>Enter time like <code>1:02.50</code> or <code>62.50</code>. Compare tab supports individual events only.</div>")

def _reset(_=None):
    category_select.value = "Seniors"
    gender_select.value   = "Male"
    stroke_select.value   = "Freestyle"
    event_select.options  = event_options("Freestyle")
    event_select.value    = 50
    student_name.value    = ""
    student_time.value    = ""
    chart_w.value         = 920
    chart_h.value         = 460
reset_btn.on_click(_reset)

def _on_stroke(evt):
    event_select.options = event_options(evt.new)
    if event_select.value not in event_select.options:
        event_select.value = event_select.options[0]
stroke_select.param.watch(_on_stroke, "value")

# ---------- Compare ----------
def best_record_row(category: str, gender: str, stroke: str, distance: int) -> pd.Series | None:
    df = records.copy()
    df = df[(df["category"]==category)&(df["gender_norm"]==gender)]
    df = df[(df["stroke_clean"]==stroke)&(~df["is_relay"])]
    df = df[pd.to_numeric(df["distance_m"], errors="coerce")==int(distance)]
    if df.empty: return None
    return df.sort_values("time_s").iloc[0]

def compare_fig(category, gender, stroke, distance, student_time_text, w, h):
    if is_relay(stroke):
        fig = go.Figure()
        fig.add_annotation(text="Compare supports individual strokes only.", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
        fig.update_layout(width=w, height=h, template="plotly_white", title="Compare",
                          margin=dict(l=40,r=40,t=40,b=40), font=PLOTLY_FONT, legend=PLOTLY_LEGEND_TOP)
        return fig, None

    rec = best_record_row(category, gender, stroke, distance)
    s_sec = to_seconds(student_time_text) if student_time_text else None

    labels, vals, txts = [], [], []
    meta = None

    if rec is not None:
        labels.append("Indian Record")
        vals.append(float(rec["time_s"]))
        txts.append(fmt_seconds(rec["time_s"]))
        meta = {
            "holder": rec.get("record_holder", ""),
            "team": rec.get("team",""),
            "date_event": rec.get("date_event",""),
            "time": fmt_seconds(rec["time_s"])
        }

    if s_sec is not None:
        labels.append("Student")
        vals.append(s_sec)
        txts.append(fmt_seconds(s_sec))

    fig = go.Figure()
    if labels:
        fig.add_bar(x=labels, y=vals, text=txts, textposition="auto")
    title = f"{stroke} {int(distance)}m — {category} • {gender}"
    fig.update_layout(width=w, height=h, template="plotly_white",
                      title=title, margin=dict(l=40,r=40,t=50,b=50),
                      font=PLOTLY_FONT, legend=PLOTLY_LEGEND_TOP)
    fig.update_yaxes(title="Time (s) — lower is faster", gridcolor="#e5e7eb")

    if (rec is not None) and (s_sec is not None):
        delta = s_sec - float(rec["time_s"])
        pct = (delta/float(rec["time_s"])) * 100 if float(rec["time_s"])>0 else 0.0
        fig.add_annotation(
            text=f"Gap: {delta:+.2f}s ({pct:+.1f}%)",
            x=0.5, y=max(vals)*0.92, xref="paper", yref="y", showarrow=False
        )

    if not labels:
        fig.add_annotation(text="Enter a valid student time to compare.",
                           x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    return fig, meta

def compare_panel(category, gender, stroke, distance, student_time_text, w, h, student_name_text):
    fig, meta = compare_fig(category, gender, stroke, distance, student_time_text, w, h)
    if meta:
        who = f" ({student_name_text.strip()})" if student_name_text.strip() else ""
        info = (
            f"**Record:** {meta['time']} by **{meta['holder']}**  \n"
            f"**Event:** {meta['date_event'] or '—'}  \n"
            f"**Team:** {meta['team'] or '—'}{who}"
        )
        details = pn.pane.Markdown(info)
    else:
        details = pn.pane.Markdown("—")
    return pn.Column(fig, details, sizing_mode="stretch_both")

compare_pane = pn.bind(
    compare_panel,
    category_select, gender_select, stroke_select, event_select,
    student_time, chart_w, chart_h, student_name
)

# ---------- Exercises ----------
def exercises_table(stroke: str, distance: int):
    if exercises.empty:
        return pn.pane.Markdown("**Add `swim_exercises.csv` to see recommendations.**")
    df = exercises.copy()
    df = df[df["stroke_clean"]==stroke] if "stroke_clean" in df.columns else df

    bucket = bucket_for_distance(distance)
    if "event_bucket" in df.columns and bucket is not None:
        sub = df[df["event_bucket"].fillna("").str.lower()==bucket.lower()]
        if not sub.empty: df = sub

    pref = ["stroke","event_m","event_bucket","exercise_name","exercise_type","equipment","primary_muscles",
            "sets","reps_or_time","distance_or_load","interval_or_rest_s","RPE_1_10","progression","regression","notes"]
    cols = [c for c in pref if c in df.columns]
    if not cols:
        cols = df.columns.tolist()
    return pn.widgets.Tabulator(
        df[cols].reset_index(drop=True),
        pagination="remote", page_size=12, sizing_mode="stretch_width", height=360, show_index=False
    )

exercises_pane = pn.bind(exercises_table, stroke_select, event_select)

# ---------- Records Table ----------
def records_table(category: str, gender: str, include_mixed_relays: bool):
    df = records.copy()
    df = df[df["category"]==category]
    mask = df["gender_norm"]==gender
    if include_mixed_relays:
        mask = mask | ((df["gender_norm"]=="Mixed") & (df["is_relay"]))
    df = df[mask]
    show_cols = ["category","gender_norm","stroke","event","record_holder","time_str","team","date_event"]
    df = df[show_cols].rename(columns={"gender_norm":"gender"})
    return pn.widgets.Tabulator(
        df.reset_index(drop=True), pagination="remote", page_size=18,
        sizing_mode="stretch_width", height=420, show_index=False
    )

records_pane = pn.bind(records_table, category_select, gender_select, include_mixed)

# ---------- Analytics (legend moved to bottom to avoid overlap) ----------
def speed_vs_distance(category: str, gender: str):
    df = records.copy()
    df = df[(df["category"]==category)&(df["gender_norm"]==gender)&(~df["is_relay"])]
    df = df.dropna(subset=["distance_m","time_s"])
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Record Speed vs Distance (no data)",
                          font=PLOTLY_FONT, legend=PLOTLY_LEGEND_BOTTOM, margin=dict(b=90))
        return fig
    df["stroke_group"] = df["stroke_clean"]
    df["speed_mps"] = df["distance_m"]/df["time_s"]
    fig = px.scatter(df, x="distance_m", y="speed_mps", color="stroke_group",
                     hover_data=["record_holder","time_str","event"],
                     labels={"distance_m":"Event distance (m)","speed_mps":"Speed (m/s)","stroke_group":"Stroke"})
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(template="plotly_white",
                      title=f"Record Speed vs Distance — {category} • {gender}",
                      font=PLOTLY_FONT, legend=PLOTLY_LEGEND_BOTTOM,
                      margin=dict(l=40, r=40, t=60, b=90))
    fig.update_xaxes(gridcolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb")
    return fig

analytics_pane = pn.bind(speed_vs_distance, category_select, gender_select)

# ---------- Pace Predictions (power-law) ----------
ml_tip = pn.pane.Markdown(
    "<div class='hint'>Add 2+ distances to fit a student curve. With one time, the model uses the "
    "record exponent and calibrates at that distance.</div>"
)

# Collapsible editor removes visual gap when unused
_times_df_empty = pd.DataFrame(columns=["distance_m","time_str"])
times_editor = pn.widgets.Tabulator(value=_times_df_empty, show_index=False, height=140, sizing_mode="stretch_width")
times_acc = pn.Accordion(("Student Times (optional)", times_editor), active=[])

ml_seed_btn  = pn.widgets.Button(name="Use current event & time", button_type="primary")
ml_clear_btn = pn.widgets.Button(name="Clear table", button_type="default")

def _seed_from_current(event=None):
    d = int(event_select.value) if event_select.value else None
    t = student_time.value.strip()
    if not d or not t:
        pn.state.notifications.info("Enter a valid Student Time and Event first.")
        return
    df = times_editor.value if isinstance(times_editor.value, pd.DataFrame) else _times_df_empty.copy()
    if "distance_m" not in df.columns:
        df = pd.DataFrame(columns=["distance_m","time_str"])
    if not df.empty and (df["distance_m"].astype(str) == str(d)).any():
        df.loc[df["distance_m"].astype(str) == str(d), "time_str"] = t
    else:
        df = pd.concat([df, pd.DataFrame([{"distance_m": d, "time_str": t}])], ignore_index=True)
    times_editor.value = df
    if times_acc.active == []:
        times_acc.active = [0]

ml_seed_btn.on_click(_seed_from_current)

def _ml_clear(event=None):
    times_editor.value = _times_df_empty.copy()
    times_acc.active = []

ml_clear_btn.on_click(_ml_clear)

# Power-law fit in log–log space: t = a * d^b
def fit_power_curve(distances_m, times_s):
    x = np.log(np.asarray(distances_m, dtype=float))
    y = np.log(np.asarray(times_s, dtype=float))
    b, log_a = np.polyfit(x, y, 1)
    a = np.exp(log_a)
    return a, b

def predict_time(a, b, d):
    return float(a) * (float(d) ** float(b))

def pace_predictions_view(category, gender, stroke, current_dist, current_time_text, ml_df):
    rec_df = records[(records["category"]==category)&(records["gender_norm"]==gender)&
                     (records["stroke_clean"]==stroke)&(~records["is_relay"])]
    rec_df = rec_df.dropna(subset=["distance_m","time_s"])
    if rec_df["distance_m"].nunique() < 2:
        return pn.pane.Markdown("Insufficient record data to fit a pace curve."), pn.Spacer(), pn.Spacer()

    a_r, b_r = fit_power_curve(rec_df["distance_m"], rec_df["time_s"])

    # Student points
    stu_pts = []
    if isinstance(ml_df, pd.DataFrame) and not ml_df.empty:
        for _, r in ml_df.iterrows():
            try:
                dd = int(r.get("distance_m"))
            except Exception:
                continue
            tt = to_seconds(r.get("time_str"))
            if dd in VALID_EVENTS.get(stroke, []) and tt is not None:
                stu_pts.append((dd, tt))
    if len(stu_pts) == 0:
        tt = to_seconds(current_time_text)
        if tt is not None and current_dist in VALID_EVENTS.get(stroke, []):
            stu_pts.append((int(current_dist), tt))

    if len(stu_pts) >= 2:
        d_list, t_list = zip(*stu_pts)
        a_s, b_s = fit_power_curve(d_list, t_list)
    elif len(stu_pts) == 1:
        d0, t0 = stu_pts[0]
        b_s = b_r
        a_s = t0 / (d0 ** b_s)
    else:
        a_s = b_s = None

    dist_all = VALID_EVENTS.get(stroke, [])
    rows_out = []
    for d in dist_all:
        rec_row = best_record_row(category, gender, stroke, d)
        rec_time = float(rec_row["time_s"]) if rec_row is not None else predict_time(a_r, b_r, d)
        stu_pred = predict_time(a_s, b_s, d) if (a_s is not None and b_s is not None) else None
        gap_s = (stu_pred - rec_time) if (stu_pred is not None) else None
        gap_pct = (gap_s / rec_time * 100) if (gap_s is not None and rec_time>0) else None
        rows_out.append({
            "distance_m": d,
            "record_time": fmt_seconds(rec_time),
            "student_pred": fmt_seconds(stu_pred) if stu_pred is not None else "—",
            "gap_s": f"{gap_s:.2f}" if gap_s is not None else "—",
            "gap_pct": f"{gap_pct:.1f}%" if gap_pct is not None else "—"
        })
    pred_df = pd.DataFrame(rows_out)

    focus_df = pred_df.copy()
    focus_df["gap_pct_val"] = pd.to_numeric(focus_df["gap_pct"].str.replace("%",""), errors="coerce")
    focus_df = focus_df.dropna(subset=["gap_pct_val"]).sort_values("gap_pct_val", ascending=False).head(2)
    focus_tbl = pn.widgets.Tabulator(
        focus_df[["distance_m","student_pred","record_time","gap_s","gap_pct"]],
        height=150, sizing_mode="stretch_width", show_index=False
    )

    # Curves and points
    fig = go.Figure()
    x_plot = dist_all
    y_rec_curve = [predict_time(a_r, b_r, d) for d in x_plot]
    fig.add_trace(go.Scatter(x=x_plot, y=y_rec_curve, mode="lines", name="Record Curve", line=dict(width=3)))
    rec_points = rec_df.groupby("distance_m", as_index=False)["time_s"].min()
    fig.add_trace(go.Scatter(x=rec_points["distance_m"], y=rec_points["time_s"],
                             mode="markers", name="Record (actual)", marker=dict(size=10)))
    if a_s is not None and b_s is not None:
        y_stu_curve = [predict_time(a_s, b_s, d) for d in x_plot]
        fig.add_trace(go.Scatter(x=x_plot, y=y_stu_curve, mode="lines", name="Student Curve", line=dict(width=3)))
    if len(stu_pts) > 0:
        xs, ys = zip(*stu_pts)
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="markers", name="Student (measured)", marker=dict(size=10)))

    fig.update_layout(template="plotly_white",
                      title=f"Pace Curves — {stroke} • {category} • {gender}",
                      xaxis_title="Distance (m)", yaxis_title="Time (s)",
                      margin=dict(l=40,r=40,t=60,b=90),
                      font=PLOTLY_FONT, legend=PLOTLY_LEGEND_BOTTOM)
    fig.update_xaxes(gridcolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb")

    pred_tbl = pn.widgets.Tabulator(pred_df, pagination="remote", page_size=12,
                                    sizing_mode="stretch_width", height=260, show_index=False)

    controls = pn.Row(ml_seed_btn, ml_clear_btn, pn.Spacer(), ml_tip, sizing_mode="stretch_width")
    return pn.Column(controls, pn.Accordion(("Student Times (optional)", times_editor), active=[]),
                     pn.Spacer(height=6), fig,
                     pn.pane.Markdown("### Predictions"), pred_tbl,
                     pn.pane.Markdown("### Focus next (largest gaps)"), focus_tbl)

pace_pane = pn.bind(
    pace_predictions_view,
    category_select, gender_select, stroke_select, event_select, student_time, times_editor.param.value
)

# ---------- Layout ----------
filters_card = pn.Card(
    pn.Column(
        pn.pane.Markdown("### Filters", css_classes=["card-title"]),
        category_select, gender_select, stroke_select, event_select,
        student_name, student_time, hint_md,
        reset_btn
    ), title="Controls", width=360
)
controls_card = pn.Card(
    pn.Column(
        pn.pane.Markdown("### Chart Controls", css_classes=["card-title"]),
        chart_w, chart_h, include_mixed
    ), title="Display", width=360
)

template = pn.template.FastListTemplate(
    title="Swim Standard",
    sidebar=[filters_card, controls_card],
    main=[pn.Tabs(
        ("Compare", compare_pane),
        ("Exercises", exercises_pane),
        ("Records", records_pane),
        ("Analytics", analytics_pane),
        ("Pace Predictions", pace_pane),
    )],
    header_background="#0a2540",
    theme_toggle=False
)

# ---------- Main ----------
if __name__ == "__main__":
    if not RECORDS_CSV.exists():
        pn.state.notifications.error(f"Missing file: {RECORDS_CSV.name} (place it next to this script).")
    if not EXERCISES_CSV.exists():
        pn.state.notifications.warning("Optional: swim_exercises.csv not found — Exercises tab will be empty.")
    pn.serve(template, port=PORT, show=True)
