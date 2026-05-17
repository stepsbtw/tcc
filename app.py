import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from streamlit_shortcuts import shortcut_button

TRIALS_ROOT = Path("IPqM-Fall/trials_90hz")
META_FILE = "window_labels.csv"
FS = 90

st.set_page_config(layout="wide")
meta = pd.read_csv(META_FILE)
st.sidebar.title("Trial Viewer")
trial_files = sorted(meta["file"].unique())
selected_trial = st.sidebar.selectbox("Select trial",trial_files)

trial_meta = meta[meta["file"] == selected_trial]
trial_path = TRIALS_ROOT / selected_trial
trial_df = pd.read_parquet(trial_path)

channels = [c for c in trial_df.columns if c != "timestamp"]
window_ids = trial_meta["window_id"].tolist()

selected_window = st.sidebar.selectbox("Select window", window_ids)

selected_row = trial_meta[trial_meta["window_id"] == selected_window].iloc[0]

start_idx = int(selected_row["start_idx"])
end_idx = int(selected_row["end_idx"])

window_df = trial_df.iloc[start_idx:end_idx]

reviewed_count = int(meta["reviewed"].sum())
total_count = len(meta)

progress = reviewed_count / total_count

st.sidebar.markdown("---")
st.sidebar.write(f"Reviewed: {reviewed_count}/{total_count}")
st.sidebar.progress(progress)

labels = [
    "ADL_1",   # standing
    "ADL_2",   # walking
    "ADL_3",   # running
    "ADL_4",   # jumping
    "ADL_5",   # stair climbing
    "ADL_6",   # stair descending
    "ADL_7",   # sitting in chair
    "ADL_8",   # standing from chair
    "ADL_11",  # uphill walking
    "ADL_12",  # downhill walking
    "ADL_13",  # uphill running
    "ADL_14",  # downhill running
    "ADL_15",  # stair hopping

    "OM_1",    # sweep walking
    "OM_2",    # sweep quick engagement
    "OM_3",    # kneeling shooting position (from standing)
    "OM_4",    # kneeling shooting position (from walking)
    "OM_5",    # kneeling shooting position (from running)
    "OM_6",    # prone shooting position (from standing)
    "OM_7",    # prone shooting position (from walking)
    "OM_8",    # prone shooting position (from running)
    "OM_9",    # crawling

    "FALL_1",  # frontal fall (supine)
    "FALL_2",  # frontal fall (prone)
    "FALL_3",  # backward fall
    "FALL_5",  # lateral fall (right)
    "FALL_6",  # lateral fall (left)

    "UNKNOWN"
]

label_shortcuts = {
    "1": "ADL_1",   # standing
    "2": "ADL_2",   # walking
    "3": "ADL_3",   # running
    "4": "ADL_4",   # jumping
    "5": "ADL_5",   # stair climbing
    "6": "ADL_6",   # stair descending
    "7": "ADL_7",   # sitting in chair
    "8": "ADL_8",   # standing from chair

    "q": "ADL_11",  # uphill walking
    "w": "ADL_12",  # downhill walking
    "e": "ADL_13",  # uphill running
    "r": "ADL_14",  # downhill running
    "t": "ADL_15",  # stair hopping

    "a": "OM_1",    # sweep walking
    "s": "OM_2",    # sweep quick engagement
    "d": "OM_3",    # kneeling from standing
    "f": "OM_4",    # kneeling from walking
    "g": "OM_5",    # kneeling from running

    "h": "OM_6",    # prone from standing
    "j": "OM_7",    # prone from walking
    "k": "OM_8",    # prone from running
    "l": "OM_9",    # crawling

    "z": "FALL_1",  # frontal supine
    "x": "FALL_2",  # frontal prone
    "c": "FALL_3",  # backward
    "v": "FALL_5",  # lateral right
    "b": "FALL_6",  # lateral left

    "0": "UNKNOWN"
}

current_label = selected_row["label"]

if current_label not in labels:
    current_label = labels[0]

if "selected_label" not in st.session_state:
    st.session_state.selected_label = current_label

left_col, right_col = st.columns([3, 2])

with left_col:
    fig = go.Figure()
    for c in channels:
        fig.add_trace(
            go.Scatter(
                x=trial_df.index / FS,
                y=trial_df[c],
                mode="lines",
                name=c
            )
        )
    for _, row in trial_meta.iterrows():

        color = (
            "rgba(255,0,0,0.15)"
            if row["reviewed"]
            else "rgba(0,0,255,0.05)"
        )

        fig.add_vrect(
            x0=int(row["start_idx"]) / FS,
            x1=int(row["end_idx"]) / FS,
            fillcolor=color,
            line_width=0
        )
    
    fig.add_vrect(
        x0=start_idx / FS,
        x1=end_idx / FS,
        fillcolor="rgba(0,255,0,0.25)",
        line_width=2,
        line_color="green"
    )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),
        xaxis_title="Seconds"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right_col:
    fig2 = go.Figure()
    for c in channels:
        fig2.add_trace(
            go.Scatter(
                x=(window_df.index - window_df.index[0]) / FS,
                y=window_df[c],
                mode="lines",
                name=c
            )
        )

    fig2.update_layout(
        height=350,
        hovermode="x unified",
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),
        xaxis_title="Seconds"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.write(f"Current label: **{current_label}**")
    st.write(f"Reviewed: **{selected_row['reviewed']}**")

    shortcut_cols = st.columns(5)
    shortcut_triggered = False

    for idx, (key, label) in enumerate(label_shortcuts.items()):
        with shortcut_cols[idx % 5]:
            if shortcut_button(
                label=f"{key}",
                shortcut=key
            ):

                st.session_state.selected_label = label
                shortcut_triggered = True

    current_selection = st.session_state.selected_label

    if current_selection not in labels:
        current_selection = current_label

    new_label = st.selectbox(
        "Label",
        labels,
        index=labels.index(current_selection)
    )

    st.session_state.selected_label = new_label

    if st.button("Save Label", use_container_width=True):

        meta.loc[
            meta["window_id"] == selected_window,
            "label"
        ] = new_label

        meta.loc[
            meta["window_id"] == selected_window,
            "reviewed"
        ] = True

        meta.to_csv(META_FILE, index=False)

        st.success("Saved")

    st.caption(
        " | ".join(
            [f"{k}:{v}" for k, v in label_shortcuts.items()]
        )
    )