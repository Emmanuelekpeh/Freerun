import json
import os
import sys
import socket
import subprocess
import time

import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_PORT = 8765
METRICS_PATH = os.path.join(APP_DIR, "training_metrics.json")


def is_server_running() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("localhost", SERVER_PORT)) == 0


def start_server():
    if is_server_running():
        return
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [sys.executable, os.path.join(APP_DIR, "server.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )
    for _ in range(20):
        time.sleep(0.25)
        if is_server_running():
            return


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def _start_training():
    if st.session_state.get("_train_pid"):
        return
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        [sys.executable, os.path.join(APP_DIR, "training.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )
    st.session_state["_train_pid"] = proc.pid


def _stop_training():
    pid = st.session_state.get("_train_pid")
    if not pid:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True, timeout=5)
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
    st.session_state["_train_pid"] = None


def _load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


# ─── Streamlit App ───────────────────────────────────────────────────

st.set_page_config(page_title="Freerun", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .stApp { background-color: #0a0e17; }
    .block-container { padding-top: 1rem !important; padding-bottom: 0 !important; max-width: 100% !important; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "page" not in st.session_state:
    st.session_state.page = "lobby"

# ─── Navigation ──────────────────────────────────────────────────────
if st.session_state.page != "game":
    nav_cols = st.columns([1, 1, 4])
    with nav_cols[0]:
        if st.button("Play", use_container_width=True,
                      type="primary" if st.session_state.page == "lobby" else "secondary"):
            st.session_state.page = "lobby"
            st.rerun()
    with nav_cols[1]:
        if st.button("Train", use_container_width=True,
                      type="primary" if st.session_state.page == "train" else "secondary"):
            st.session_state.page = "train"
            st.rerun()

# ─── Lobby ───────────────────────────────────────────────────────────
if st.session_state.page == "lobby":
    st.markdown(
        "<h1 style='text-align:center; color:#fff; font-family:Segoe UI,sans-serif; "
        "margin-top:10vh; letter-spacing:6px;'>FREERUN</h1>"
        "<p style='text-align:center; color:#556; font-size:14px;'>infinite tag</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("lobby_form", clear_on_submit=False):
            name = st.text_input(
                "Your name",
                max_chars=16,
                placeholder="Enter a name...",
                label_visibility="collapsed",
            )
            mode = st.radio(
                "Game Mode",
                ["classic", "infestation"],
                horizontal=True,
                captions=["Tag swaps — 1 IT at a time", "IT spreads — last one standing wins"],
            )
            submitted = st.form_submit_button(
                "PLAY", use_container_width=True, type="primary"
            )
            if submitted:
                if name.strip():
                    st.session_state.player_name = name.strip()
                    st.session_state.game_mode = mode
                    st.session_state.page = "game"
                    start_server()
                    st.rerun()
                else:
                    st.warning("Enter a name first.")

    st.markdown(
        "<p style='text-align:center; color:#334; font-size:12px; margin-top:4rem;'>"
        "Desktop: WASD + SPACE dash + E break &nbsp;·&nbsp; Mobile: Joystick + buttons</p>",
        unsafe_allow_html=True,
    )

# ─── Game ────────────────────────────────────────────────────────────
elif st.session_state.page == "game":
    if st.button("← Leave", key="leave"):
        st.session_state.page = "lobby"
        st.rerun()

    client_path = os.path.join(APP_DIR, "client.html")
    with open(client_path, "r", encoding="utf-8") as f:
        html = f.read()

    ws_host = get_local_ip()
    player_name = st.session_state.get("player_name", "Player")
    game_mode = st.session_state.get("game_mode", "classic")

    html = html.replace("__PLAYER_NAME__", player_name.replace('"', '\\"'))
    html = html.replace("__WS_HOST__", ws_host)
    html = html.replace("__GAME_MODE__", game_mode)

    st.components.v1.html(html, height=760, scrolling=False)

# ─── Training Dashboard ─────────────────────────────────────────────
elif st.session_state.page == "train":
    st.markdown("<h2 style='color:#fff;'>Bot Training Dashboard</h2>", unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns([1, 1, 2])
    with tc1:
        if st.button("Start Training", use_container_width=True, type="primary"):
            _start_training()
            st.rerun()
    with tc2:
        if st.button("Stop Training", use_container_width=True):
            _stop_training()
            st.rerun()
    with tc3:
        pid = st.session_state.get("_train_pid")
        if pid:
            st.success(f"Training running (PID {pid})")
        else:
            st.info("Training not running")

    metrics = _load_metrics()

    if metrics is None:
        st.markdown(
            "<p style='color:#667; margin-top:2rem;'>No training data yet. "
            "Click <b>Start Training</b> to begin.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Episode", metrics.get("episode", 0))
        m2.metric("Elapsed", f"{metrics.get('elapsed_s', 0):.0f}s")
        m3.metric("Avg Reward", f"{metrics.get('avg_reward_recent', 0):+.3f}")
        pop = metrics.get("population", [])
        if pop:
            m4.metric("Top ELO", f"{pop[0]['elo']:.0f} ({pop[0]['tier']})")

        rh = metrics.get("reward_history", [])
        if rh:
            st.markdown("### Reward Over Episodes")
            st.line_chart(rh)

        elo_snaps = metrics.get("elo_snapshots", [])
        if elo_snaps:
            avg_elos = [s["avg_elo"] for s in elo_snaps]
            top_elos = [s["top_elo"] for s in elo_snaps]
            st.markdown("### ELO Progression")
            import pandas as pd
            elo_df = pd.DataFrame({"Avg ELO": avg_elos, "Top ELO": top_elos})
            st.line_chart(elo_df)

            latest_tiers = elo_snaps[-1].get("tier_counts", {})
            if latest_tiers:
                st.markdown("### Tier Distribution")
                tier_cols = st.columns(len(latest_tiers))
                colors = {"Bronze": "#cd7f32", "Silver": "#c0c0c0",
                           "Gold": "#ffd700", "Diamond": "#b9f2ff"}
                for i, (tier, count) in enumerate(latest_tiers.items()):
                    with tier_cols[i]:
                        st.markdown(
                            f"<div style='text-align:center;'>"
                            f"<span style='color:{colors.get(tier, '#fff')}; "
                            f"font-size:24px; font-weight:bold;'>{count}</span><br>"
                            f"<span style='color:#888; font-size:12px;'>{tier}</span></div>",
                            unsafe_allow_html=True,
                        )

        if pop:
            st.markdown("### Population Leaderboard")
            for i, agent in enumerate(pop):
                tier_color = {"Bronze": "#cd7f32", "Silver": "#c0c0c0",
                              "Gold": "#ffd700", "Diamond": "#b9f2ff"}.get(agent["tier"], "#fff")
                st.markdown(
                    f"<div style='display:flex; align-items:center; gap:12px; padding:4px 0;'>"
                    f"<span style='color:#556; width:24px;'>#{i+1}</span>"
                    f"<span style='color:{tier_color}; font-weight:bold;'>{agent['tier']}</span>"
                    f"<span style='color:#aaa;'>ELO {agent['elo']:.0f}</span>"
                    f"<span style='color:#556;'>· {agent['games']} games · {agent['tags']} tags</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        if st.button("Refresh", use_container_width=True):
            st.rerun()
