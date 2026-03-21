import os
import sys
import socket
import subprocess
import time

import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_PORT = int(os.environ.get("PORT", 8765))
WS_EXTERNAL_URL = os.environ.get("WS_URL", "")


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
                    if not WS_EXTERNAL_URL:
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

    player_name = st.session_state.get("player_name", "Player")
    game_mode = st.session_state.get("game_mode", "classic")

    if WS_EXTERNAL_URL:
        ws_url = WS_EXTERNAL_URL
    else:
        ws_url = f"ws://{get_local_ip()}:{SERVER_PORT}/ws"

    html = html.replace("__PLAYER_NAME__", player_name.replace('"', '\\"'))
    html = html.replace("__WS_URL__", ws_url)
    html = html.replace("__GAME_MODE__", game_mode)

    st.components.v1.html(html, height=760, scrolling=False)
