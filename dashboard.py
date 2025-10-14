import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# --- Configuration ---
ROOT_DIRECTORY = Path(__file__).parent
BASE_PORT = 8601  # Starting port for the apps to avoid conflicts

# --- Helper Functions ---

def get_project_paths(root_dir: Path) -> dict[str, Path]:
    """Dynamically finds project directories containing an 'app.py' file."""
    projects = {}
    for item in root_dir.iterdir():
        # A project is a directory with an 'app.py' inside
        if item.is_dir() and (item / "app.py").exists():
            projects[item.name] = item / "app.py"
    return projects

def stop_process(name: str):
    """Stops a running Streamlit app process by name."""
    proc = st.session_state.processes.get(name)
    if proc:
        proc.terminate()
        # Wait a moment for the process to terminate
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill() # Force kill if it doesn't terminate gracefully
        del st.session_state.processes[name]
        st.success(f"Stopped {name}.")
        time.sleep(0.5) # Give Streamlit time to re-render
        st.rerun()

def start_process(name: str, path: Path, port: int):
    """Starts a new Streamlit app process."""
    if name in st.session_state.processes:
        st.warning(f"{name} is already running.")
        return

    python_executable = sys.executable or "python"
    command = [
        python_executable,
        "-m", "streamlit", "run", str(path),
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.address", "localhost",
    ]

    # Use DEVNULL to hide subprocess output from the dashboard's console
    # For debugging, you could redirect to a log file instead
    # log_file = open(f"{name}_log.txt", "w")
    # proc = subprocess.Popen(command, cwd=str(path.parent), stdout=log_file, stderr=log_file)

    proc = subprocess.Popen(
        command,
        cwd=str(path.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    st.session_state.processes[name] = proc
    st.success(f"Starting {name} on port {port}...")
    time.sleep(1) # Give the process a moment to start up
    st.rerun()

# --- Main App ---

st.set_page_config(page_title="Projects Dashboard", page_icon="🗂️", layout="wide")

# Initialize session state for storing subprocesses
if "processes" not in st.session_state:
    st.session_state.processes = {}

# --- Cloud Mode (Deployed Apps) ---
# Checks for APP_URLS in Streamlit secrets to link to deployed apps.
try:
    # Safely get secrets
    APP_URLS = dict(st.secrets.get("APP_URLS", {}))
except Exception:
    APP_URLS = {}
CLOUD_MODE = bool(APP_URLS)


st.title("🗂️ Projects Dashboard")

if CLOUD_MODE:
    st.info("☁️ Cloud mode: Linking to pre-deployed applications.")
    
    project_names = sorted(APP_URLS.keys())
    
    for name in project_names:
        url = APP_URLS.get(name)
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader(name)
                st.link_button("Open App ↗️", url)
            with col2:
                st.caption(f"URL: `{url}`")

    st.divider()
    st.caption("Configure `[APP_URLS]` in Streamlit secrets to manage links.")

# --- Local Mode (Running Apps Locally) ---
else:
    st.info("🖥️ Local mode: Launching applications on this machine.")
    
    # Dynamically find projects
    projects = get_project_paths(ROOT_DIRECTORY)

    if not projects:
        st.warning(
            "No projects found! "
            f"Create a folder in this directory ('{ROOT_DIRECTORY}') and add an 'app.py' file to it."
        )
    else:
        st.header("Available Projects", divider="rainbow")
        
        sorted_projects = sorted(projects.items())
        
        for idx, (name, path) in enumerate(sorted_projects):
            port = BASE_PORT + idx
            
            with st.container(border=True):
                col1, col2, col3 = st.columns([2, 3, 2])
                
                with col1:
                    st.subheader(name)
                
                proc = st.session_state.processes.get(name)
                is_running = proc and proc.poll() is None
                
                with col2:
                    if is_running:
                        st.success(f"✅ Running on port {port}")
                        url = f"http://localhost:{port}"
                        st.markdown(f"**[Open {name} ↗️]({url})**")
                    else:
                        st.info("⚪ Stopped")
                        
                with col3:
                    if is_running:
                        st.button("Stop App", key=f"stop_{name}", on_click=stop_process, args=(name,), type="primary")
                    else:
                        st.button("Start App", key=f"start_{name}", on_click=start_process, args=(name, path, port))


    st.divider()
    st.caption("Each project is a standalone Streamlit app inside its own folder.")
