#!/bin/bash
# ==============================================================================
# Launcher for the YouTube Shorts automation pipeline.
# Double-click the companion "Start YouTube Automation.desktop" file to run
# this, or run it directly from a terminal: ./run_youtube_automation.sh
# ==============================================================================

# Always run from the directory this script lives in, regardless of where
# it was launched from (double-clicking can start it with a different cwd).
cd "$(dirname "$0")" || {
    echo "ERROR: could not cd into script directory."
    read -p "Press Enter to close..."
    exit 1
}

echo "================================================================"
echo " YouTube Shorts Automation"
echo " Working directory: $(pwd)"
echo "================================================================"
echo ""

# --- Locate and activate the virtual environment ---------------------------
# Checks the common venv folder names/locations in order. If your venv lives
# somewhere else, just edit VENV_ACTIVATE below to point at it directly.
VENV_ACTIVATE=""
for candidate in "venv/bin/activate" ".venv/bin/activate" "env/bin/activate" "../venv/bin/activate"; do
    if [ -f "$candidate" ]; then
        VENV_ACTIVATE="$candidate"
        break
    fi
done

if [ -n "$VENV_ACTIVATE" ]; then
    echo "Activating virtual environment: $VENV_ACTIVATE"
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
else
    echo "WARNING: no venv found (checked venv/, .venv/, env/, ../venv/)."
    echo "Edit run_youtube_automation.sh and set VENV_ACTIVATE to your venv's"
    echo "bin/activate path if the script fails to import a package below."
fi

echo ""

# --- Locate the pipeline script ---------------------------------------------
# Checks known filenames in order (newest/preferred first) so this launcher
# keeps working even after renaming/versioning the pipeline script.
PIPELINE_SCRIPT=""
for candidate in "yt_shorts_automation_V2.py" "youtube_shorts_automation.py"; do
    if [ -f "$candidate" ]; then
        PIPELINE_SCRIPT="$candidate"
        break
    fi
done

if [ -z "$PIPELINE_SCRIPT" ]; then
    echo "ERROR: no pipeline script found in $(pwd)"
    echo "Looked for: yt_shorts_automation_V2.py, youtube_shorts_automation.py"
    echo "Make sure this launcher sits in the same folder as the pipeline script,"
    echo "or edit PIPELINE_SCRIPT candidates in run_youtube_automation.sh."
    read -p "Press Enter to close..."
    exit 1
fi

echo "Running: $PIPELINE_SCRIPT"
python3 "$PIPELINE_SCRIPT"
STATUS=$?

echo ""
echo "================================================================"
if [ $STATUS -eq 0 ]; then
    echo " Finished (exit code 0)."
else
    echo " Exited with an error (exit code $STATUS). Scroll up for details."
fi
echo "================================================================"
read -p "Press Enter to close this window..."
