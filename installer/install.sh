#!/usr/bin/env bash
# Sideris Desktop 5.0.0 - per-user installer for Linux and macOS.
#
#   ./installer/install.sh                install with the local inference runtime
#   ./installer/install.sh --no-inference skip llama-cpp-python (install later in Setup)
#   ./installer/install.sh --uninstall    remove the app (your chats and models stay)
#
# No sudo, no system-wide changes. Everything lives under
# ~/.local/share/sideris and ~/.local/bin.
set -euo pipefail

APP_NAME="Sideris"
APP_VERSION="5.0.0"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/sideris"
VENV="$DATA_HOME/venv"
SHARE="$DATA_HOME/share"
BIN_DIR="${HOME}/.local/bin"
LAUNCHER="$BIN_DIR/sideris"
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$DESKTOP_DIR/sideris.desktop"
PYTHON="${PYTHON:-python3}"
WITH_INFERENCE=1

say() { printf '%s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

for arg in "$@"; do
  case "$arg" in
    --no-inference) WITH_INFERENCE=0 ;;
    --uninstall)
      say "Removing ${APP_NAME} application files (chats, models and settings are kept)."
      rm -f "$LAUNCHER" "$DESKTOP_FILE"
      rm -rf "$VENV" "$SHARE"
      say "Done. Your data directory still contains your models and conversations:"
      say "  $DATA_HOME"
      exit 0
      ;;
    --help|-h)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) fail "unknown option: $arg" ;;
  esac
done

command -v "$PYTHON" >/dev/null 2>&1 || fail "Python 3 was not found. Install Python 3.10+ and retry."
"$PYTHON" - <<'PY' || exit 1
import sys
if sys.version_info < (3, 10):
    print("error: Python 3.10 or newer is required; found %d.%d" % sys.version_info[:2], file=sys.stderr)
    raise SystemExit(1)
print("Using Python %d.%d" % sys.version_info[:2])
PY

mkdir -p "$DATA_HOME" "$SHARE" "$BIN_DIR" "$DESKTOP_DIR"

if [ ! -x "$VENV/bin/python" ]; then
  say "Creating a private environment in $VENV"
  "$PYTHON" -m venv "$VENV"
fi

say "Installing ${APP_NAME} ${APP_VERSION}"
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV/bin/python" -m pip install "$ROOT"

if [ "$WITH_INFERENCE" = "1" ]; then
  say "Installing the llama.cpp inference runtime (this can take a minute)"
  if ! "$VENV/bin/python" -m pip install "llama-cpp-python==0.3.36"; then
    say "warning: llama-cpp-python could not be installed automatically."
    say "         You can retry later from Setup, or install it with:"
    say "         $VENV/bin/python -m pip install llama-cpp-python"
  fi
fi

# Assets stay outside the Python package in the repository; copy them to
# a stable share directory and point the launcher at them.
rm -rf "$SHARE/assets"
cp -R "$ROOT/assets" "$SHARE/assets"
cp "$ROOT/README.md" "$SHARE/README.md" 2>/dev/null || true
cp "$ROOT/LICENSE" "$SHARE/LICENSE" 2>/dev/null || true

cat > "$LAUNCHER" <<EOF
#!/bin/sh
export SIDERIS_ASSETS="$SHARE/assets"
exec "$VENV/bin/python" -m sideris_desktop "\$@"
EOF
chmod +x "$LAUNCHER"

if [ "$(uname -s)" = "Darwin" ]; then
  cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=${APP_NAME}
Comment=Private local AI assistant
Exec=${LAUNCHER}
Terminal=false
Icon=${SHARE}/assets/logo-256.png
Categories=Utility;Office;
EOF
else
  cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=${APP_NAME}
Comment=Private local AI assistant
Exec=${LAUNCHER}
Terminal=false
Icon=${SHARE}/assets/logo-256.png
Categories=Utility;Office;
StartupWMClass=Sideris
EOF
fi
chmod +x "$DESKTOP_FILE" 2>/dev/null || true

say ""
say "Installed. The 'sideris' command is in $BIN_DIR"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) say "Add it to your PATH:  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
say ""
say "Next: download a model and choose permissions (nothing downloads without --yes):"
say "  sideris setup --model qwen2.5-1.5b --yes"
say "Then open the app:"
say "  sideris"
say ""
say "Training and the background service stay off until you grant them permission,"
say "either in the app or with:  sideris setup --training --background --yes"
