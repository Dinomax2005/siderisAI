"""Frozen-app entry point for Sideris Desktop.

PyInstaller executes its entry script as ``__main__``.  If we pointed it
straight at ``sideris_desktop/__main__.py`` the module would lose its
package context and every relative import in it would fail with
"attempted relative import with no known parent package".  Importing the
package here keeps ``sideris_desktop`` a real package, so the frozen app
and ``python -m sideris_desktop`` take the exact same code path.
"""

from __future__ import annotations

from sideris_desktop.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
