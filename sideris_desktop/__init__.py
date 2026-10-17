"""Sideris Desktop - a private AI assistant that runs on your own computer.

The package is standard-library first: the graphical shell, the local
HTTP service, the model downloader, the setup state machine and the
training supervisor all run with no third-party dependencies.  Model
inference uses a bundled / pip-installed llama.cpp runtime, and LoRA
training uses an optional torch + transformers + peft runtime that is
installed on request into a private virtual environment.
"""

from __future__ import annotations

__version__ = "5.0.0"
__app_name__ = "Sideris"
__release_name__ = "Sideris Desktop"

# The name the product shipped under before 5.0.0.  Kept so older data
# directories and the LUNO_HOME environment variable keep working.
__legacy_name__ = "Luno"

__all__ = [
    "__version__",
    "__app_name__",
    "__release_name__",
    "__legacy_name__",
]
