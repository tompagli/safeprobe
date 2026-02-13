"""
safeprobe.utils.logging - Structured logging for evaluation events.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False

def setup_logging(log_dir: Optional[str] = None, verbose: bool = False):
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger("safeprobe")
    root.setLevel(level)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(formatter)
    sh.setLevel(level)
    root.addHandler(sh)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / "safeprobe.log", encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)
    _CONFIGURED = True

def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
