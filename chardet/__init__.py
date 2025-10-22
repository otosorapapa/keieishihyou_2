"""A tiny subset of the :mod:`chardet` API used by the app.

The production application previously depended on the third-party
``chardet`` package for encoding detection.  Streamlit Cloud's minimal
runtime does not provide the dependency which resulted in
``ModuleNotFoundError`` before the UI even booted.  To keep the
application self-contained we provide a lightweight drop-in module that
implements the very small portion of the public API we rely on.

The goal of this shim is not to perform sophisticated statistical
analysis – we only need enough logic to reliably differentiate between
UTF-8 (optionally with BOM) and CP932 encoded CSV files.  The helper
functions are intentionally simple and deterministic so they remain easy
to reason about and cover with unit tests.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

__all__ = ["detect", "UniversalDetector"]

_DEFAULT_ENCODING = "cp932"
_PRIMARY_CANDIDATES: Iterable[str] = ("utf-8", "utf-8-sig", _DEFAULT_ENCODING)


def _pick_encoding(raw: bytes) -> str:
    """Return the most plausible encoding for *raw* bytes.

    The heuristics are intentionally minimal: prefer UTF-8 when the data is
    valid, fall back to CP932 otherwise.  If a UTF-8 BOM is present we treat it
    as authoritative.
    """

    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"

    for candidate in _PRIMARY_CANDIDATES:
        try:
            raw.decode(candidate)
            return candidate
        except UnicodeDecodeError:
            continue
    return _DEFAULT_ENCODING


def detect(raw: bytes) -> Dict[str, Optional[str]]:
    """Mimic :func:`chardet.detect` for the encodings we care about."""

    encoding = _pick_encoding(raw)
    confidence = 0.8 if encoding != _DEFAULT_ENCODING else 0.6
    return {"encoding": encoding, "confidence": confidence, "language": None}


@dataclass
class UniversalDetector:
    """Very small stand-in for :class:`chardet.UniversalDetector`.

    The detector simply buffers data fed to it and performs the same heuristic
    detection as :func:`detect` when :meth:`close` is invoked.  This mirrors the
    subset of behaviour that our data loading utilities rely on while keeping
    the implementation compact.
    """

    _buffer: bytearray = field(default_factory=bytearray)
    done: bool = False
    result: Dict[str, Optional[str]] = None  # type: ignore[assignment]

    def feed(self, data: bytes) -> None:
        if self.done:
            return
        self._buffer.extend(data)

    def close(self) -> Dict[str, Optional[str]]:
        if not self.done:
            self.result = detect(bytes(self._buffer))
            self.done = True
        return self.result

    def reset(self) -> None:
        self._buffer.clear()
        self.done = False
        self.result = None
