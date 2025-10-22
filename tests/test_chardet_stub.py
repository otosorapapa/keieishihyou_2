from __future__ import annotations

import importlib


def test_detect_prefers_utf8_with_bom():
    chardet = importlib.import_module("chardet")
    raw = "テスト".encode("utf-8-sig")
    result = chardet.detect(raw)
    assert result["encoding"] == "utf-8-sig"
    assert result["confidence"] >= 0.6


def test_detect_falls_back_to_cp932():
    chardet = importlib.import_module("chardet")
    raw = "株式会社".encode("cp932")
    result = chardet.detect(raw)
    assert result["encoding"] == "cp932"


def test_universal_detector_collects_chunks():
    from chardet.universaldetector import UniversalDetector

    detector = UniversalDetector()
    chunks = ["経営".encode("utf-8")[:3], "経営".encode("utf-8")[3:]]
    for chunk in chunks:
        detector.feed(chunk)
    result = detector.close()
    assert result["encoding"] == "utf-8"
    assert detector.done

    detector.reset()
    assert not detector.done
    assert detector.result is None
