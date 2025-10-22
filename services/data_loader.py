from __future__ import annotations

import codecs
import io
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - fallback for non-Streamlit environments
    class _StreamlitStub:
        """Minimal stub providing ``cache_data`` decorator used in tests.

        The real Streamlit package is only required when running the web
        application. Allowing the import to succeed without Streamlit makes it
        possible to exercise the data loading helpers in isolation (e.g. in
        unit tests or automated scripts).
        """

        def cache_data(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    st = _StreamlitStub()

from services.duckdb_store import DuckDBStore

DATA_PATH = Path("/mnt/data/産業構造マップ_中小企業経営分析_推移　bs pl　従業員数.csv")
FALLBACK_DATA_PATH = Path(__file__).resolve().parents[1] / "assets" / "sample_data.csv"
DEFAULT_ENCODING = "cp932"
FALLBACK_ENCODING = "utf-8-sig"
KEY_COLUMNS = [
    "集計年",
    "産業大分類コード",
    "業種中分類コード",
    "集計形式",
]
CATEGORY_COLUMNS = [
    "産業大分類名",
    "業種中分類名",
]


class DataLoadError(RuntimeError):
    """Raised when loading the CSV fails."""


def _detect_encoding(raw: bytes) -> str:
    """Detect encoding using lightweight heuristics.

    The CSV files we handle are primarily encoded in CP932 or UTF-8 with BOM.
    To avoid depending on optional third-party detectors, we perform a couple of
    quick checks: first, look for a UTF-8 BOM; otherwise try to decode with the
    encodings we care about and return the first one that succeeds.
    """

    if raw.startswith(codecs.BOM_UTF8):
        return FALLBACK_ENCODING

    for candidate in (DEFAULT_ENCODING, "utf-8", FALLBACK_ENCODING):
        try:
            raw.decode(candidate)
            return candidate
        except UnicodeDecodeError:
            continue

    return DEFAULT_ENCODING


def _candidate_paths(path: Path) -> Iterable[Path]:
    if path.exists():
        yield path
        return

    if path == DATA_PATH and FALLBACK_DATA_PATH.exists():
        yield FALLBACK_DATA_PATH


def _read_csv_from_path(path: Path) -> pd.DataFrame:
    candidates = list(_candidate_paths(path))
    if not candidates:
        message = f"CSV ファイルが見つかりません: {path}"
        if path == DATA_PATH:
            message += f" または {FALLBACK_DATA_PATH}"
        raise DataLoadError(message)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        raw = candidate.read_bytes()
        encoding = _detect_encoding(raw)
        buffer = io.BytesIO(raw)

        for encoding_candidate in [encoding, DEFAULT_ENCODING, FALLBACK_ENCODING]:
            try:
                buffer.seek(0)
                df = pd.read_csv(buffer, encoding=encoding_candidate)
                return df
            except UnicodeDecodeError as exc:  # pragma: no cover - extremely rare
                last_error = exc
                continue

    raise DataLoadError("CSV の読み込みに失敗しました。文字コードをご確認ください。") from last_error


def load_uploaded_csv(file) -> pd.DataFrame:
    """Load CSV from an uploaded file-like object using encoding detection."""
    raw = file.read()
    if not raw:
        raise DataLoadError("アップロードされたファイルが空です。")
    buffer = io.BytesIO(raw)
    encoding = _detect_encoding(raw)
    for candidate in [encoding, DEFAULT_ENCODING, FALLBACK_ENCODING]:
        try:
            buffer.seek(0)
            df = pd.read_csv(buffer, encoding=candidate)
            df = _normalize_keys(df)
            df = _coerce_numeric_columns(df)
            df = _drop_duplicate_keys(df)
            return df
        except UnicodeDecodeError:
            continue
    raise DataLoadError("アップロード CSV の読み込みに失敗しました。")


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns: Iterable[str] = [
        col
        for col in df.columns
        if col not in KEY_COLUMNS + CATEGORY_COLUMNS and df[col].dtype != "O"
    ]

    # Explicitly coerce everything except keys and category labels
    for col in df.columns:
        if col in KEY_COLUMNS or col in CATEGORY_COLUMNS:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "集計年" in df.columns:
        df["集計年"] = pd.to_numeric(df["集計年"], errors="coerce").astype("Int64")
    return df


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["産業大分類コード", "業種中分類コード", "集計形式"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def _drop_duplicate_keys(df: pd.DataFrame) -> pd.DataFrame:
    if all(col in df.columns for col in KEY_COLUMNS):
        df = df.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    return df


@st.cache_data(show_spinner=False)
def load_initial_csv(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the baseline CSV once and cache it."""
    df = _read_csv_from_path(path)
    df = _normalize_keys(df)
    df = _coerce_numeric_columns(df)
    df = _drop_duplicate_keys(df)
    return df


def merge_with_store(base_df: pd.DataFrame, store: DuckDBStore) -> pd.DataFrame:
    """Merge the baseline CSV with DuckDB persisted records."""
    store_df = store.read_all()
    if store_df is None or store_df.empty:
        return base_df

    combined = pd.concat([base_df, store_df], ignore_index=True)
    combined = _normalize_keys(combined)
    combined = _coerce_numeric_columns(combined)
    combined = _drop_duplicate_keys(combined)
    return combined


def load_dataset(store: Optional[DuckDBStore] = None) -> pd.DataFrame:
    base_df = load_initial_csv()
    if store is None:
        return base_df
    try:
        return merge_with_store(base_df, store)
    except Exception:
        # When DuckDB is unavailable, gracefully fall back to CSV only
        return base_df


def filter_by_selection(
    df: pd.DataFrame,
    major_code: str,
    middle_name_with_code: str,
) -> pd.DataFrame:
    """Return data filtered by major code and middle classification (name with code)."""
    target_df = df.copy()
    if major_code:
        target_df = target_df[target_df["産業大分類コード"] == major_code]
    if middle_name_with_code:
        # Expect format like "設備工事業 (123)" or "設備工事業"
        name = middle_name_with_code.split("(")[0].strip()
        target_df = target_df[target_df["業種中分類名"].str.strip() == name]
    return target_df


def available_middle_options(df: pd.DataFrame, major_code: str) -> pd.DataFrame:
    filtered = df[df["産業大分類コード"] == major_code]
    return (
        filtered[["業種中分類コード", "業種中分類名"]]
        .drop_duplicates()
        .sort_values("業種中分類コード")
    )


def build_middle_display(row: pd.Series) -> str:
    code = str(row.get("業種中分類コード", "")).strip()
    name = str(row.get("業種中分類名", "")).strip()
    if code:
        return f"{name} ({code})"
    return name


def ensure_year_range(df: pd.DataFrame, year_from: Optional[int], year_to: Optional[int]) -> pd.DataFrame:
    if "集計年" not in df.columns:
        return df
    target = df.copy()
    if year_from is not None:
        target = target[target["集計年"] >= year_from]
    if year_to is not None:
        target = target[target["集計年"] <= year_to]
    return target


__all__ = [
    "DATA_PATH",
    "DuckDBStore",
    "DataLoadError",
    "_detect_encoding",
    "available_middle_options",
    "build_middle_display",
    "ensure_year_range",
    "filter_by_selection",
    "load_uploaded_csv",
    "load_dataset",
    "load_initial_csv",
]
