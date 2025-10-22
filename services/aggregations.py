from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from services.metrics import compute_yearly_metrics


def compute_peer_groups(df: pd.DataFrame, major_code: Optional[str]) -> Dict[str, pd.DataFrame]:
    """Return yearly metrics for the selected major classification and overall averages."""
    peers: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return peers

    if major_code:
        major_df = df[df["産業大分類コード"] == major_code]
        peers["major"] = compute_yearly_metrics(major_df)
    peers["overall"] = compute_yearly_metrics(df)
    return peers


def get_suggestions(df: pd.DataFrame, major_code: Optional[str], current_middle: Optional[str]) -> Dict[str, pd.DataFrame]:
    suggestions: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return suggestions

    if major_code:
        same_major = df[df["産業大分類コード"] == major_code]
        if current_middle:
            name_only = current_middle.split("(")[0].strip()
            same_major = same_major[same_major["業種中分類名"].str.strip() != name_only]
        suggestions["same_major"] = (
            same_major[["業種中分類コード", "業種中分類名"]]
            .drop_duplicates()
            .sort_values("業種中分類コード")
            .head(5)
        )
    suggestions["overall"] = (
        df[["産業大分類コード", "産業大分類名"]]
        .drop_duplicates()
        .sort_values("産業大分類コード")
        .head(5)
    )
    return suggestions


__all__ = ["compute_peer_groups", "get_suggestions"]
