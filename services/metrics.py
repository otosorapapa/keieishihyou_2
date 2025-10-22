from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class KPIResult:
    key: str
    label: str
    latest_value: Optional[float]
    delta_value: Optional[float]
    value_type: str  # "currency", "ratio", "number"
    sparkline_years: List[int]
    sparkline_values: List[Optional[float]]
    tooltip: Optional[str] = None


COLUMN_SETS: Dict[str, Sequence[str]] = {
    "sales": ["売上高（百万円）"],
    "operating_profit": ["営業利益（百万円）"],
    "ordinary_profit": ["経常利益（経常損失）（百万円）"],
    "gross_profit": ["売上総利益（百万円）"],
    "net_income": ["税引後当期純利益（百万円）"],
    "value_added": ["付加価値額（百万円）"],
    "personnel_cost": ["人件費（百万円）"],
    "interest": ["支払利息・割引料（百万円）"],
    "assets": ["資産（百万円）"],
    "current_assets": ["流動資産（百万円）"],
    "fixed_assets": ["固定資産（百万円）"],
    "liabilities": ["負債（百万円）"],
    "current_liabilities": ["流動負債（百万円）"],
    "fixed_liabilities": ["固定負債（百万円）"],
    "short_term_loans": ["短期借入金（金融機関）（百万円）", "短期借入金（金融機関以外）（百万円）"],
    "long_term_loans": ["長期借入金（金融機関）（百万円）", "長期借入金（金融機関以外）（百万円）"],
    "bonds": ["社債（百万円）"],
    "equity": ["純資産（百万円）"],
    "depreciation": ["減価償却費（百万円）", "減価償却費（百万円）.1"],
}

FTE_COLUMNS: Sequence[str] = [
    "常用雇用者",
    "合計_正社員・正職員以外（就業時間換算人数）",
    "他社からの出向従業者（出向役員を含む）及び派遣従業者の合計",
]


def _sum_existing(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    available = [col for col in columns if col in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[available].sum(axis=1, min_count=1)


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator
    mask = (denominator == 0) | denominator.isna()
    result = result.mask(mask, np.nan)
    return result


def compute_growth(series: pd.Series) -> pd.Series:
    previous = series.shift(1)
    growth = (series - previous) / previous
    mask = (previous == 0) | previous.isna()
    return growth.mask(mask, np.nan)


def compute_cagr(series: pd.Series, periods: int = 3) -> pd.Series:
    previous = series.shift(periods)
    ratio = safe_div(series, previous)
    cagr = ratio.pow(1 / periods) - 1
    return cagr.mask(previous <= 0, np.nan)


def compute_yearly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "集計年" not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.dropna(subset=["集計年"])
        .groupby("集計年")
        .mean(numeric_only=True)
        .sort_index()
    )
    if grouped.empty:
        return grouped

    metrics = pd.DataFrame(index=grouped.index)
    for key, columns in COLUMN_SETS.items():
        metrics[key] = _sum_existing(grouped, columns)

    metrics["depreciation"] = metrics[["depreciation"]].sum(axis=1, min_count=1)
    metrics["interest_bearing_debt"] = (
        _sum_existing(grouped, COLUMN_SETS["short_term_loans"])
        + _sum_existing(grouped, COLUMN_SETS["long_term_loans"])
        + _sum_existing(grouped, COLUMN_SETS["bonds"])
    )

    metrics["fte"] = _sum_existing(grouped, FTE_COLUMNS)

    metrics["ebitda"] = metrics["operating_profit"] + _sum_existing(
        grouped, COLUMN_SETS["depreciation"]
    )
    metrics["gross_margin"] = safe_div(metrics["gross_profit"], metrics["sales"])
    metrics["operating_margin"] = safe_div(
        metrics["operating_profit"], metrics["sales"]
    )
    metrics["ordinary_margin"] = safe_div(
        metrics["ordinary_profit"], metrics["sales"]
    )
    metrics["equity_ratio"] = safe_div(metrics["equity"], metrics["assets"])
    metrics["asset_turnover"] = safe_div(metrics["sales"], metrics["assets"])
    metrics["labor_productivity"] = safe_div(
        metrics["value_added"], metrics["fte"]
    )
    metrics["labor_share"] = safe_div(
        metrics["personnel_cost"], metrics["value_added"]
    )
    metrics["interest_coverage"] = safe_div(
        metrics["ebitda"], metrics["interest"]
    )
    metrics["interest_bearing_dependence"] = safe_div(
        metrics["interest_bearing_debt"], metrics["assets"]
    )
    metrics["roe"] = safe_div(metrics["net_income"], metrics["equity"])
    metrics["dupont_net_margin"] = safe_div(metrics["net_income"], metrics["sales"])
    metrics["dupont_leverage"] = safe_div(metrics["assets"], metrics["equity"])
    metrics["dupont_asset_turnover"] = metrics["asset_turnover"]
    metrics["dupont_roe"] = (
        metrics["dupont_net_margin"]
        * metrics["dupont_asset_turnover"]
        * metrics["dupont_leverage"]
    )

    metrics["sales_yoy"] = compute_growth(metrics["sales"])
    metrics["operating_yoy"] = compute_growth(metrics["operating_profit"])
    metrics["sales_cagr_3y"] = compute_cagr(metrics["sales"], periods=3)

    return metrics


KPI_DEFINITIONS = [
    {"key": "sales", "label": "売上高", "value_type": "currency"},
    {"key": "operating_profit", "label": "営業利益", "value_type": "currency"},
    {"key": "ebitda", "label": "EBITDA", "value_type": "currency"},
    {"key": "equity_ratio", "label": "自己資本比率", "value_type": "ratio"},
    {"key": "asset_turnover", "label": "総資本回転率", "value_type": "ratio"},
    {"key": "labor_productivity", "label": "労働生産性", "value_type": "currency"},
    {"key": "labor_share", "label": "労働分配率", "value_type": "ratio"},
    {"key": "ordinary_margin", "label": "経常利益率", "value_type": "ratio"},
]

KPI_TOOLTIPS = {
    "labor_productivity": "付加価値額とFTE（常用雇用者等）が必要です。",
    "labor_share": "人件費と付加価値額が必要です。",
    "equity_ratio": "純資産と総資産が必要です。",
    "asset_turnover": "売上高と総資産が必要です。",
    "ordinary_margin": "経常利益と売上高が必要です。",
}


def _latest_non_null(series: pd.Series) -> Optional[float]:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return float(non_null.iloc[-1])


def _previous_non_null(series: pd.Series) -> Optional[float]:
    non_null = series.dropna()
    if len(non_null) < 2:
        return None
    return float(non_null.iloc[-2])


def build_kpi_cards(metrics: pd.DataFrame) -> List[KPIResult]:
    cards: List[KPIResult] = []
    if metrics is None or metrics.empty:
        return cards

    years = metrics.index.astype(int).tolist()
    for definition in KPI_DEFINITIONS:
        key = definition["key"]
        series = metrics.get(key, pd.Series(dtype=float))
        latest = _latest_non_null(series)
        previous = _previous_non_null(series)
        delta = None
        if latest is not None and previous is not None:
            if definition["value_type"] == "currency":
                delta = compute_growth(series).dropna().iloc[-1]
            elif definition["value_type"] == "ratio":
                delta = latest - previous
            else:
                delta = latest - previous
        tooltip = None
        if series.isna().all():
            tooltip = KPI_TOOLTIPS.get(key)
        cards.append(
            KPIResult(
                key=key,
                label=definition["label"],
                latest_value=latest,
                delta_value=delta,
                value_type=definition["value_type"],
                sparkline_years=years,
                sparkline_values=[
                    float(v) if pd.notna(v) else None for v in series.tolist()
                ],
                tooltip=tooltip,
            )
        )
    return cards


def build_summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    table = metrics[
        [
            "sales",
            "operating_profit",
            "ebitda",
            "ordinary_profit",
            "gross_margin",
            "operating_margin",
            "ordinary_margin",
            "equity_ratio",
            "asset_turnover",
            "labor_productivity",
            "labor_share",
            "roe",
        ]
    ].copy()
    table.index.name = "集計年"
    return table


__all__ = [
    "KPIResult",
    "build_kpi_cards",
    "build_summary_table",
    "compute_yearly_metrics",
    "compute_cagr",
    "compute_growth",
    "safe_div",
]
