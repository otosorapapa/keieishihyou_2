from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FONT_FAMILY = "Hiragino Kaku Gothic ProN, Hiragino Sans, Noto Sans JP, Meiryo, sans-serif"


def _default_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        font=dict(family=FONT_FAMILY),
        margin=dict(l=60, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def create_sparkline(years: Iterable[int], values: Iterable[Optional[float]]) -> go.Figure:
    fig = go.Figure()
    cleaned_years: List[int] = [int(y) for y in years]
    cleaned_values: List[Optional[float]] = [v if v is None or np.isfinite(v) else None for v in values]
    fig.add_trace(
        go.Scatter(
            x=cleaned_years,
            y=cleaned_values,
            mode="lines",
            line=dict(color="#2563eb", width=2),
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        font=dict(family=FONT_FAMILY),
        margin=dict(l=0, r=0, t=0, b=0),
        height=80,
        showlegend=False,
        template="plotly_white",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def sales_profit_trend(
    primary: pd.DataFrame,
    peer_major: Optional[pd.DataFrame] = None,
    peer_overall: Optional[pd.DataFrame] = None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if primary is None or primary.empty:
        return _default_layout(fig, "売上・利益推移")

    years = primary.index.astype(int).tolist()
    fig.add_trace(
        go.Scatter(
            x=years,
            y=primary["sales"],
            name="売上高",
            mode="lines+markers",
            line=dict(color="#2563eb", width=3),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=primary["operating_profit"],
            name="営業利益",
            mode="lines+markers",
            line=dict(color="#10b981", width=3),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=primary["ordinary_profit"],
            name="経常利益",
            mode="lines+markers",
            line=dict(color="#f97316", width=3),
        ),
        secondary_y=True,
    )

    def _add_peer(peer: Optional[pd.DataFrame], label: str) -> None:
        if peer is None or peer.empty:
            return
        peer_years = peer.index.astype(int).tolist()
        fig.add_trace(
            go.Scatter(
                x=peer_years,
                y=peer["sales"],
                name=f"売上高 ({label})",
                mode="lines",
                line=dict(color="#9ca3af", dash="dash"),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=peer_years,
                y=peer["ordinary_profit"],
                name=f"経常利益 ({label})",
                mode="lines",
                line=dict(color="#9ca3af", dash="dot"),
            ),
            secondary_y=True,
        )

    _add_peer(peer_major, "大分類平均")
    _add_peer(peer_overall, "全体平均")

    fig.update_yaxes(title_text="売上高（百万円）", secondary_y=False)
    fig.update_yaxes(title_text="利益（百万円）", secondary_y=True)
    return _default_layout(fig, "売上・利益推移")


def margin_trend(primary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if primary is None or primary.empty:
        return _default_layout(fig, "利益率推移")

    years = primary.index.astype(int).tolist()
    for key, name, color in [
        ("gross_margin", "総利益率", "#2563eb"),
        ("operating_margin", "営業利益率", "#10b981"),
        ("ordinary_margin", "経常利益率", "#f97316"),
    ]:
        if key in primary:
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=primary[key],
                    name=name,
                    mode="lines+markers",
                    line=dict(width=3, color=color),
                )
            )
    fig.update_yaxes(title="利益率", tickformat=".0%")
    return _default_layout(fig, "利益率推移")


def bs_composition(primary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, shared_y=True, horizontal_spacing=0.12)
    if primary is None or primary.empty:
        return _default_layout(fig, "BS 構成比")

    years = primary.index.astype(int).tolist()
    assets = primary["assets"].replace(0, np.nan)
    asset_ratio_current = primary.get("current_assets", pd.Series(index=primary.index)) / assets
    asset_ratio_fixed = primary.get("fixed_assets", pd.Series(index=primary.index)) / assets
    liabilities_ratio_current = primary.get("current_liabilities", pd.Series(index=primary.index)) / assets
    liabilities_ratio_fixed = primary.get("fixed_liabilities", pd.Series(index=primary.index)) / assets
    equity_ratio = primary.get("equity", pd.Series(index=primary.index)) / assets

    fig.add_trace(
        go.Bar(
            y=years,
            x=asset_ratio_current,
            orientation="h",
            name="流動資産",
            marker=dict(color="#60a5fa"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=years,
            x=asset_ratio_fixed,
            orientation="h",
            name="固定資産",
            marker=dict(color="#1d4ed8"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=years,
            x=liabilities_ratio_current,
            orientation="h",
            name="流動負債",
            marker=dict(color="#fca5a5"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            y=years,
            x=liabilities_ratio_fixed,
            orientation="h",
            name="固定負債",
            marker=dict(color="#ef4444"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            y=years,
            x=equity_ratio,
            orientation="h",
            name="純資産",
            marker=dict(color="#10b981"),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title="資産構成比", tickformat=".0%", row=1, col=1)
    fig.update_xaxes(title="負債・純資産構成比", tickformat=".0%", row=1, col=2)
    fig.update_yaxes(title="集計年")
    fig.update_layout(barmode="stack", height=400)
    return _default_layout(fig, "BS 構成比")


def ebitda_coverage(primary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if primary is None or primary.empty:
        return _default_layout(fig, "EBITDA と利払")

    years = primary.index.astype(int).tolist()
    fig.add_trace(
        go.Bar(
            x=years,
            y=primary["ebitda"],
            name="EBITDA",
            marker=dict(color="#2563eb"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=years,
            y=primary["interest"],
            name="支払利息",
            marker=dict(color="#94a3b8"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=primary["interest_coverage"],
            name="Interest Coverage",
            mode="lines+markers",
            line=dict(color="#f97316", width=3),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title="金額（百万円）", secondary_y=False)
    fig.update_yaxes(title="カバレッジ", secondary_y=True, tickformat=".1f")
    return _default_layout(fig, "EBITDA と利払")


def productivity_vs_labor_share(primary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if primary is None or primary.empty:
        return _default_layout(fig, "労働生産性と労働分配率")

    years = primary.index.astype(int).tolist()
    fig.add_trace(
        go.Scatter(
            x=primary["labor_share"],
            y=primary["labor_productivity"],
            mode="markers+lines",
            text=years,
            name="労働指標",
            marker=dict(size=10, color="#2563eb"),
            line=dict(color="#2563eb"),
            hovertemplate="%{text}年<br>労働分配率: %{x:.1%}<br>労働生産性: %{y:,.0f} 百万円<extra></extra>",
        )
    )
    fig.update_xaxes(title="労働分配率", tickformat=".0%")
    fig.update_yaxes(title="労働生産性（百万円）")
    return _default_layout(fig, "労働生産性と労働分配率")


def dupont_lines(primary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if primary is None or primary.empty:
        return _default_layout(fig, "DuPont 分解")

    years = primary.index.astype(int).tolist()
    for key, name, color in [
        ("dupont_net_margin", "当期純利益率", "#2563eb"),
        ("dupont_asset_turnover", "総資産回転率", "#10b981"),
        ("dupont_leverage", "レバレッジ", "#f97316"),
    ]:
        if key in primary:
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=primary[key],
                    name=name,
                    mode="lines+markers",
                    line=dict(width=3, color=color),
                )
            )
    fig.update_yaxes(title="指標", tickformat=".2f")
    return _default_layout(fig, "DuPont 分解")


__all__ = [
    "FONT_FAMILY",
    "create_sparkline",
    "dupont_lines",
    "ebitda_coverage",
    "margin_trend",
    "productivity_vs_labor_share",
    "sales_profit_trend",
    "bs_composition",
]
