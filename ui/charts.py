from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import altair as alt
except ModuleNotFoundError:  # pragma: no cover - executed when package missing
    alt = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from plotly.graph_objects import Figure as PlotlyFigure
else:
PlotlyFigure = Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from altair import Chart as AltairChart
else:
    AltairChart = Any

FONT_FAMILY = "Hiragino Kaku Gothic ProN, Hiragino Sans, Noto Sans JP, Meiryo, sans-serif"
PLOTLY_IMPORT_ERROR_MESSAGE = "Plotly がインストールされていないためグラフを表示できません。requirements.txt を参照して Plotly を追加してください。"


def _plotly_available() -> bool:
    return go is not None and make_subplots is not None


def _altair_available() -> bool:
    return alt is not None


def _apply_altair_theme(chart: AltairChart, title: str, *, height: Optional[int] = None) -> AltairChart:
    if alt is None:
        return chart
    if title:
        chart = chart.properties(title=title)
    if height is not None:
        chart = chart.properties(height=height)
    return (
        chart.configure_axis(labelFont=FONT_FAMILY, titleFont=FONT_FAMILY)
        .configure_title(font=FONT_FAMILY, fontSize=16)
        .configure_legend(labelFont=FONT_FAMILY, titleFont=FONT_FAMILY)
        .configure_view(strokeOpacity=0)
    )


def _altair_placeholder(title: str) -> Optional[AltairChart]:
    if not _altair_available():
        return None
    placeholder_df = pd.DataFrame({"x": [0], "y": [0]})
    chart = (
        alt.Chart(placeholder_df)
        .mark_text(text="データがありません", font=FONT_FAMILY, fontSize=14)
        .encode(x="x", y="y")
    )
    return _apply_altair_theme(chart, title, height=200)


def _default_layout(fig: PlotlyFigure, title: str) -> PlotlyFigure:
    fig.update_layout(
        title=title,
        font=dict(family=FONT_FAMILY),
        margin=dict(l=60, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def create_sparkline(
    years: Iterable[int], values: Iterable[Optional[float]]
) -> Optional[Any]:
    cleaned_years: List[int] = [int(y) for y in years]
    cleaned_values: List[Optional[float]] = [v if v is None or np.isfinite(v) else None for v in values]

    if _plotly_available():
        fig = go.Figure()
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

    if _altair_available():
        data = pd.DataFrame({"year": cleaned_years, "value": cleaned_values})
        chart = (
            alt.Chart(data)
            .mark_line(color="#2563eb")
            .encode(
                x=alt.X("year:O", axis=None),
                y=alt.Y("value:Q", axis=None),
                tooltip=[alt.Tooltip("year:O", title="年"), alt.Tooltip("value:Q", title="値", format=",")],
            )
            .properties(height=80)
        )
        return _apply_altair_theme(chart, "")

    return None


def sales_profit_trend(
    primary: pd.DataFrame,
    peer_major: Optional[pd.DataFrame] = None,
    peer_overall: Optional[pd.DataFrame] = None,
) -> Optional[Any]:
    if _plotly_available():
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

    if not _altair_available():
        return None

    if primary is None or primary.empty:
        return _altair_placeholder("売上・利益推移")

    records: List[dict[str, Any]] = []
    color_map: dict[str, str] = {
        "売上高": "#2563eb",
        "営業利益": "#10b981",
        "経常利益": "#f97316",
        "売上高 (大分類平均)": "#9ca3af",
        "経常利益 (大分類平均)": "#9ca3af",
        "売上高 (全体平均)": "#d1d5db",
        "経常利益 (全体平均)": "#d1d5db",
    }
    dash_map: dict[str, str] = {
        "売上高": "solid",
        "営業利益": "solid",
        "経常利益": "solid",
        "売上高 (大分類平均)": "dash",
        "経常利益 (大分類平均)": "dot",
        "売上高 (全体平均)": "dash",
        "経常利益 (全体平均)": "dot",
    }
    axis_map: dict[str, str] = {
        "売上高": "left",
        "営業利益": "right",
        "経常利益": "right",
        "売上高 (大分類平均)": "left",
        "経常利益 (大分類平均)": "right",
        "売上高 (全体平均)": "left",
        "経常利益 (全体平均)": "right",
    }

    def _collect_records(df: pd.DataFrame, column: str, label: str) -> None:
        if column not in df:
            return
        for year, value in df[column].items():
            if pd.isna(value):
                continue
            records.append(
                {
                    "year": int(year),
                    "value": float(value),
                    "series": label,
                    "axis": axis_map[label],
                    "dash": dash_map[label],
                }
            )

    _collect_records(primary, "sales", "売上高")
    _collect_records(primary, "operating_profit", "営業利益")
    _collect_records(primary, "ordinary_profit", "経常利益")
    if peer_major is not None and not peer_major.empty:
        _collect_records(peer_major, "sales", "売上高 (大分類平均)")
        _collect_records(peer_major, "ordinary_profit", "経常利益 (大分類平均)")
    if peer_overall is not None and not peer_overall.empty:
        _collect_records(peer_overall, "sales", "売上高 (全体平均)")
        _collect_records(peer_overall, "ordinary_profit", "経常利益 (全体平均)")

    if not records:
        return _altair_placeholder("売上・利益推移")

    data = pd.DataFrame(records)
    dash_styles = {
        "solid": [1],
        "dash": [6, 3],
        "dot": [2, 2],
    }
    color_domain = list(color_map.keys())
    base = (
        alt.Chart(data)
        .encode(
            x=alt.X("year:O", title="年"),
            color=alt.Color(
                "series:N",
                title="指標",
                scale=alt.Scale(domain=color_domain, range=[color_map[key] for key in color_domain]),
            ),
            strokeDash=alt.StrokeDash(
                "dash:N",
                scale=alt.Scale(
                    domain=list(dash_styles.keys()),
                    range=[dash_styles[name] for name in dash_styles],
                ),
            ),
            tooltip=[
                alt.Tooltip("year:O", title="年"),
                alt.Tooltip("series:N", title="指標"),
                alt.Tooltip("value:Q", title="値", format=",.0f"),
            ],
        )
    )

    left_chart = (
        base.transform_filter(alt.datum.axis == "left")
        .mark_line(point=True)
        .encode(
            y=alt.Y(
                "value:Q",
                axis=alt.Axis(title="売上高（百万円）", titleColor="#2563eb"),
                scale=alt.Scale(zero=False),
            )
        )
    )
    right_chart = (
        base.transform_filter(alt.datum.axis == "right")
        .mark_line(point=True)
        .encode(
            y=alt.Y(
                "value:Q",
                axis=alt.Axis(title="利益（百万円）", titleColor="#10b981"),
                scale=alt.Scale(zero=False),
            )
        )
    )
    chart = alt.layer(left_chart, right_chart).resolve_scale(y="independent")
    return _apply_altair_theme(chart, "売上・利益推移", height=400)


def margin_trend(primary: pd.DataFrame) -> Optional[Any]:
    if _plotly_available():
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

    if not _altair_available():
        return None

    if primary is None or primary.empty:
        return _altair_placeholder("利益率推移")

    records: List[dict[str, Any]] = []
    color_map = {
        "総利益率": "#2563eb",
        "営業利益率": "#10b981",
        "経常利益率": "#f97316",
    }
    for column, label in [
        ("gross_margin", "総利益率"),
        ("operating_margin", "営業利益率"),
        ("ordinary_margin", "経常利益率"),
    ]:
        if column not in primary:
            continue
        for year, value in primary[column].items():
            if pd.isna(value):
                continue
            records.append({"year": int(year), "value": float(value), "series": label})

    if not records:
        return _altair_placeholder("利益率推移")

    data = pd.DataFrame(records)
    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="年"),
            y=alt.Y("value:Q", title="利益率", axis=alt.Axis(format=".0%"), scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                title="指標",
                scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
            ),
            tooltip=[
                alt.Tooltip("year:O", title="年"),
                alt.Tooltip("series:N", title="指標"),
                alt.Tooltip("value:Q", title="値", format=".1%"),
            ],
        )
    )
    return _apply_altair_theme(chart, "利益率推移", height=400)


def bs_composition(primary: pd.DataFrame) -> Optional[Any]:
    if _plotly_available():
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

    if not _altair_available():
        return None

    if primary is None or primary.empty:
        return _altair_placeholder("BS 構成比")

    years = primary.index.astype(int)
    assets = primary["assets"].replace(0, np.nan)
    asset_ratio_current = primary.get("current_assets", pd.Series(index=primary.index)) / assets
    asset_ratio_fixed = primary.get("fixed_assets", pd.Series(index=primary.index)) / assets
    liabilities_ratio_current = primary.get("current_liabilities", pd.Series(index=primary.index)) / assets
    liabilities_ratio_fixed = primary.get("fixed_liabilities", pd.Series(index=primary.index)) / assets
    equity_ratio = primary.get("equity", pd.Series(index=primary.index)) / assets

    records: List[dict[str, Any]] = []
    color_map = {
        "流動資産": "#60a5fa",
        "固定資産": "#1d4ed8",
        "流動負債": "#fca5a5",
        "固定負債": "#ef4444",
        "純資産": "#10b981",
    }
    for year in years:
        for value, label, group in [
            (asset_ratio_current.get(year), "流動資産", "資産構成比"),
            (asset_ratio_fixed.get(year), "固定資産", "資産構成比"),
            (liabilities_ratio_current.get(year), "流動負債", "負債・純資産構成比"),
            (liabilities_ratio_fixed.get(year), "固定負債", "負債・純資産構成比"),
            (equity_ratio.get(year), "純資産", "負債・純資産構成比"),
        ]:
            if pd.isna(value):
                continue
            records.append(
                {
                    "year": int(year),
                    "value": float(value),
                    "category": label,
                    "group": group,
                }
            )

    if not records:
        return _altair_placeholder("BS 構成比")

    data = pd.DataFrame(records)
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            y=alt.Y("year:O", title="集計年", sort="ascending"),
            x=alt.X("value:Q", title="構成比", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "category:N",
                title="区分",
                scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
            ),
            tooltip=[
                alt.Tooltip("year:O", title="年"),
                alt.Tooltip("category:N", title="区分"),
                alt.Tooltip("value:Q", title="構成比", format=".1%"),
            ],
            column=alt.Column("group:N", title=""),
        )
    )
    chart = chart.properties(spacing=12)
    return _apply_altair_theme(chart, "BS 構成比", height=400)


def ebitda_coverage(primary: pd.DataFrame) -> Optional[Any]:
    if _plotly_available():
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

    if not _altair_available():
        return None

    if primary is None or primary.empty:
        return _altair_placeholder("EBITDA と利払")

    years = primary.index.astype(int)
    bar_records: List[dict[str, Any]] = []
    for column, label, color in [
        ("ebitda", "EBITDA", "#2563eb"),
        ("interest", "支払利息", "#94a3b8"),
    ]:
        if column not in primary:
            continue
        for year, value in primary[column].items():
            if pd.isna(value):
                continue
            bar_records.append(
                {
                    "year": int(year),
                    "value": float(value),
                    "metric": label,
                    "color": color,
                }
            )

    line_records: List[dict[str, Any]] = []
    if "interest_coverage" in primary:
        for year, value in primary["interest_coverage"].items():
            if pd.isna(value):
                continue
            line_records.append({"year": int(year), "value": float(value)})

    if not bar_records and not line_records:
        return _altair_placeholder("EBITDA と利払")

    bar_df = pd.DataFrame(bar_records) if bar_records else pd.DataFrame(columns=["year", "value", "metric", "color"])
    line_df = pd.DataFrame(line_records) if line_records else pd.DataFrame(columns=["year", "value"])

    bar_chart = (
        alt.Chart(bar_df)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="年"),
            xOffset="metric:N",
            y=alt.Y("value:Q", title="金額（百万円）"),
            color=alt.Color(
                "metric:N",
                title="項目",
                scale=alt.Scale(domain=["EBITDA", "支払利息"], range=["#2563eb", "#94a3b8"]),
            ),
            tooltip=[
                alt.Tooltip("year:O", title="年"),
                alt.Tooltip("metric:N", title="項目"),
                alt.Tooltip("value:Q", title="金額", format=",.0f"),
            ],
        )
    )

    line_chart = (
        alt.Chart(line_df)
        .mark_line(point=True, color="#f97316")
        .encode(
            x=alt.X("year:O", title="年"),
            y=alt.Y(
                "value:Q",
                axis=alt.Axis(title="カバレッジ", titleColor="#f97316"),
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("year:O", title="年"),
                alt.Tooltip("value:Q", title="カバレッジ", format=".1f"),
            ],
        )
    )

    chart = alt.layer(bar_chart, line_chart).resolve_scale(y="independent")
    return _apply_altair_theme(chart, "EBITDA と利払", height=400)


def productivity_vs_labor_share(primary: pd.DataFrame) -> Optional[Any]:
    if _plotly_available():
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

    if not _altair_available():
        return None

    if primary is None or primary.empty:
        return _altair_placeholder("労働生産性と労働分配率")

    if "labor_share" not in primary or "labor_productivity" not in primary:
        return _altair_placeholder("労働生産性と労働分配率")

    data = (
        primary.reset_index()
        .rename(columns={primary.index.name or "index": "year"})
        [["year", "labor_share", "labor_productivity"]]
    )
    data["year"] = data["year"].astype(int)

    chart = (
        alt.Chart(data)
        .mark_line(point=True, color="#2563eb")
        .encode(
            x=alt.X("labor_share:Q", title="労働分配率", axis=alt.Axis(format=".0%")),
            y=alt.Y("labor_productivity:Q", title="労働生産性（百万円）"),
            order=alt.Order("year:O"),
            tooltip=[
                alt.Tooltip("year:O", title="年"),
                alt.Tooltip("labor_share:Q", title="労働分配率", format=".1%"),
                alt.Tooltip("labor_productivity:Q", title="労働生産性", format=",.0f"),
            ],
        )
    )
    return _apply_altair_theme(chart, "労働生産性と労働分配率", height=400)


def dupont_lines(primary: pd.DataFrame) -> Optional[Any]:
    if _plotly_available():
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

    if not _altair_available():
        return None

    if primary is None or primary.empty:
        return _altair_placeholder("DuPont 分解")

    records: List[dict[str, Any]] = []
    color_map = {
        "当期純利益率": "#2563eb",
        "総資産回転率": "#10b981",
        "レバレッジ": "#f97316",
    }
    for column, label in [
        ("dupont_net_margin", "当期純利益率"),
        ("dupont_asset_turnover", "総資産回転率"),
        ("dupont_leverage", "レバレッジ"),
    ]:
        if column not in primary:
            continue
        for year, value in primary[column].items():
            if pd.isna(value):
                continue
            records.append({"year": int(year), "value": float(value), "series": label})

    if not records:
        return _altair_placeholder("DuPont 分解")

    data = pd.DataFrame(records)
    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="年"),
            y=alt.Y("value:Q", title="指標", axis=alt.Axis(format=".2f"), scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                title="指標",
                scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
            ),
            tooltip=[
                alt.Tooltip("year:O", title="年"),
                alt.Tooltip("series:N", title="指標"),
                alt.Tooltip("value:Q", title="値", format=".2f"),
            ],
        )
    )
    return _apply_altair_theme(chart, "DuPont 分解", height=400)


__all__ = [
    "FONT_FAMILY",
    "PLOTLY_IMPORT_ERROR_MESSAGE",
    "create_sparkline",
    "dupont_lines",
    "ebitda_coverage",
    "margin_trend",
    "productivity_vs_labor_share",
    "sales_profit_trend",
    "bs_composition",
]
