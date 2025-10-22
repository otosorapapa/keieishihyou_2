from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, List, Optional

import pandas as pd
import streamlit as st

try:  # pragma: no cover - optional dependency
    from st_aggrid import AgGrid, GridOptionsBuilder
except ModuleNotFoundError:  # pragma: no cover - executed when package missing
    AgGrid = None  # type: ignore[assignment]
    GridOptionsBuilder = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import altair as alt
except ModuleNotFoundError:  # pragma: no cover - executed when package missing
    alt = None  # type: ignore[assignment]

AGGRID_IMPORT_ERROR_MESSAGE = "表の高度な表示機能を利用するには streamlit-aggrid をインストールしてください。"

from services.metrics import KPIResult
from ui.charts import FONT_FAMILY, create_sparkline
from ui.messages import PLOTLY_IMPORT_ERROR_MESSAGE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from plotly.graph_objects import Figure
else:
    Figure = object


def _is_plotly_figure(obj: object) -> bool:
    return hasattr(obj, "to_plotly_json")


def _render_chart(fig: Optional[object], *, use_container_width: bool = True, config: Optional[dict[str, object]] = None) -> str:
    if fig is None:
        st.warning(PLOTLY_IMPORT_ERROR_MESSAGE)
        return "none"
    if _is_plotly_figure(fig):
        st.plotly_chart(fig, use_container_width=use_container_width, config=config)
        return "plotly"
    if alt is not None and isinstance(fig, alt.Chart):
        st.altair_chart(fig, use_container_width=use_container_width)
        return "altair"
    st.warning("グラフを表示できませんでした。Plotly か Altair をインストールしてください。")
    return "none"


def apply_base_style(css_path: str = "assets/styles.css") -> None:
    try:
        with open(css_path, "r", encoding="utf-8") as fh:
            css = fh.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("スタイルシートが見つかりませんでした。")


def format_value(value: Optional[float], value_type: str) -> str:
    if value is None or pd.isna(value):
        return "—"
    if value_type == "currency":
        return f"{value:,.0f} 百万円"
    if value_type == "ratio":
        return f"{value * 100:.1f}%"
    return f"{value:,.2f}"


def format_delta(delta: Optional[float], value_type: str) -> str:
    if delta is None or pd.isna(delta):
        return "前年比: —"
    if value_type == "currency":
        return f"前年比: {delta * 100:+.1f}%"
    if value_type == "ratio":
        return f"前年比: {delta * 100:+.1f}pt"
    return f"前年比: {delta:+.1f}"


def delta_class(delta: Optional[float]) -> str:
    if delta is None or pd.isna(delta):
        return ""
    return "positive" if delta >= 0 else "negative"


def render_kpi_cards(cards: List[KPIResult]) -> None:
    if not cards:
        st.info("KPI を表示するデータがありません。")
        return

    cols = st.columns(4)
    for idx, card in enumerate(cards):
        column = cols[idx % 4]
        with column:
            container = st.container()
            with container:
                st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
                if card.tooltip:
                    st.markdown(
                        f'<span class="tooltip-icon" title="{card.tooltip}">ℹ️</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div class="kpi-title">{card.label}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="kpi-value">{format_value(card.latest_value, card.value_type)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="kpi-subtext">{format_delta(card.delta_value, card.value_type)}</div>',
                    unsafe_allow_html=True,
                )
                badge_class = delta_class(card.delta_value)
                if badge_class:
                    arrow = "↑" if card.delta_value and card.delta_value >= 0 else "↓"
                    st.markdown(
                        f'<span class="kpi-badge {badge_class}">{arrow}</span>',
                        unsafe_allow_html=True,
                    )
                sparkline_fig = create_sparkline(card.sparkline_years, card.sparkline_values)
                result = _render_chart(
                    sparkline_fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                if result == "none":
                    st.caption(PLOTLY_IMPORT_ERROR_MESSAGE)
                st.markdown("</div>", unsafe_allow_html=True)


def render_chart_block(title: str, fig: Optional[Figure], key: str) -> None:
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="chart-header"><span class="chart-title">{title}</span></div>',
            unsafe_allow_html=True,
        )
        _render_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_chart_with_download(title: str, fig: Optional[Figure], key: str) -> None:
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="chart-header"><span class="chart-title">{title}</span>'
            f'<span></span></div>',
            unsafe_allow_html=True,
        )
        result = _render_chart(fig, use_container_width=True)
        if result == "plotly" and fig is not None:
            try:
                image_bytes = fig.to_image(format="png")
            except Exception:
                image_bytes = None
            if image_bytes:
                st.download_button(
                    label="PNG ダウンロード",
                    data=image_bytes,
                    file_name=f"{key}.png",
                    mime="image/png",
                )
        elif result == "altair":
            st.caption("PNG ダウンロードは Plotly 利用時のみ利用できます。")
        st.markdown("</div>", unsafe_allow_html=True)


def render_summary_table(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("表示可能な年次サマリーがありません。")
        return
    table_df = df.reset_index()

    if GridOptionsBuilder is None or AgGrid is None:
        st.warning(AGGRID_IMPORT_ERROR_MESSAGE)
        st.dataframe(table_df, use_container_width=True)
    else:
        gb = GridOptionsBuilder.from_dataframe(table_df)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        gb.configure_default_column(resizable=True, sortable=True, filter=True)
        grid_options = gb.build()
        AgGrid(table_df, gridOptions=grid_options, theme="streamlit")
    csv_bytes = table_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="CSV ダウンロード",
        data=csv_bytes,
        file_name="年次KPI一覧.csv",
        mime="text/csv",
    )


def show_toast(message: str, icon: str = "✅") -> None:
    st.toast(f"{icon} {message}")


def render_suggestions(suggestions: dict) -> None:
    if not suggestions:
        return
    st.warning("選択された組合せのデータは未登録です。以下を参考にしてください。")
    if "same_major" in suggestions and not suggestions["same_major"].empty:
        st.markdown("#### 同一大分類の他業種")
        st.dataframe(suggestions["same_major"], use_container_width=True)
    if "overall" in suggestions and not suggestions["overall"].empty:
        st.markdown("#### 他の大分類候補")
        st.dataframe(suggestions["overall"], use_container_width=True)


__all__ = [
    "AgGrid",
    "apply_base_style",
    "format_delta",
    "format_value",
    "render_chart_block",
    "render_chart_with_download",
    "render_kpi_cards",
    "render_suggestions",
    "render_summary_table",
    "show_toast",
]
