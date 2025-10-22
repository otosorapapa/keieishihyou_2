from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import streamlit as st

from services.aggregations import compute_peer_groups, get_suggestions
from services.data_loader import (
    DataLoadError,
    available_middle_options,
    build_middle_display,
    ensure_year_range,
    load_dataset,
    load_initial_csv,
    load_uploaded_csv,
)
from services.duckdb_store import DuckDBStore
from services.metrics import build_kpi_cards, build_summary_table, compute_yearly_metrics
from ui.messages import PLOTLY_IMPORT_ERROR_MESSAGE

try:
    from ui import charts
except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
    if exc.name and exc.name.startswith("plotly"):
        st.error(PLOTLY_IMPORT_ERROR_MESSAGE)
        st.stop()
    raise

from ui.components import (
    apply_base_style,
    render_chart_with_download,
    render_kpi_cards,
    render_suggestions,
    render_summary_table,
    show_toast,
)

PAGE_TITLE = "中小企業向け経営分析ダッシュボード"


def _get_query_value(params: Dict[str, str], key: str) -> Optional[str]:
    if key not in params:
        return None
    value = params[key]
    if isinstance(value, list):
        return value[0]
    return value


def _update_query_params(major: str, middle: str, year_from: int, year_to: int) -> None:
    st.query_params.update(
        {
            "maj": major,
            "mid": middle,
            "y1": str(year_from),
            "y2": str(year_to),
        }
    )


def _format_middle(name: str) -> str:
    if not name:
        return ""
    return name.split("(")[0].strip()


def _default_selection(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    defaults: Dict[str, Optional[str]] = {"major": None, "middle": None}
    if df.empty:
        return defaults
    majors = (
        df[["産業大分類コード", "産業大分類名"]]
        .drop_duplicates()
        .sort_values("産業大分類コード")
    )
    if "D" in majors["産業大分類コード"].values:
        defaults["major"] = "D"
    else:
        defaults["major"] = str(majors.iloc[0]["産業大分類コード"])
    middle_df = available_middle_options(df, defaults["major"])
    candidate = middle_df[middle_df["業種中分類名"].str.contains("設備工事", na=False)]
    if not candidate.empty:
        defaults["middle"] = build_middle_display(candidate.iloc[0])
    elif not middle_df.empty:
        defaults["middle"] = build_middle_display(middle_df.iloc[0])
    return defaults


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    apply_base_style()

    store = DuckDBStore()
    try:
        df = load_dataset(store)
    except DataLoadError as exc:
        st.error(str(exc))
        return

    if df.empty:
        st.error("データが読み込めませんでした。CSV の内容をご確認ください。")
        return

    params = dict(st.query_params)
    defaults = _default_selection(df)

    major_options = (
        df[["産業大分類コード", "産業大分類名"]]
        .drop_duplicates()
        .sort_values("産業大分類コード")
    )
    major_map = {
        str(row["産業大分類コード"]): f"{row['産業大分類コード']}: {row['産業大分類名']}"
        for _, row in major_options.iterrows()
    }

    major_query = _get_query_value(params, "maj")
    major_keys = list(major_map.keys())
    selected_major = major_query or defaults["major"] or (major_keys[0] if major_keys else "")

    st.title(PAGE_TITLE)
    header_cols = st.columns([2.5, 2, 2, 2.5])

    with header_cols[0]:
        st.markdown("### 産業分類の選択")
        major_display = st.selectbox(
            "産業大分類コード",
            options=major_keys,
            format_func=lambda x: major_map.get(x, x),
            index=major_keys.index(selected_major)
            if selected_major in major_map
            else 0,
        )

    middle_options_df = available_middle_options(df, major_display)
    middle_options_display = [build_middle_display(row) for _, row in middle_options_df.iterrows()]
    if not middle_options_display:
        render_suggestions(get_suggestions(df, major_display, None))
        st.stop()
    middle_query = _get_query_value(params, "mid")
    if middle_query and middle_query not in middle_options_display:
        middle_query = None
    selected_middle = middle_query or defaults["middle"] or (middle_options_display[0] if middle_options_display else "")

    with header_cols[1]:
        st.markdown("### 業種中分類")
        middle_display = st.selectbox(
            "業種中分類",
            options=middle_options_display,
            index=middle_options_display.index(selected_middle)
            if selected_middle in middle_options_display
            else 0,
        )

    year_series = df["集計年"].dropna().astype(int)
    year_min = int(year_series.min())
    year_max = int(year_series.max())

    query_y1 = _get_query_value(params, "y1")
    query_y2 = _get_query_value(params, "y2")
    default_from = int(query_y1) if query_y1 else year_min
    default_to = int(query_y2) if query_y2 else year_max
    default_from = min(max(default_from, year_min), year_max)
    default_to = min(max(default_to, default_from), year_max)

    with header_cols[2]:
        st.markdown("### 表示年範囲")
        year_from, year_to = st.slider(
            "表示年範囲",
            min_value=year_min,
            max_value=year_max,
            value=(default_from, default_to),
        )

    with header_cols[3]:
        st.markdown("### データ管理")
        if st.button("データ更新", use_container_width=True):
            load_initial_csv.clear()
            st.cache_data.clear()
            show_toast("データを再読込しました。")
            st.rerun()
        uploaded_file = st.file_uploader("データ取り込み", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None and st.button("取り込み実行", use_container_width=True):
            try:
                new_df = load_uploaded_csv(uploaded_file)
                inserted = store.upsert(new_df)
                load_initial_csv.clear()
                st.cache_data.clear()
                show_toast(f"{inserted} 件のレコードを更新しました。")
                st.rerun()
            except DataLoadError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)

    _update_query_params(major_display, middle_display, year_from, year_to)

    filtered = df[(df["産業大分類コード"] == major_display)]
    filtered = filtered[
        filtered["業種中分類名"].str.strip().fillna("") == _format_middle(middle_display)
    ]
    filtered = ensure_year_range(filtered, year_from, year_to)

    metrics_df = compute_yearly_metrics(filtered)
    metrics_in_range = metrics_df.loc[(metrics_df.index >= year_from) & (metrics_df.index <= year_to)]

    if metrics_in_range.empty:
        render_suggestions(get_suggestions(df, major_display, middle_display))
        return

    kpi_cards = build_kpi_cards(metrics_in_range)
    render_kpi_cards(kpi_cards)

    peers = compute_peer_groups(df, major_display)
    peers_trimmed: Dict[str, pd.DataFrame] = {}
    for key, peer_df in peers.items():
        if peer_df is None or peer_df.empty:
            continue
        peers_trimmed[key] = peer_df.loc[(peer_df.index >= year_from) & (peer_df.index <= year_to)]

    sales_fig = charts.sales_profit_trend(
        metrics_in_range,
        peers_trimmed.get("major"),
        peers_trimmed.get("overall"),
    )
    margin_fig = charts.margin_trend(metrics_in_range)
    bs_fig = charts.bs_composition(metrics_in_range)
    ebitda_fig = charts.ebitda_coverage(metrics_in_range)
    productivity_fig = charts.productivity_vs_labor_share(metrics_in_range)
    dupont_fig = charts.dupont_lines(metrics_in_range)

    render_chart_with_download("売上・利益推移", sales_fig, "sales_profit")
    render_chart_with_download("利益率推移", margin_fig, "margin_trend")
    render_chart_with_download("BS 構成比", bs_fig, "bs_composition")
    render_chart_with_download("EBITDA と利払", ebitda_fig, "ebitda_coverage")
    render_chart_with_download("労働生産性と労働分配率", productivity_fig, "productivity_labor")
    render_chart_with_download("DuPont 分解", dupont_fig, "dupont")

    latest_year = int(metrics_in_range.index.max())
    latest_metrics = metrics_in_range.loc[latest_year]
    st.markdown("### 成長性")
    growth_cols = st.columns(3)
    growth_cols[0].metric("売上高 YoY", f"{latest_metrics['sales_yoy'] * 100:.1f}%" if pd.notna(latest_metrics['sales_yoy']) else "—")
    growth_cols[1].metric("営業利益 YoY", f"{latest_metrics['operating_yoy'] * 100:.1f}%" if pd.notna(latest_metrics['operating_yoy']) else "—")
    growth_cols[2].metric("売上高 3年CAGR", f"{latest_metrics['sales_cagr_3y'] * 100:.1f}%" if pd.notna(latest_metrics['sales_cagr_3y']) else "—")

    summary_table = build_summary_table(metrics_in_range)
    st.markdown("### 年次 KPI 一覧")
    render_summary_table(summary_table)


if __name__ == "__main__":
    main()
