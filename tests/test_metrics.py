import numpy as np
import pandas as pd

from services.metrics import compute_yearly_metrics, safe_div


def test_safe_div_handles_zero_denominator():
    numerator = pd.Series([10, 20, 30], index=[2019, 2020, 2021])
    denominator = pd.Series([5, 0, np.nan], index=[2019, 2020, 2021])
    result = safe_div(numerator, denominator)
    assert np.isclose(result.iloc[0], 2.0)
    assert np.isnan(result.iloc[1])
    assert np.isnan(result.iloc[2])


def test_compute_yearly_metrics_combines_depreciation_columns():
    data = pd.DataFrame(
        {
            "集計年": [2020, 2021],
            "売上高（百万円）": [1000, 1200],
            "営業利益（百万円）": [100, 150],
            "減価償却費（百万円）": [40, 50],
            "減価償却費（百万円）.1": [10, 5],
        }
    )
    metrics = compute_yearly_metrics(data)
    assert np.isclose(metrics.loc[2020, "ebitda"], 150.0)
    assert np.isclose(metrics.loc[2021, "ebitda"], 205.0)


def test_compute_yearly_metrics_growth_values():
    data = pd.DataFrame(
        {
            "集計年": [2019, 2020, 2021],
            "売上高（百万円）": [100, 120, 180],
            "営業利益（百万円）": [10, 12, 15],
        }
    )
    metrics = compute_yearly_metrics(data)
    assert np.isclose(metrics.loc[2020, "sales_yoy"], 0.2)
    assert np.isclose(metrics.loc[2021, "sales_yoy"], 0.5)
    assert np.isclose(metrics.loc[2020, "operating_yoy"], 0.2)
