from __future__ import annotations

import pandas as pd

from services import duckdb_store


def test_upsert_gracefully_handles_missing_duckdb(monkeypatch, caplog):
    caplog.set_level("WARNING")
    monkeypatch.setattr(duckdb_store, "DUCKDB_AVAILABLE", False)

    store = duckdb_store.DuckDBStore()
    df = pd.DataFrame(
        {
            "集計年": [2020],
            "産業大分類コード": ["A"],
            "業種中分類コード": ["001"],
            "集計形式": ["test"],
        }
    )

    inserted = store.upsert(df)

    assert inserted == 0
    assert "uploaded data was ignored" in caplog.text
