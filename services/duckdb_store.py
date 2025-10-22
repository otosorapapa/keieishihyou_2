from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

DUCKDB_PATH = Path("app.duckdb")
TABLE_NAME = "financials"


class DuckDBStore:
    def __init__(self, path: Path = DUCKDB_PATH) -> None:
        self.path = path

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.path))

    def ensure_table(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        with self._connect() as conn:
            conn.register("tmp_df", df)
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS
                SELECT * FROM tmp_df WHERE 1 = 0
                """
            )

    def upsert(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0

        self.ensure_table(df)
        with self._connect() as conn:
            conn.register("tmp_df", df)
            conn.execute(
                f"""
                DELETE FROM {TABLE_NAME}
                WHERE (集計年, 産業大分類コード, 業種中分類コード, 集計形式) IN (
                    SELECT DISTINCT 集計年, 産業大分類コード, 業種中分類コード, 集計形式 FROM tmp_df
                )
                """
            )
            conn.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM tmp_df")
            result = conn.execute("SELECT rowcount()")
            inserted = result.fetchone()[0]
        return inserted

    def read_all(self) -> Optional[pd.DataFrame]:
        if not self.path.exists():
            return None
        with self._connect() as conn:
            tables = conn.execute("SHOW TABLES").fetchall()
            if (TABLE_NAME,) not in tables:
                return None
            df = conn.execute(f"SELECT * FROM {TABLE_NAME}").fetchdf()
        return df


__all__ = ["DuckDBStore", "DUCKDB_PATH", "TABLE_NAME"]
