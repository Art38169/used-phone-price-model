# tests/test_load_data.py

import pandas as pd

from src.data.load_data import load_data


def test_load_data_returns_dataframe():
    df = load_data()

    assert isinstance(df, pd.DataFrame)


def test_load_data_is_not_empty():
    df = load_data()

    assert not df.empty