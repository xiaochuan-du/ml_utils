"""
    Test strcuture data processing workflow
"""
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../ml_utils')

import pandas as pd
from ml_utils.structure.utils import (get_info_from_date, add_ts_elapsed)

# from ml_utils import structure


def test_get_info_from_date():
    test_timestamp = 1506787200000
    test_datetime = pd.to_datetime(test_timestamp, unit='ms', utc=True)
    input_df = pd.DataFrame({
        'Date': [test_datetime] * 5,
        'indicator': [1, 2, 3, 4, 5],
    })
    output_df = get_info_from_date(input_df, ['Date'])
    ordered_indicators = input_df.indicator.tolist()
    expected_indicators = output_df.indicator.tolist()
    assert ordered_indicators == expected_indicators


def test_add_ts_elapsed():
    "test_add_ts_elapsed"
    fld = 'holiday_flg'
    test_datetime = pd.date_range(start='2017-02-03', periods=5, freq='D')

    input_df = pd.DataFrame({
        'Date': test_datetime,
        'indicator': [1, 2, 3, 4, 5],
        fld: [0, 1, 0, 0, 1]
    })
    # input_df.Date = input_df.Date.dt.strftime('%Y-%m-%d')

    output_df = add_ts_elapsed(fld, ['af_', 'be_'], input_df, 'Date')
    # hol = add_ts_elapsed(fld, ['dur_'], hol)
    print(output_df.head())
    ordered_indicators = input_df.indicator.tolist()
    expected_indicators = output_df.indicator.tolist()
    assert ordered_indicators == expected_indicators


def test_add_ts_elapsed_dur():
    "test_add_ts_elapsed_dur"
    fld = 'holiday_flg'
    test_datetime = pd.date_range(start='2017-02-03', periods=5, freq='D')

    input_df = pd.DataFrame({
        'Date': test_datetime,
        'indicator': [1, 2, 3, 4, 5],
        fld: [0, 1, 1, 0, 1]
    })
    # input_df.Date = input_df.Date.dt.strftime('%Y-%m-%d')

    output_df = add_ts_elapsed(fld, ['dur_'], input_df, 'Date')
    ordered_indicators = input_df.indicator.tolist()
    expected_indicators = output_df.indicator.tolist()
    assert ordered_indicators == expected_indicators
