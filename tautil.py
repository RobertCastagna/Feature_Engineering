import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import talib

def ohlcv(df):
    df.columns = [i.lower() for i in df.columns]
    close = pd.to_numeric(df.close)
    open = pd.to_numeric(df.open)
    high = pd.to_numeric(df.high)
    low = pd.to_numeric(df.low)
    volume = pd.to_numeric(df.volume)
    df_ohlcv = pd.DataFrame([open,high,low,close,volume]).T
    return df_ohlcv

def ta_stationary_test(df_):
    all_ta = df_ 
    result_df = pd.DataFrame()
    for i in all_ta.columns:
        adf = adfuller(all_ta[i].dropna(),autolag='AIC')
        result_df['{}'.format(i)] = pd.Series(adf[0:2],index=['Test Statistic','p_value'])
    return result_df.T

def remove_non_stationary_ta(df_):
    result_df = ta_stationary_test(df_)
    index_check = result_df[result_df['p_value']<0.05].index
    return df_[index_check]



#----------------------------------------- TA

#https://ta-lib.github.io/ta-lib-python/func_groups/momentum_indicators.html

# Momentum Indicators
from talib import (
    ADX,
    ADXR,
    MFI
    )


# Overlap Studies Indicators
from talib import (
    BBANDS,
    DEMA,
    EMA,
    HT_TRENDLINE
    )

# Volatility Indicators
from talib import (
    ATR,
    NATR,
    TRANGE
)

# Volume Indicators
from talib import (
    AD,
    ADOSC,
    OBV
    )


def get_stationary_ta_window_0(
    df_: pd.DataFrame,
    mt = 1
) -> pd.DataFrame:
    """Add volume technical analysis features to dataframe.
    Args:
        df (pandas.core.frame.DataFrame): including ohlcv
    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
        
    mt: int: multiplier for windows (i set windows 10, 15 or 20)
    """
    df = ohlcv(df_)
    open='open';high='high';low='low';close='close';volume='volume'
    
    # Average Directional Movement Index
    df[f"mom_adx_{mt}"] = ADX(
        high=df[high], low=df[low], close=df[close], timeperiod=mt
    )

    # Average Directional Movement Index Rating
    df[f"mom_adxr_{mt}"] = ADXR(
        high=df[high], low=df[low], close=df[close], timeperiod=mt
    )
    
    # Moving Average Convergence/Divergence
    df[f'mom_mfi_{mt}'] = MFI(high=df[high], low=df[low], close=df[close], volume=df[volume], timeperiod=mt)
    
    # Bollinger Bands
    uppperband, middleband, lowerband = BBANDS(df[close], timeperiod=mt)
    df[f'bb_upperband_{mt}'] = uppperband
    df[f'bb_middleband_{mt}'] = middleband
    df[f'bb_lowerband_{mt}'] = lowerband

    # Double Exponential Moving Average
    df[f'dema_{mt}'] = DEMA(df[close], timeperiod=2*mt)

    # Exponential Moving Average
    df[f'ema_{mt}'] = EMA(df[close], timeperiod=2*mt)

    # Hilbert Transform - Instantaneous Trendline
    df[f'ht_trendline_{mt}'] = HT_TRENDLINE(df[close])

    # Average True Range
    df[f'atr_{mt}'] = ATR(high=df[high], low=df[low], close=df[close], timeperiod=mt)

    # Normalized Average True Range
    df[f'natr_{mt}'] = NATR(high=df[high], low=df[low], close=df[close], timeperiod=mt)

    # True Range
    df[f'trange'] = TRANGE(high=df[high], low=df[low], close=df[close])

    # Chaikin A/D Line
    df['chaikin_ad_line'] = AD(high=df[high], low=df[low], close=df[close], volume=df[volume])

    # Chaikin A/D Oscillator
    df[f'chaikin_ad_osc_{mt}'] = ADOSC(high=df[high], low=df[low], close=df[close], volume=df[volume], fastperiod=mt, slowperiod=3*mt)
    
    # On Balance Volume
    df['obv'] = OBV(df[close], df[volume])

    df = df.drop(columns=['open','high','low','close','volume'])
    return df.dropna()


def get_stationary_ta_windows(df,mts):
    TA = []
    for mt in mts:
        TA.append(get_stationary_ta_window_0(df, mt=mt))
    TA = pd.concat(TA,axis=1)
    TA = TA.loc[:,~TA.columns.duplicated()]
    TA = TA.reindex(sorted(TA.columns),axis=1)
    return TA

    
