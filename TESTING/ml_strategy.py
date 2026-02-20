"""Rolling Walk-Forward LightGBM alpha engine (Phase 5 - Enhanced).

Key improvements over Phase 3/4:
  - 5-day forward return target (less noisy than 1-day)
  - 20+ technical features (Bollinger, ADX, CCI, OBV, etc.)
  - Early stopping with temporal validation split
  - Stronger regularization (reg_alpha, reg_lambda)
  - No StandardScaler (unnecessary for tree models)
"""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, cast

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import ML_PARAMS

logger = logging.getLogger(__name__)

try:
    talib: Any = import_module("talib")
except ImportError:  # pragma: no cover
    talib = None


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------

def _read_price_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df = df.sort_index()
    needed_cols = ["open", "high", "low", "close", "vol", "amount"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing cols: {missing}")
    return df[needed_cols].copy()


# ---------------------------------------------------------------------------
# Enhanced Feature Engineering (20+ features)
# ---------------------------------------------------------------------------

def _calc_features(df: pd.DataFrame) -> pd.DataFrame:
    if talib is None:
        raise ImportError("TA-Lib not found. Please install ta-lib first.")

    feat = df.copy()

    c = np.asarray(feat["close"].to_numpy(dtype=float), dtype=float)
    h = np.asarray(feat["high"].to_numpy(dtype=float), dtype=float)
    lo = np.asarray(feat["low"].to_numpy(dtype=float), dtype=float)
    v = np.asarray(feat["vol"].to_numpy(dtype=float), dtype=float)
    safe_c = np.where(c == 0, np.nan, c).astype(float)

    # --- Original 10 features ---
    feat["rsi14"] = talib.RSI(c, timeperiod=14)
    feat["roc5"] = talib.ROC(c, timeperiod=5)
    feat["roc20"] = talib.ROC(c, timeperiod=20)

    macd_line, macd_signal, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    feat["macd"] = macd_line
    feat["macd_hist"] = macd_hist

    sma5 = talib.SMA(c, timeperiod=5)
    sma20 = talib.SMA(c, timeperiod=20)
    sma60 = talib.SMA(c, timeperiod=60)
    safe_sma60 = np.where(sma60 == 0, np.nan, sma60).astype(float)
    feat["sma_ratio_5_60"] = sma5 / safe_sma60

    atr14 = talib.ATR(h, lo, c, timeperiod=14)
    feat["atr14_norm"] = atr14 / safe_c

    vol_sma20 = talib.SMA(v, timeperiod=20)
    safe_vol_sma20 = np.where(vol_sma20 == 0, np.nan, vol_sma20).astype(float)
    feat["vol_ratio_20"] = v / safe_vol_sma20

    feat["ret_1"] = feat["close"].pct_change(1)
    feat["ret_5"] = feat["close"].pct_change(5)

    # --- NEW: Momentum & Trend features ---
    feat["rsi5"] = talib.RSI(c, timeperiod=5)
    feat["roc10"] = talib.ROC(c, timeperiod=10)
    feat["adx14"] = talib.ADX(h, lo, c, timeperiod=14)
    feat["cci14"] = talib.CCI(h, lo, c, timeperiod=14)
    feat["willr14"] = talib.WILLR(h, lo, c, timeperiod=14)
    feat["mfi14"] = talib.MFI(h, lo, c, v, timeperiod=14)

    # --- NEW: Bollinger Band position ---
    bb_upper, bb_mid, bb_lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
    bb_range = np.where((bb_upper - bb_lower) == 0, np.nan, bb_upper - bb_lower)
    feat["bb_pos"] = (c - bb_lower) / bb_range  # 0~1, position within bands

    # --- NEW: Moving average cross features ---
    safe_sma20 = np.where(sma20 == 0, np.nan, sma20).astype(float)
    feat["sma_ratio_5_20"] = sma5 / safe_sma20
    feat["price_vs_sma20"] = c / safe_sma20 - 1.0  # distance from MA20

    # --- NEW: Volatility features ---
    feat["atr5_norm"] = talib.ATR(h, lo, c, timeperiod=5) / safe_c
    feat["ret_10"] = feat["close"].pct_change(10)
    feat["ret_20"] = feat["close"].pct_change(20)

    # --- NEW: Volume dynamics ---
    vol_sma5 = talib.SMA(v, timeperiod=5)
    safe_vol5 = np.where(vol_sma5 == 0, np.nan, vol_sma5).astype(float)
    feat["vol_ratio_5"] = v / safe_vol5
    # price-volume divergence
    feat["pv_corr_10"] = feat["close"].rolling(10).corr(feat["vol"])

    # --- NEW: Candlestick pattern features ---
    body = c - feat["open"].values.astype(float)
    candle_range = h - lo
    safe_range = np.where(candle_range == 0, np.nan, candle_range)
    feat["body_ratio"] = body / safe_range  # -1 to 1

    # --- Label: 5-day forward return > 1% threshold ---
    feat["fwd_ret_5"] = feat["close"].shift(-5) / feat["close"] - 1.0
    feat["label"] = (feat["fwd_ret_5"] > 0.01).astype(int)

    return feat


# ---------------------------------------------------------------------------
# Model Training with Early Stopping
# ---------------------------------------------------------------------------

def _fit_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> Optional[lgb.LGBMClassifier]:
    """Train LightGBM with temporal validation split and early stopping."""
    X_all = train_df[feature_cols].values
    y_all = train_df["label"].values.astype(int)

    # Temporal split: train on first 80%, validate on last 20%
    split_idx = int(len(X_all) * 0.8)
    if split_idx < 50 or (len(X_all) - split_idx) < 10:
        # Not enough data for proper split, train without early stopping
        X_train, y_train = X_all, y_all
        eval_set = None
    else:
        X_train, y_train = X_all[:split_idx], y_all[:split_idx]
        X_val, y_val = X_all[split_idx:], y_all[split_idx:]
        eval_set = [(X_val, y_val)]

    # Check label balance
    if len(np.unique(y_train)) < 2:
        return None

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=ML_PARAMS.n_estimators,
        learning_rate=ML_PARAMS.learning_rate,
        num_leaves=ML_PARAMS.num_leaves,
        min_child_samples=ML_PARAMS.min_child_samples,
        subsample=ML_PARAMS.subsample,
        colsample_bytree=ML_PARAMS.colsample_bytree,
        reg_alpha=ML_PARAMS.reg_alpha,
        reg_lambda=ML_PARAMS.reg_lambda,
        random_state=ML_PARAMS.random_state,
        n_jobs=ML_PARAMS.n_jobs_model,
        class_weight="balanced",
        verbosity=-1,
    )

    callbacks = []
    if eval_set is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))
        callbacks.append(lgb.log_evaluation(period=-1))  # suppress logging

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        callbacks=callbacks if callbacks else None,
    )
    return model


# ---------------------------------------------------------------------------
# Rolling Walk-Forward with pred_prob output (0.0 ~ 1.0)
# ---------------------------------------------------------------------------

def _walk_forward_predict(feature_df: pd.DataFrame) -> pd.DataFrame:
    cols = ML_PARAMS.feature_columns
    work = feature_df.copy()

    model: Optional[lgb.LGBMClassifier] = None
    last_retrain_idx = -999
    retrain_count = 0

    total_len = len(work)
    start_idx = max(ML_PARAMS.min_train_size, ML_PARAMS.lookback_window)

    pred_probs = np.zeros(total_len, dtype=float)

    for current_idx in range(start_idx, total_len):
        # ---- Retrain check ----
        if (current_idx - last_retrain_idx) >= ML_PARAMS.retrain_frequency:
            train_end = current_idx - 1
            train_start = max(0, train_end - ML_PARAMS.lookback_window + 1)

            train_slice = work.iloc[train_start: train_end + 1].dropna(
                subset=cols + ["label"]
            )
            if (
                len(train_slice) >= ML_PARAMS.min_train_size
                and train_slice["label"].nunique() > 1
            ):
                fitted = _fit_model(train_slice, cols)
                if fitted is not None:
                    model = fitted
                    last_retrain_idx = current_idx
                    retrain_count += 1
            else:
                model = None

        # ---- Daily probability prediction ----
        if model is not None:
            row = work.iloc[current_idx]
            if not row[cols].isna().any():
                X_live = np.asarray(row[cols], dtype=float).reshape(1, -1)
                proba = np.asarray(model.predict_proba(X_live))
                prob = float(proba[0, 1])
                pred_probs[current_idx] = prob

    work["pred_prob"] = pred_probs
    logger.debug(
        "walk-forward done: retrains=%d, non-zero rows=%d",
        retrain_count,
        int((pred_probs > 0).sum()),
    )
    return work


# ---------------------------------------------------------------------------
# 单票入口
# ---------------------------------------------------------------------------

def process_single_ticker(csv_path: Path) -> tuple[str, Optional[pd.DataFrame]]:
    ticker = csv_path.stem
    try:
        raw = _read_price_data(csv_path)
        feat = _calc_features(raw)
        pred_df = _walk_forward_predict(feat)
        out = pred_df[["open", "high", "low", "close", "vol", "amount", "pred_prob"]].copy()
        out = out.dropna(subset=["open", "high", "low", "close", "vol", "amount"])
        return ticker, out
    except Exception as exc:
        logger.exception("处理 %s 失败: %s", ticker, exc)
        return ticker, None


# ---------------------------------------------------------------------------
# 并行调度
# ---------------------------------------------------------------------------

def train_and_predict_all(
    data_dir: Path,
    output_file: Path,
    n_jobs: int = -1,
) -> Dict[str, pd.DataFrame]:
    """并行训练全股票并输出预测字典 (pred_prob 版)。"""
    if ML_PARAMS.min_train_size > ML_PARAMS.lookback_window:
        raise ValueError(
            f"参数冲突: min_train_size({ML_PARAMS.min_train_size}) > "
            f"lookback_window({ML_PARAMS.lookback_window})"
        )

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未在 {data_dir} 找到可训练 CSV 数据")

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(process_single_ticker)(path) for path in csv_files
    )
    if results is None:
        raise RuntimeError("并行执行失败，未返回结果")
    results = cast(list[tuple[str, Optional[pd.DataFrame]]], results)

    signal_dict: Dict[str, pd.DataFrame] = {}
    total_nonzero = 0
    for ticker, df in results:
        if df is not None and not df.empty:
            signal_dict[ticker] = df
            total_nonzero += int((df["pred_prob"] > 0).sum())

    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(signal_dict, output_file)
    logger.info(
        "ML 预测完成: 成功 %d / 总计 %d, 非零概率行=%d, 输出: %s",
        len(signal_dict),
        len(csv_files),
        total_nonzero,
        output_file,
    )
    return signal_dict


def load_prediction_dict(pickle_path: Path) -> Dict[str, pd.DataFrame]:
    if not pickle_path.exists():
        raise FileNotFoundError(f"预测文件不存在: {pickle_path}")
    data = joblib.load(pickle_path)
    if not isinstance(data, dict):
        raise ValueError("预测文件格式错误，期望 Dict[str, DataFrame]")
    return data
