# -*- coding: utf-8 -*-
"""공통 지표 계산 (numpy 기반, 외부 라이브러리 추가 없음)
- 모든 함수는 '전달된 배열의 끝'까지만 사용한다 → 호출자가 기준일까지 잘라서 넘기면 룩어헤드가 없다.
"""
import numpy as np
import pandas as pd


def sma(x, n):
    """단순 이동평균. 앞쪽 n-1개는 NaN"""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if n > 0 and len(x) >= n:
        cs = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def wilder(values, n):
    """Wilder 평활: 첫 값 = 처음 n개의 단순평균, 이후 (직전값×(n-1) + 현재값) / n"""
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < n:
        return out
    out[n - 1] = values[:n].mean()
    for i in range(n, len(values)):
        out[i] = (out[i - 1] * (n - 1) + values[i]) / n
    return out


def rsi(close, n=14):
    """RSI(n), Wilder 방식. 첫 유효값은 인덱스 n (n개의 가격 변화가 필요)"""
    c = np.asarray(close, dtype=float)
    if len(c) < n + 1:
        return np.full(len(c), np.nan)
    d = np.diff(c)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag, al = wilder(gain, n), wilder(loss, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 - 100.0 / (1.0 + ag / al)
    r = np.where(al == 0, np.where(ag == 0, 50.0, 100.0), r)
    r = np.where(np.isnan(ag), np.nan, r)
    return np.r_[np.nan, r]


def atr(high, low, close, n=14):
    """ATR(n), Wilder 방식. TR = max(고-저, |고-전일종가|, |저-전일종가|). 첫 유효값은 인덱스 n-1"""
    h, l, c = (np.asarray(a, dtype=float) for a in (high, low, close))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    tr[0] = h[0] - l[0]
    return wilder(tr, n)


def volume_dry_ratio(v, end, short=5, base=20):
    """v[end-short:end] 평균 / 그 직전 v[end-short-base:end-short] 평균. end는 포함하지 않는 끝 인덱스"""
    if end - short - base < 0:
        return None
    a = float(np.mean(v[end - short:end]))
    b = float(np.mean(v[end - short - base:end - short]))
    return None if b <= 0 else a / b


def breakout_volume_mult(v, t, avg_days=50):
    """t일 거래량 / 직전 avg_days일 평균 (t일 자신은 평균에 포함하지 않음)"""
    if t - avg_days < 0:
        return None
    b = float(np.mean(v[t - avg_days:t]))
    return None if b <= 0 else float(v[t]) / b


def slope_pct(ma, t, days):
    """이동평균의 days일 변화율"""
    if t - days < 0 or not np.isfinite(ma[t]) or not np.isfinite(ma[t - days]) or ma[t - days] == 0:
        return None
    return float(ma[t] / ma[t - days] - 1)


def relative_strength(close, bench, lookback=63, slope_days=20, high_days=252):
    """벤치마크 대비 상대강도. 반환: (dict 또는 None, 계산 불가 사유)
    close, bench: 날짜 인덱스 Series. bench는 close의 날짜에 맞춰 과거 값으로만 채움(ffill)."""
    if bench is None or len(bench) == 0:
        return None, "벤치마크 데이터 없음"
    b = bench.reindex(close.index).ffill(limit=3)
    if pd.isna(b.iloc[-1]):
        return None, "기준일 벤치마크 데이터 없음"
    line = (close / b).dropna()
    if len(line) < max(lookback, slope_days) + 1:
        return None, "벤치마크와 겹치는 기간 부족"
    excess = float(line.iloc[-1] / line.iloc[-1 - lookback] - 1)
    y = np.log(line.values[-slope_days:].astype(float))
    slope = float(np.polyfit(np.arange(slope_days), y, 1)[0] * slope_days)
    win = line.values[-high_days:]
    new_high = bool(line.iloc[-1] >= win.max() * 0.98) if len(win) >= 60 else None
    return dict(excess=excess, slope=slope, new_high=new_high), None


def price_discontinuities(close, threshold=0.31, lookback=260):
    """하루 변동률 절댓값이 threshold를 넘는 날 수 (가격제한폭 ±30% 초과 → 분할·병합 미반영 의심)"""
    c = np.asarray(close, dtype=float)[-lookback - 1:]
    if len(c) < 2:
        return 0
    r = c[1:] / c[:-1] - 1
    return int((np.abs(r) > threshold).sum())
