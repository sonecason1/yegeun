# -*- coding: utf-8 -*-
"""
미너비니 5대 진입모델 독립 탐지 + 모델별 점수화
==============================================
- 공통 추세 필터(이평선 위치·기울기, 유동성, 장기 하락, 급락·하락일 대량거래)는 공유하고,
  패턴 판정은 모델마다 독립 함수로 구현: VCP / 컵 위드 핸들 / 3C / 플랫 베이스 / 파워 플레이
- 룩어헤드 방지: 기준일 T 판정에서 '구조(피벗·무효화 가격)'는 T-1일까지의 봉만으로 계산하고,
  T일 봉은 돌파·무효화 여부 판단에만 쓴다. 스윙 고·저점도 이미 지나간 봉으로만 확정한다.
- 점수 = Σ(가중치 × 항목점수 0~1). 계산 불가 항목(예: 벤치마크 없음)은 빼고 정규화(0~100)한다.
- config.json의 모든 임계값·가중치는 '초기값'이며 검증된 최적값이 아니다.
"""
import json
import os
import re

import numpy as np
import pandas as pd

from indicators import (atr, breakout_volume_mult, price_discontinuities, relative_strength, rsi, slope_pct, sma,
                        volume_dry_ratio)
from vcp import swings  # 기존 VCP 스크리너의 스윙 탐지 재사용

HERE = os.path.dirname(os.path.abspath(__file__))
FORM, SIGNAL, WARN, NODATA = "패턴 형성 중", "진입 신호 확인", "주의/무효화", "데이터 부족"
STATUS_ORDER = {SIGNAL: 0, FORM: 1, WARN: 2, NODATA: 3}
MODEL_KEYS = ["vcp", "cup", "c3", "flat", "power"]
COMP_KEYS = ["trend", "pattern", "contraction", "volume", "atr", "rsi", "rs", "pivot", "liquidity"]
COMP_NAMES = dict(trend="추세", pattern="패턴", contraction="수축·범위", volume="거래량", atr="ATR 변동성",
                  rsi="RSI", rs="상대강도", pivot="피벗 근접·돌파", liquidity="유동성·데이터")
COLS = ["Open", "High", "Low", "Close", "Volume"]


def load_config(path=None):
    with open(path or os.path.join(HERE, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _num(x, nd=4):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def _nf(reason):
    return {"found": False, "reason": reason}


def short_reason(why):
    """탈락 사유 집계용: 괄호 속 수치 제거"""
    return re.sub(r"\s*\(.*?\)", "", why or "").strip()


# ───────────────────────── 공통 컨텍스트 ─────────────────────────
def build_context(d, bench, cfg):
    raw = d.dropna(subset=COLS)
    dc = cfg["data"]
    zero = int((raw["Volume"].iloc[-dc["zero_volume_lookback"]:] <= 0).sum()) if len(raw) else 0
    d = raw[raw["Volume"] > 0]                      # 거래량 0(거래정지)일은 기존 방식대로 제외하고 횟수만 기록
    n = len(d)
    ctx = dict(df=d, n=n, zero_vol_days=zero, date=d.index[-1] if n else None, flags=[])
    if n < dc["min_bars_power"]:
        return ctx
    O, H, L, C, V = (d[k].values.astype(float) for k in COLS)
    T = n - 1
    mc = cfg["ma"]
    ma50, ma150, ma200 = sma(C, mc["fast"]), sma(C, mc["mid"]), sma(C, mc["slow"])
    r = rsi(C, cfg["rsi"]["period"])
    a = atr(H, L, C, cfg["atr"]["period"])
    atrp = a / C
    ac, vc, rc = cfg["atr"], cfg["volume"], cfg["relative_strength"]
    f = lambda arr: _num(arr[T]) if np.isfinite(arr[T]) else None  # noqa: E731
    rsd = cfg["rsi"]["slope_days"]
    rec = np.nanmean(atrp[T - ac["recent_days"] + 1:T + 1])
    pri = np.nanmean(atrp[T - ac["recent_days"] - ac["prior_days"] + 1:T - ac["recent_days"] + 1])
    atr_tr = float(rec / pri) if np.isfinite(rec) and np.isfinite(pri) and pri > 0 else None
    rs, rs_na = relative_strength(pd.Series(C, index=d.index), bench, rc["lookback"], rc["slope_days"], rc["new_high_days"])
    ctx.update(
        T=T, o=O, h=H, l=L, c=C, v=V, close=float(C[T]), high=float(H[T]), low=float(L[T]),
        change=float(C[T] / C[T - 1] - 1),
        ma50=f(ma50), ma150=f(ma150), ma200=f(ma200),
        s50=slope_pct(ma50, T, mc["slope_days"]), s150=slope_pct(ma150, T, mc["slope_days"]),
        s200=slope_pct(ma200, T, mc["slope_days"]),
        rsi=f(r), rsi_chg=_num(r[T] - r[T - rsd]) if np.isfinite(r[T]) and np.isfinite(r[T - rsd]) else None,
        atrp=f(atrp), atrp_trend=atr_tr,
        vol_dry=volume_dry_ratio(V, T, vc["short_days"], vc["base_days"]),        # 기준일 제외(구조 구간)
        brk_mult=breakout_volume_mult(V, T, vc["breakout_avg_days"]),            # 기준일 / 직전 50일 평균
        turnover20=float(np.mean((C * V)[-cfg["universe"]["turnover_days"]:])),
        hi52=float(H[-252:].max()), lo52=float(L[-252:].min()),
        rs=rs, rs_na=rs_na,
        sx=dict(h=H[:T], l=L[:T], c=C[:T], v=V[:T], atrp=atrp[:T], ma50=ma50[:T], ma200=ma200[:T]),
    )
    # 위험 신호
    rf = cfg["risk_flags"]
    ret = np.r_[np.nan, C[1:] / C[:-1] - 1]
    avg50 = np.r_[np.full(50, np.nan), sma(V, 50)[49:-1]] if n > 50 else np.full(n, np.nan)
    crash = [j for j in range(max(51, T - rf["crash_lookback"] + 1), T + 1)
             if ret[j] <= rf["crash_day_return"] and V[j] >= rf["crash_volume_mult"] * avg50[j]]
    if crash:
        ctx["flags"].append(dict(text=f"급락+대량거래 ({d.index[crash[-1]]:%m-%d}, {ret[crash[-1]]:.0%})", scope="all"))
    dist = sum(1 for j in range(max(51, T - rf["distribution_days"] + 1), T + 1)
               if ret[j] < -0.002 and V[j] >= rf["distribution_volume_mult"] * avg50[j])
    ctx["distribution"] = dist
    if zero > dc["max_zero_volume_days"]:
        ctx["flags"].append(dict(text=f"거래정지·무거래일 다수 (최근 {dc['zero_volume_lookback']}일 중 {zero}일)", scope="all"))
    disc = price_discontinuities(C, dc["max_abs_daily_change"], dc["discontinuity_lookback"])
    if disc:
        ctx["flags"].append(dict(text=f"가격 불연속 {disc}회 (분할·병합 미반영 의심)", scope="all"))
    return ctx


def common_fields(ctx, cfg):
    out = dict(asof=ctx["date"].strftime("%Y-%m-%d") if ctx["date"] is not None else None, bars=ctx["n"])
    if "T" not in ctx:
        return out
    tr = ctx["atrp_trend"]
    rs = ctx["rs"] or {}
    out.update(
        close=_num(ctx["close"], 2), change=_num(ctx["change"]), rsi=_num(ctx["rsi"], 1), rsi_chg=_num(ctx["rsi_chg"], 1),
        atrp=_num(ctx["atrp"]), atrp_trend=_num(tr, 3),
        atrp_dir=None if tr is None else ("축소" if tr < 0.9 else "확대" if tr > 1.1 else "유지"),
        vol_dry=_num(ctx["vol_dry"], 3), brk_mult=_num(ctx["brk_mult"], 2),
        ma50=_num(ctx["ma50"], 2), ma150=_num(ctx["ma150"], 2), ma200=_num(ctx["ma200"], 2),
        ma50_slope=_num(ctx["s50"]), ma150_slope=_num(ctx["s150"]), ma200_slope=_num(ctx["s200"]),
        rs_excess=_num(rs.get("excess")), rs_slope=_num(rs.get("slope")), rs_new_high=rs.get("new_high"),
        rs_na=ctx["rs_na"], turnover_eok=_num(ctx["turnover20"] / 1e8, 1),
        from_high=_num(ctx["close"] / ctx["hi52"] - 1), distribution_days=ctx["distribution"],
        warnings=[f["text"] for f in ctx["flags"]],
    )
    return out


# ───────────────────────── 스윙 구간 ─────────────────────────
def _alternate(pts):
    out = []
    for p in pts:
        if out and out[-1][1] == p[1]:
            if (p[1] == "H" and p[2] >= out[-1][2]) or (p[1] == "L" and p[2] <= out[-1][2]):
                out[-1] = p
            continue
        out.append(p)
    return out


def _segments(h, l, start, k, ms):
    """start(기준 고점) 이후 고점-저점 교대 목록. 마지막 미확정 구간은 기존 VCP와 같은 규칙으로 처리"""
    seg_h, seg_l = h[start:], l[start:]
    if len(seg_h) < 2:
        return []
    sw = _alternate([(0, "H", float(seg_h[0]))] + [p for p in swings(seg_h, seg_l, k, ms) if p[0] > 0])
    last_i = sw[-1][0]
    if last_i < len(seg_h) - 1:
        th, tl = seg_h[last_i + 1:], seg_l[last_i + 1:]
        if sw[-1][1] == "H":
            j = int(np.argmin(tl))
            if 1 - tl[j] / sw[-1][2] >= ms / 2:
                sw.append((last_i + 1 + j, "L", float(tl[j])))
        else:
            j = int(np.argmax(th))
            sw.append((last_i + 1 + j, "H", float(th[j])))
    return [(start + i, t, p) for i, t, p in sw]


def _ratio_score(r, strong=0.7, partial=1.0):
    if r is None or not np.isfinite(r):
        return None
    return 1.0 if r <= strong else 0.5 if r <= partial else 0.0


# ───────────────────────── A. VCP ─────────────────────────
def detect_vcp(ctx, m, cfg):
    x = ctx["sx"]
    h, l, v, ap = x["h"], x["l"], x["v"], x["atrp"]
    n = len(h)
    s0 = max(0, n - m["lookback"])
    bi = s0 + int(np.argmax(h[s0:]))
    base_len = n - 1 - bi
    if base_len < m["min_base_days"]:
        return _nf(f"베이스 고점 이후 기간 부족 ({base_len}일)")
    pre = l[max(0, bi - 126):bi]
    if len(pre) < 20:
        return _nf("선행 상승 측정 구간 부족")
    runup = h[bi] / pre.min() - 1
    if runup < m["prior_runup_min"]:
        return _nf(f"선행 상승 부족 ({runup:.0%})")
    sw = _segments(h, l, bi, cfg["swing"]["k"], cfg["swing"]["min_swing"])
    pairs = [(a, b, 1 - b[2] / a[2]) for a, b in zip(sw, sw[1:]) if a[1] == "H" and b[1] == "L"]
    if len(pairs) < m["min_contractions"]:
        return _nf(f"유효 수축 부족 ({len(pairs)}회)")
    pairs = pairs[-m["max_contractions"]:]
    dep = [p[2] for p in pairs]
    if dep[0] > m["max_first_depth"]:
        return _nf(f"첫 수축 과도 ({dep[0]:.0%})")
    if dep[-1] > m["max_last_depth"]:
        return _nf(f"마지막 수축 과도 ({dep[-1]:.0%})")
    if dep[-1] >= dep[0]:
        return _nf(f"수축이 작아지지 않음 ({dep[0]:.0%}→{dep[-1]:.0%})")
    (pi, _, piv), (li, _, low) = pairs[-1][0], pairs[-1][1]
    if li + 1 < n and h[li + 1:].max() > piv:
        return _nf("구조 기간 안에서 피벗 상향 돌파 완료")
    tol = m["shrink_tolerance"]
    shrink = float(np.mean([dep[i] <= dep[i - 1] * tol for i in range(1, len(dep))]))
    ratio = dep[-1] / dep[0]
    ratio_s = 1.0 if ratio <= 0.35 else 0.6 if ratio <= 0.6 else 0.3
    last_s = 1.0 if dep[-1] <= m["last_depth_good"] else 0.6 if dep[-1] <= m["last_depth_partial"] else 0.3
    nc = len(dep)
    cnt_s = 1.0 if 3 <= nc <= 4 else 0.8 if nc >= 5 else 0.7
    run_s = 1.0 if runup >= m["prior_runup_good"] else 0.6
    len_s = 1.0 if base_len >= 25 else 0.7
    segv = [float(v[a[0]:b[0] + 1].mean()) for a, b, _ in pairs]
    sega = [float(np.nanmean(ap[a[0]:b[0] + 1])) for a, b, _ in pairs]
    vtrend = float(np.mean([segv[i] <= segv[i - 1] for i in range(1, len(segv))]))
    vrat = segv[-1] / segv[0] if segv[0] > 0 else None
    vb = _ratio_score(vrat, 0.7, 1.0)
    plus, minus = [], []
    (plus if shrink >= 0.99 else minus).append(f"수축 {'→'.join(f'{d*100:.0f}%' for d in dep)}")
    (plus if dep[-1] <= m["last_depth_good"] else minus).append(f"마지막 수축 {dep[-1]:.1%}")
    if vrat is not None:
        (plus if vrat <= 0.85 else minus).append(f"수축 구간 거래량 첫→마지막 {vrat:.2f}배")
    return dict(found=True, pivot=float(piv), invalid=float(low),
                invalid_rule=f"종가가 마지막 수축 저점 {low:,.0f} 아래로 마감",
                pattern=(cnt_s + run_s + len_s) / 3, contraction=0.5 * last_s + 0.3 * shrink + 0.2 * ratio_s,
                vol_base=None if vb is None else 0.5 * vtrend + 0.5 * vb,
                atr_ratio=sega[-1] / sega[0] if sega[0] > 0 else None,
                info=dict(contractions="→".join(f"{d*100:.0f}%" for d in dep), n_contr=nc, base_days=base_len,
                          prior_runup=_num(runup, 3), seg_vol_ratio=_num(vrat, 3)),
                plus=plus, minus=minus)


# ───────────────────────── B. 컵 위드 핸들 ─────────────────────────
def detect_cup(ctx, m, cfg):
    x = ctx["sx"]
    h, l, c, v, ap = x["h"], x["l"], x["c"], x["v"], x["atrp"]
    n = len(h)
    hmin, hmax = m["handle_min_days"], m["handle_max_days"]
    a, b = max(0, n - m["cup_max_days"] - hmin), n - m["cup_min_days"] - hmin
    if b - a < 10:
        return _nf("컵 측정 데이터 부족")
    lp = a + int(np.argmax(h[a:b]))
    if h[lp + 1:].max() > h[lp] * (1 + m["max_rim_above_left"]):
        return _nf("왼쪽 고점을 크게 넘는 상승 (컵 아님 또는 이미 돌파)")
    bt = lp + int(np.argmin(l[lp:]))
    if n - 1 - bt < hmin + 5:
        return _nf("컵 바닥이 너무 최근")
    r = bt + int(np.argmax(h[bt:]))
    hlen = n - 1 - r
    if hlen < hmin:
        return _nf(f"손잡이 형성 전 ({hlen}일)")
    if hlen > hmax:
        return _nf(f"손잡이 기간 초과 ({hlen}일)")
    depth = 1 - l[bt] / h[lp]
    if depth < m["min_depth"]:
        return _nf(f"컵 깊이 부족 ({depth:.0%})")
    if depth > m["max_depth_deep"]:
        return _nf(f"컵 깊이 과도 ({depth:.0%})")
    cup_len = r - lp
    if cup_len < m["cup_min_days"]:
        return _nf(f"컵 기간 부족 ({cup_len}일)")
    rim = (h[r] - l[bt]) / (h[lp] - l[bt])
    if rim < m["min_rim_recovery"]:
        return _nf(f"오른쪽 고점 회복 부족 ({rim:.0%})")
    pre = l[max(0, lp - 126):lp]
    runup = h[lp] / pre.min() - 1 if len(pre) >= 20 else None
    if runup is None or runup < m["prior_runup_min"]:
        return _nf("선행 상승 부족" + ("" if runup is None else f" ({runup:.0%})"))
    hl_i = r + 1 + int(np.argmin(l[r + 1:]))
    hlow = l[hl_i]
    hdepth = 1 - hlow / h[r]
    mid = l[bt] + 0.5 * (h[lp] - l[bt])
    if hlow < mid:
        return _nf("손잡이가 컵 하단 절반으로 내려감")
    if hdepth > m["handle_max_depth"]:
        return _nf(f"손잡이 조정 과도 ({hdepth:.0%})")
    lo_n, hi_n = m["normal_depth"]
    deep = bool(depth > hi_n)
    depth_s = 1.0 if lo_n <= depth <= hi_n else 0.5 if deep else 0.7
    third = l[bt] + (h[lp] - l[bt]) / 3
    lower_share = float(np.mean(c[lp:r + 1] <= third))
    u_s = min(1.0, lower_share / 0.25)
    pos = (bt - lp) / cup_len
    sym_s = 1.0 if 0.25 <= pos <= 0.75 else 0.5
    g0, g1 = m["handle_good_depth"]
    hd_s = 1.0 if g0 <= hdepth <= g1 else 0.7 if hdepth < g0 else 0.4
    hc = c[r + 1:]
    drift = float(np.polyfit(np.arange(len(hc)), hc, 1)[0]) if len(hc) >= 3 else 0.0
    hv = float(v[r + 1:].mean()) / float(v[max(0, r - 50):r].mean())
    ha = float(np.nanmean(ap[r + 1:])) / float(np.nanmean(ap[lp:r + 1]))
    plus, minus = [], []
    plus.append(f"컵 깊이 {depth:.0%}, {cup_len // 5}주" + (" (깊은 컵)" if deep else ""))
    (plus if u_s >= 0.8 else minus).append(f"U자형 정도 {u_s:.2f} (바닥권 체류 {lower_share:.0%})")
    (plus if g0 <= hdepth <= g1 else minus).append(f"손잡이 {hdepth:.1%}, {hlen}일, 컵 상단 절반")
    (plus if hv <= 0.85 else minus).append(f"손잡이 거래량 {hv:.2f}배")
    return dict(found=True, pivot=float(h[r]), invalid=float(hlow), invalid_rule=f"종가가 손잡이 저점 {hlow:,.0f} 아래로 마감",
                pattern=(depth_s + u_s + sym_s + hd_s) / 4,
                contraction=(1.0 if hdepth / depth <= 0.35 else 0.6 if hdepth / depth <= 0.5 else 0.3) * 0.7
                + (0.3 if drift <= 0 else 0.15),
                vol_base=_ratio_score(hv, 0.7, 0.85) if hv <= 0.85 else 0.2, atr_ratio=ha,
                info=dict(cup_depth=_num(depth, 3), cup_weeks=cup_len // 5, deep_cup=deep, handle_depth=_num(hdepth, 3),
                          handle_days=hlen, u_score=_num(u_s, 2), handle_vol_ratio=_num(hv, 3), prior_runup=_num(runup, 3)),
                plus=plus, minus=minus)


# ───────────────────────── C. 3C (컵 완성 전 치트) ─────────────────────────
def detect_c3(ctx, m, cfg):
    x = ctx["sx"]
    h, l, v, ap = x["h"], x["l"], x["v"], x["atrp"]
    n = len(h)
    a = max(0, n - m["lookback"])
    lp = a + int(np.argmax(h[a:]))
    since = n - 1 - lp
    if since < m["min_days_since_peak"]:
        return _nf(f"고점 이후 기간 짧음 ({since}일)")
    bt = lp + int(np.argmin(l[lp:]))
    depth = 1 - l[bt] / h[lp]
    if not (m["min_depth"] <= depth <= m["max_depth"]):
        return _nf(f"컵 깊이 범위 밖 ({depth:.0%})")
    if n - 1 - bt < m["min_rally_days"]:
        return _nf(f"바닥 이후 반등 기간 부족 ({n - 1 - bt}일)")
    best = None
    for s in range(m["shelf_max_days"], m["shelf_min_days"] - 1, -1):
        if n - s <= bt:
            continue
        rng = 1 - l[n - s:].min() / h[n - s:].max()
        if rng <= m["shelf_max_range"]:
            best, best_rng = s, rng
            break
    if best is None:
        return _nf("선반 없음 (1~2주, 범위 6% 이내)")
    s0 = n - best
    shi, slo = float(h[s0:].max()), float(l[s0:].min())
    rally = slo / l[bt] - 1
    if rally < m["min_rally"]:
        return _nf(f"우측 상승 미진입 (바닥 대비 {rally:.0%})")
    if h[bt:s0].max() > shi * 1.05:
        return _nf("선반 전 더 높은 반등 고점 (반등 후 되밀림)")
    pos = (slo - l[bt]) / (h[lp] - l[bt])
    if not (m["min_position"] <= pos <= m["max_position"]):
        return _nf(f"선반 위치가 컵 범위 밖 ({pos:.0%})")
    overhead = h[lp] / shi - 1
    if overhead < m["min_overhead"]:
        return _nf("컵 고점에 근접 (3C가 아닌 컵·손잡이 단계)")
    pre = l[max(0, lp - 126):lp]
    runup = h[lp] / pre.min() - 1 if len(pre) >= 20 else None
    if runup is None or runup < m["prior_runup_min"]:
        return _nf("컵 이전 선행 상승 부족")
    label = "로우 치트" if pos < 1 / 3 else "치트" if pos < 2 / 3 else "높은 치트"
    pv = s0 - 20
    sv = float(v[s0:].mean()) / float(v[max(0, pv):s0].mean())
    sa = float(np.nanmean(ap[s0:])) / float(np.nanmean(ap[max(0, pv):s0]))
    tight_s = 1.0 if best_rng <= 0.03 else 0.8 if best_rng <= m["shelf_good_range"] else 0.5
    plus = [f"{label}: 컵 깊이 {depth:.0%}, 선반 {best}일·범위 {best_rng:.1%}", f"컵 고점까지 남은 매물대 {overhead:.0%}"]
    minus = [] if sv <= 0.85 else [f"선반 거래량 감소 미흡 ({sv:.2f}배)"]
    return dict(found=True, pivot=shi, invalid=slo, invalid_rule=f"종가가 선반 저점 {slo:,.0f} 아래로 마감",
                pattern=((1.0 if depth <= 0.35 else 0.6) + (1.0 if pos <= 2 / 3 else 0.7) + min(1.0, rally / 0.15)) / 3,
                contraction=tight_s, vol_base=_ratio_score(sv, 0.7, 0.85) if sv <= 0.85 else 0.2, atr_ratio=sa,
                info=dict(label=label, cup_depth=_num(depth, 3), shelf_days=best, shelf_range=_num(best_rng, 3),
                          position=_num(pos, 2), overhead=_num(overhead, 3), shelf_vol_ratio=_num(sv, 3)),
                plus=plus, minus=minus)


# ───────────────────────── D. 플랫 베이스 ─────────────────────────
def _base_count(x, s0, m):
    """현재 베이스 이전의 '횡보 후 돌파' 횟수로 베이스 순서를 추정 (2단계 시작 = 마지막 200일선 하회 이후)"""
    c, h, l, ma200 = x["c"], x["h"], x["l"], x["ma200"]
    lo = max(25, s0 - m["base_count_window"])
    if s0 <= lo:
        return 1, None
    with np.errstate(invalid="ignore"):
        below = np.where(c[lo:s0] < ma200[lo:s0])[0]
    st = lo + below[-1] + 1 if len(below) else lo
    cnt, last, j = 0, None, max(st, 25)
    while j < s0:
        wh, wl = h[j - 25:j], l[j - 25:j]
        if c[j] > wh.max() and 1 - wl.min() / wh.max() <= 0.25 and int(np.argmax(wh)) <= 15:
            cnt, last = cnt + 1, j
            j += 20
        else:
            j += 1
    return cnt + 1, last


def detect_flat(ctx, m, cfg):
    x = ctx["sx"]
    h, l, c, v, ap, ma50 = x["h"], x["l"], x["c"], x["v"], x["atrp"], x["ma50"]
    n = len(h)
    best = None
    for L_ in range(m["min_days"], min(m["max_days"], n - 60) + 1):
        rng = 1 - l[n - L_:].min() / h[n - L_:].max()
        if rng <= m["max_depth"]:
            best, best_rng = L_, rng
        else:
            break                                   # 창을 넓힐수록 범위는 줄지 않으므로 중단
    if best is None:
        return _nf(f"{m['min_days'] // 5}주 이상·{m['max_depth']:.0%} 이내 횡보 없음")
    s0 = n - best
    hi_i = s0 + int(np.argmax(h[s0:]))
    if n - 1 - hi_i < m["min_days_since_high"]:
        return _nf("베이스 고점이 직전 봉 (이미 돌파 진행 중)")
    piv = float(h[hi_i])
    pre = l[max(0, s0 - 126):s0]
    runup = piv / pre.min() - 1 if len(pre) >= 20 else None
    if runup is None or runup < m["prior_runup_min"]:
        return _nf("선행 상승 부족" + ("" if runup is None else f" ({runup:.0%})"))
    with np.errstate(invalid="ignore"):
        hold = float(np.mean(c[s0:] >= ma50[s0:]))
    ma50_up = np.isfinite(ma50[-1]) and np.isfinite(ma50[s0]) and ma50[-1] > ma50[s0]
    half = s0 + best // 2
    vr = float(v[half:].mean()) / float(v[s0:half].mean())
    ar = float(np.nanmean(ap[half:])) / float(np.nanmean(ap[s0:half]))
    t3 = np.array_split(np.arange(s0, n), 3)
    lows = [float(l[ix].min()) for ix in t3]
    rising = bool(lows[0] <= lows[1] <= lows[2])
    sw = [p for p in _segments(h, l, s0, cfg["swing"]["k"], cfg["swing"]["min_swing"]) if p[1] == "L"]
    inv = float(sw[-1][2]) if sw else float(l[s0:].min())
    tail = 1 - l[-10:].min() / h[-10:].max()
    base_no, last_bo = _base_count(x, s0, m)
    plus, minus = [], []
    plus.append(f"베이스 {best // 5}주, 범위 {best_rng:.1%}, 선행 상승 {runup:.0%}")
    (plus if hold >= m["ma50_hold_ratio"] and ma50_up else minus).append(f"50일선 위 유지 {hold:.0%}"
                                                                       + (", 50일선 상승" if ma50_up else ", 50일선 정체"))
    (plus if rising else minus).append("저점 상승" if rising else "저점 상승 아님")
    if base_no >= 3:
        minus.append(f"{base_no}번째 베이스 추정 (후반 베이스는 실패율 높음)")
    return dict(found=True, pivot=piv, invalid=inv, invalid_rule=f"종가가 베이스 마지막 저점 {inv:,.0f} 아래로 마감",
                pattern=((1.0 if best_rng <= m["good_depth"] else 0.7) + (1.0 if runup >= m["prior_runup_good"] else 0.6)
                         + (1.0 if hold >= m["ma50_hold_ratio"] and ma50_up else 0.4)) / 3,
                contraction=0.5 * (1.0 if rising else 0.5 if lows[2] >= lows[1] else 0.2)
                + 0.5 * (1.0 if tail <= 0.05 else 0.6 if tail <= 0.08 else 0.3),
                vol_base=_ratio_score(vr, 0.7, 0.9), atr_ratio=ar,
                info=dict(base_weeks=best // 5, base_depth=_num(best_rng, 3), prior_runup=_num(runup, 3),
                          base_number=base_no, last_breakout=None if last_bo is None
                          else ctx["df"].index[last_bo].strftime("%Y-%m-%d"), rising_lows=rising,
                          ma50_hold=_num(hold, 2), half_vol_ratio=_num(vr, 3)),
                plus=plus, minus=minus)


# ───────────────────────── E. 파워 플레이 ─────────────────────────
def detect_power(ctx, m, cfg):
    x = ctx["sx"]
    h, l, v, ap = x["h"], x["l"], x["v"], x["atrp"]
    n = len(h)
    a = max(0, n - m["peak_search_days"])
    p = a + int(np.argmax(h[a:]))
    fl = n - 1 - p
    if fl < m["flag_min_days"]:
        return _nf(f"깃발 기간 부족 ({fl}일)")
    if fl > m["flag_max_days"]:
        return _nf(f"깃발 기간 초과 ({fl}일)")
    w0 = max(0, p - m["runup_days"])
    lw = w0 + int(np.argmin(l[w0:p + 1]))
    gain = h[p] / l[lw] - 1
    if gain < m["runup_min"]:
        return _nf(f"8주 내 급등 부족 ({gain:.0%})")
    fdepth = 1 - l[p + 1:].min() / h[p]
    if fdepth > m["flag_max_depth_fail"]:
        return _nf(f"깃발 조정 과도 ({fdepth:.0%})")
    fv = float(v[p + 1:].mean()) / float(v[lw:p + 1].mean())
    fa = float(np.nanmean(ap[p + 1:])) / float(np.nanmean(ap[lw:p + 1]))
    pdays = m["pullback_days"]
    inv = float(l[-pdays:].min())
    tail = 1 - l[-pdays:].min() / h[-pdays:].max()
    plus = [f"{(p - lw) // 5}주 만에 +{gain:.0%} 급등", f"깃발 {fl}일, 조정 {fdepth:.0%}", "RSI 과열 감점 없음 (파워 플레이 예외)"]
    minus = [] if fdepth <= m["flag_max_depth"] else [f"깃발 조정 {fdepth:.0%} > {m['flag_max_depth']:.0%}"]
    return dict(found=True, pivot=float(h[p]), invalid=inv, invalid_rule=f"종가가 깃발 마지막 눌림 저점 {inv:,.0f} 아래로 마감",
                pattern=min(1.0, 0.7 + 0.3 * (gain - m["runup_min"]) / max(1e-9, m["runup_strong"] - m["runup_min"])),
                contraction=0.6 * (1.0 if fdepth <= 0.15 else 0.7 if fdepth <= m["flag_max_depth"] else 0.3)
                + 0.4 * (1.0 if tail <= 0.08 else 0.6 if tail <= 0.12 else 0.3),
                vol_base=_ratio_score(fv, 0.6, 0.85), atr_ratio=fa,
                info=dict(runup=_num(gain, 3), runup_weeks=(p - lw) // 5, flag_days=fl, flag_depth=_num(fdepth, 3),
                          flag_vol_ratio=_num(fv, 3)),
                plus=plus, minus=minus)


DETECTORS = dict(vcp=detect_vcp, cup=detect_cup, c3=detect_c3, flat=detect_flat, power=detect_power)


# ───────────────────────── 항목 점수 ─────────────────────────
def trend_score(ctx, key):
    c, m50, m150, m200 = ctx["close"], ctx["ma50"], ctx["ma150"], ctx["ma200"]
    s50, s150, s200 = ctx["s50"], ctx["s150"], ctx["s200"]
    gt = lambda a, b: a is not None and b is not None and a > b  # noqa: E731
    pos = lambda s: s is not None and s > 0  # noqa: E731
    plus, minus, warns = [], [], []
    if key == "power":
        s = 0.5 * gt(c, m50) + 0.3 * pos(s50) + 0.2 * (m200 is None or c > m200)
        (plus if gt(c, m50) else minus).append("종가 > 50일선" if gt(c, m50) else "종가 < 50일선")
        plus.append("트렌드 템플릿 면제 (급등 직후 모델)")
        if m50 is not None and not pos(s50):
            warns.append("50일선 하락")
        return s, plus, minus, warns
    stacked = gt(c, m50) and gt(m50, m150) and gt(m150, m200)
    up200 = pos(s200)
    if key == "c3":
        s = 0.3 * gt(c, m200) + 0.3 * up200 + 0.2 * gt(m50, m200) + 0.2 * pos(s150)
        if gt(c, m200) and up200:
            plus.append("200일선 위, 200일선 상승")
        elif not gt(c, m200) and not up200:
            warns.append("장기 하락 추세 (200일선 아래·하락)")
        else:
            minus.append("200일선 조건 일부 미충족")
        return s, plus, minus, warns
    s = (0.4 * stacked + 0.2 * up200 + 0.1 * pos(s50) + 0.1 * pos(s150)
         + 0.1 * (c >= ctx["hi52"] * 0.75) + 0.1 * (c >= ctx["lo52"] * 1.3))
    if stacked and up200:
        plus.append("종가 > 50 > 150 > 200일선, 200일선 상승")
    elif gt(c, m200) and up200:
        minus.append("이평선 정배열 미완성 (우선 후보 기준 미충족)")
    else:
        warns.append("장기 추세 미충족 (200일선 아래 또는 하락)")
    return s, plus, minus, warns


def rsi_score(ctx, key, cfg):
    r, ch = ctx["rsi"], ctx["rsi_chg"]
    if r is None:
        return None, None
    rc = cfg["rsi"]
    if key == "power":
        s = 1.0 if r >= rc["power_strong"] else 0.8 if r >= 50 else 0.5 if r >= 40 else 0.2
        return s, f"RSI {r:.0f} (파워 플레이: 높을수록 긍정)"
    lo, hi = rc["pullback_sweet"]
    wlo, _ = rc["weak"]
    if lo <= r <= hi:
        s, note = 1.0, f"RSI {r:.0f} 조정형 적정 구간"
    elif hi < r <= rc["overbought"]:
        s, note = 0.8, f"RSI {r:.0f}"
    elif r > rc["overbought"]:
        s, note = 0.6, f"RSI {r:.0f} 과열"
    elif wlo <= r < lo:
        s, note = (0.5, f"RSI {r:.0f} 약세지만 하락 둔화") if (ch or 0) > 0 else (0.3, f"RSI {r:.0f} 약세")
    else:
        s, note = 0.1, f"RSI {r:.0f} 30 미만"
    if ch is not None and ch > 0 and s < 1:
        s = min(1.0, s + 0.1)
    return s, note


def rs_score(ctx):
    rs = ctx["rs"]
    if rs is None:
        return None
    s = float(np.clip(0.5 + rs["excess"] * 2.5, 0, 1))
    s += 0.15 if rs["slope"] > 0 else 0
    s += 0.15 if rs["new_high"] else 0
    return min(1.0, s)


def atr_score(ratio, cfg):
    if ratio is None or not np.isfinite(ratio):
        return None
    a = cfg["atr"]
    return 1.0 if ratio <= a["shrink_strong"] else 0.6 if ratio <= a["shrink_partial"] else 0.3 if ratio <= 1.1 else 0.0


# ───────────────────────── 모델 평가 ─────────────────────────
def evaluate(ctx, key, cfg):
    """반환: (결과 dict 또는 None, 탈락 사유)"""
    m = cfg["models"][key]
    det = DETECTORS[key](ctx, m, cfg)
    if not det["found"]:
        return None, det["reason"]
    vc, bc = cfg["volume"], cfg["breakout"]
    comps = {}
    comps["trend"], tplus, tminus, warns = trend_score(ctx, key)
    comps["pattern"], comps["contraction"] = det["pattern"], det["contraction"]
    vr = ctx["vol_dry"]
    vr_s = None if vr is None else 1.0 if vr <= vc["dry_strong"] else 0.5 if vr <= vc["dry_partial"] else 0.0
    parts = [p for p in (vr_s, det.get("vol_base")) if p is not None]
    vol = float(np.mean(parts)) if parts else None
    plus, minus = list(det["plus"]) + tplus, list(det["minus"]) + tminus
    if ctx["distribution"] >= cfg["risk_flags"]["distribution_count"]:
        minus.append(f"최근 {cfg['risk_flags']['distribution_days']}일 하락일 대량거래 {ctx['distribution']}회")
        vol = None if vol is None else vol * 0.7
    if vr is not None:
        (plus if vr <= vc["dry_partial"] else minus).append(f"최근 5일/직전 20일 거래량 {vr:.2f}")
    comps["volume"] = vol
    ar = det.get("atr_ratio")
    comps["atr"] = atr_score(ar if ar is not None else ctx["atrp_trend"], cfg)
    comps["rsi"], rnote = rsi_score(ctx, key, cfg)
    if rnote:
        (plus if (comps["rsi"] or 0) >= 0.8 else minus).append(rnote)
    comps["rs"] = rs_score(ctx)
    if ctx["rs"] is not None:
        (plus if ctx["rs"]["excess"] > 0 else minus).append(f"지수 대비 3개월 {ctx['rs']['excess']:+.0%}")
    piv, inv, c = det["pivot"], det["invalid"], ctx["close"]
    dist = c / piv - 1
    brk = ctx["high"] >= piv and (c > piv if bc["require_close_above_pivot"] else True)
    mult = ctx["brk_mult"]
    vol_ok = mult is not None and mult >= vc["breakout_mult"]
    ext = bc["max_extended"]
    comps["pivot"] = (1.0 if brk and vol_ok and dist <= ext else 0.9 if -0.03 <= dist <= ext
                      else 0.6 if -0.08 <= dist < -0.03 else 0.3 if -0.15 <= dist < -0.08 else 0.1)
    mt = cfg["universe"]["min_turnover"]
    liq = 1.0 if ctx["turnover20"] >= 3 * mt else 0.7 if ctx["turnover20"] >= mt else 0.3 if ctx["turnover20"] >= 0.5 * mt else 0.0
    if ctx["flags"]:
        liq *= 0.5
    comps["liquidity"] = liq
    w = m["weights"]
    avail = {k: w[k] for k in COMP_KEYS if comps[k] is not None}
    raw = sum(w[k] * comps[k] for k in avail)
    mx = sum(avail.values())
    norm = raw / mx * 100 if mx else 0.0
    # 위험 신호
    warns += [f["text"] for f in ctx["flags"]]
    lim = cfg["atr"]["excessive_pct_power" if key == "power" else "excessive_pct"]
    if ctx["atrp"] is not None and ctx["atrp"] > lim:
        warns.append(f"변동성 과도 (ATR/종가 {ctx['atrp']:.1%})")
    # 상태
    if c < inv:
        status, detail = WARN, f"무효화: 종가 {c:,.0f} < 무효화 가격 {inv:,.0f}"
    elif brk and dist > ext:
        status, detail = WARN, f"피벗 대비 +{dist:.1%} 확장 돌파 (추격 금지)"
    elif brk and vol_ok:
        status, detail = SIGNAL, f"피벗 {piv:,.0f} 돌파, 거래량 {mult:.1f}배"
    elif brk:
        status, detail = FORM, f"돌파 시도, 거래량 {mult or 0:.1f}배로 기준 미달"
    else:
        status, detail = FORM, f"피벗까지 {-dist:.1%}"
    if warns:
        detail += " | 주의: " + ", ".join(warns)
        status = WARN
    if status in (FORM, SIGNAL) and norm < m["min_score"]:
        return None, f"점수 미달 ({norm:.0f} < {m['min_score']})"
    if status == WARN and norm < m["min_score_warn"]:
        return None, f"주의 + 점수 미달 ({norm:.0f})"
    return dict(model=key, model_name=m["name"], status=status, detail=detail, score=_num(norm, 1), score_raw=_num(raw, 1),
                score_max=_num(mx, 1), components={k: _num(comps[k], 3) for k in COMP_KEYS},
                pivot=_num(piv, 2), invalid=_num(inv, 2), invalid_rule=det["invalid_rule"], dist=_num(dist),
                brk_mult=_num(mult, 2), plus=plus, minus=minus, warnings=warns, info=det["info"],
                signal_age=0 if status == SIGNAL else None,
                signal_date=ctx["date"].strftime("%Y-%m-%d") if status == SIGNAL else None), None


def _nodata(label, n, need):
    return dict(model="data", model_name=label, status=NODATA, detail=f"일봉 {n}개 < 필요 {need}개", score=None,
                pivot=None, invalid=None, plus=[], minus=[], warnings=[], info={})


def _recent_signals(ctx, bench, cfg, out, keys):
    """최근 N일 안에 돌파 신호가 있었는지 확인 (그날까지의 데이터만으로 재판정)"""
    R, vc, ext = cfg["breakout"]["recent_signal_days"], cfg["volume"], cfg["breakout"]["max_extended"]
    v, h, c, T, dc = ctx["v"], ctx["h"], ctx["c"], ctx["T"], ctx["df"]
    cur = {e["model"]: e for e in out["models"]}
    done = set(k for k, e in cur.items() if e["status"] == SIGNAL)
    for k in range(1, R + 1):
        j = T - k
        if j < 60 or not (v[j] >= vc["breakout_mult"] * v[j - 50:j].mean() and c[j] > h[j - 5:j].max()):
            continue
        ck = build_context(dc.iloc[:j + 1], bench, cfg)
        if "T" not in ck:
            continue
        for key in keys:
            if key in done:
                continue
            e, _ = evaluate(ck, key, cfg)
            if e is None or e["status"] != SIGNAL:
                continue
            p, q, cc = e["pivot"], e["invalid"], ctx["close"]
            dist = cc / p - 1
            if cc < q:
                st, det = WARN, f"{k}일 전 돌파 후 무효화 가격 이탈"
            elif dist > ext:
                st, det = WARN, f"{k}일 전 돌파, 피벗 대비 +{dist:.1%} 확장 (추격 금지)"
            elif cc >= p:
                st, det = SIGNAL, f"{k}일 전 돌파 (거래량 {e['brk_mult']:.1f}배), 매수 구간 유지"
            else:
                st, det = WARN, f"{k}일 전 돌파 후 피벗 아래로 복귀"
            e.update(status=st, detail=det, dist=_num(dist), signal_age=k, signal_date=ck["date"].strftime("%Y-%m-%d"))
            out["models"] = [x for x in out["models"] if x["model"] != key] + [e]
            out["rejections"].pop(key, None)
            done.add(key)


def analyze_stock(d, bench=None, cfg=None, meta=None, recent=True):
    """한 종목의 일봉(기준일까지)으로 다섯 모델을 각각 판정"""
    cfg = cfg or load_config()
    ctx = build_context(d, bench, cfg)
    out = dict(meta or {})
    out.update(common_fields(ctx, cfg))
    out["models"], out["rejections"] = [], {}
    dc = cfg["data"]
    if ctx["n"] < dc["min_bars_power"]:
        out["models"].append(_nodata("전체 모델", ctx["n"], dc["min_bars_power"]))
        return out
    keys = MODEL_KEYS if ctx["n"] >= dc["min_bars"] else ["power"]
    if ctx["n"] < dc["min_bars"]:
        out["models"].append(_nodata("VCP·컵·3C·플랫", ctx["n"], dc["min_bars"]))
    for k in keys:
        e, why = evaluate(ctx, k, cfg)
        if e is None:
            out["rejections"][k] = why
        else:
            out["models"].append(e)
    if recent:
        _recent_signals(ctx, bench, cfg, out, keys)
    out["models"].sort(key=lambda e: (STATUS_ORDER[e["status"]], -(e["score"] or 0)))
    out["best_status"] = out["models"][0]["status"] if out["models"] else None
    out["best_score"] = max([e["score"] or 0 for e in out["models"]], default=None)
    return out
