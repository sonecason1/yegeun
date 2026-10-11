# -*- coding: utf-8 -*-
"""VCP(변동성 수축 패턴) 탐지 — 미너비니 스크리너에서 분리한 모듈"""
import numpy as np
import pandas as pd

CFG = {'min_marcap': 50000000000.0, 'min_turnover': 3000000000.0, 'min_bars': 230, 'rs_min': 70, 'base_lookback': 130, 'base_min_len': 15, 'swing_k': 3, 'min_swing': 0.03, 'max_first': 0.4, 'max_last': 0.12, 'near_pivot': 0.06, 'max_extended': 0.05, 'vol_dry': 0.85}

def swings(h, l, k, min_swing):
    """스윙 고점·저점을 찾아 고점-저점이 번갈아 나오도록 정리"""
    n = len(h)
    pts = []
    for i in range(k, n - k):
        if h[i] == max(h[i - k:i + k + 1]):
            pts.append((i, "H", h[i]))
        if l[i] == min(l[i - k:i + k + 1]):
            pts.append((i, "L", l[i]))
    pts.sort(key=lambda x: (x[0], 0 if x[1] == "H" else 1))
    merged = []
    for p in pts:
        if merged and merged[-1][1] == p[1]:
            if (p[1] == "H" and p[2] >= merged[-1][2]) or (p[1] == "L" and p[2] <= merged[-1][2]):
                merged[-1] = p
            continue
        merged.append(p)
    # 너무 작은 흔들림 제거
    cleaned = []
    for p in merged:
        if cleaned and cleaned[-1][1] == "H" and p[1] == "L" and (1 - p[2] / cleaned[-1][2]) < min_swing:
            continue
        if cleaned and cleaned[-1][1] == p[1]:
            if (p[1] == "H" and p[2] >= cleaned[-1][2]) or (p[1] == "L" and p[2] <= cleaned[-1][2]):
                cleaned[-1] = p
            continue
        cleaned.append(p)
    return cleaned


def detect_vcp(d):
    """
    베이스 고점(최근 6개월 최고가) 이후의 수축(고점→저점 하락폭)을 차례로 측정.
    수축이 점점 작아지고, 마지막 수축이 얕고, 거래량이 마르면 VCP 후보.
    """
    lb = CFG["base_lookback"]
    w = d.iloc[-lb:]
    h, l, c, v = w["High"].values, w["Low"].values, w["Close"].values, w["Volume"].values
    n = len(w)
    base_i = int(np.argmax(h))
    base_len = n - 1 - base_i
    res = dict(VCP=False, 상태="", 수축=[], 피벗=np.nan, 손절가=np.nan, 베이스일수=base_len)
    if base_len < CFG["base_min_len"]:
        res["상태"] = "신고가 부근/베이스 형성 초기(고점 후 15일 미만)"
        return res

    seg_h, seg_l = h[base_i:], l[base_i:]
    sw = swings(seg_h, seg_l, CFG["swing_k"], CFG["min_swing"])
    if not sw or sw[0][1] != "H":
        sw = [(0, "H", seg_h[0])] + [p for p in sw if p[0] > 0]
        # 다시 번갈아 나오도록 정리
        tmp = []
        for p in sw:
            if tmp and tmp[-1][1] == p[1]:
                if (p[1] == "H" and p[2] >= tmp[-1][2]) or (p[1] == "L" and p[2] <= tmp[-1][2]):
                    tmp[-1] = p
                continue
            tmp.append(p)
        sw = tmp

    # 마지막 확정 스윙 이후 꼬리 구간 처리 (아직 확정되지 않은 최근 고점·저점)
    last_i = sw[-1][0]
    tail_h, tail_l = seg_h[last_i + 1:], seg_l[last_i + 1:]
    if len(tail_h):
        if sw[-1][1] == "H":
            j = int(np.argmin(tail_l))
            if 1 - tail_l[j] / sw[-1][2] >= CFG["min_swing"] / 2:
                sw.append((last_i + 1 + j, "L", tail_l[j]))
        else:
            j = int(np.argmax(tail_h))
            sw.append((last_i + 1 + j, "H", tail_h[j]))

    pairs = []
    for a, b in zip(sw, sw[1:]):
        if a[1] == "H" and b[1] == "L":
            pairs.append((a, b, 1 - b[2] / a[2]))
    if len(pairs) < 2:
        res["상태"] = "수축 1회 이하"
        res["수축"] = [p[2] for p in pairs]
        return res

    # 오른쪽(최근) 끝에서부터 '점점 작아지는' 수축만 이어서 VCP로 인정
    #   예) 35% → 42% → 20% → 9% → 4%  ⇒  42% → 20% → 9% → 4%
    tol = 1.15
    start = len(pairs) - 1
    while start > 0 and pairs[start][2] <= pairs[start - 1][2] * tol:
        start -= 1
    pairs = pairs[start:]
    depths = [p[2] for p in pairs]
    res["수축"] = depths
    if len(pairs) < 2:
        res["상태"] = "최근 수축이 작아지지 않음"
        return res

    # 피벗 = 마지막 수축이 시작된 고점 / 손절가 = 마지막 수축의 저점
    last_low = pairs[-1][1]
    pivot = pairs[-1][0][2]
    res["피벗"] = pivot
    res["손절가"] = last_low[2]

    shrinking = depths[-1] < depths[0] * 0.7
    first_ok = depths[0] <= CFG["max_first"]
    last_ok = depths[-1] <= CFG["max_last"]

    vol10, vol50 = v[-10:].mean(), d["Volume"].iloc[-50:].mean()
    res["거래량수축비"] = vol10 / vol50 if vol50 else np.nan
    vol_ok = res["거래량수축비"] < CFG["vol_dry"]

    last_close = c[-1]
    dist = last_close / pivot - 1
    res["피벗거리"] = dist
    res["손절폭"] = 1 - last_low[2] / max(last_close, pivot)

    # 오늘 돌파 여부 (종가가 피벗 위 + 거래량 50일 평균의 1.4배 이상)
    vol_today = v[-1] / vol50 if vol50 else 0
    res["당일거래량배수"] = vol_today
    res["비고"] = "손절폭 8% 초과: 비중 축소 또는 패스" if res["손절폭"] > 0.08 else ""

    if not (shrinking and first_ok and last_ok):
        why = []
        if not first_ok: why.append(f"첫 수축 {depths[0]*100:.0f}%>40%")
        if not last_ok: why.append(f"마지막 수축 {depths[-1]*100:.0f}%>12%")
        if not shrinking: why.append("수축 감소 폭 부족")
        res["상태"] = "수축 패턴 불충분(" + ", ".join(why) + ")"
        return res
    if dist > CFG["max_extended"]:
        res["상태"] = "피벗 대비 과도 상승(추격 금지)"
        return res
    if dist > 0:
        res["상태"] = "돌파" + ("(거래량 동반)" if vol_today >= 1.4 else "(거래량 부족·확인 필요)")
        res["VCP"] = True
        return res
    if dist >= -CFG["near_pivot"]:
        res["상태"] = "돌파 임박" + ("" if vol_ok else "(거래량 덜 마름)")
        res["VCP"] = True
        return res
    res["상태"] = "피벗과 거리 있음"
    return res
