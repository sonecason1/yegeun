# -*- coding: utf-8 -*-
"""
미너비니 스크리너 (TradingView 데이터 기반) → 웹사이트용 JSON 생성
=================================================================
1단계  TradingView 스크리너(비공식 파이썬 패키지 tradingview-screener)로 국내 전 종목을 한 번에 조회
       - 이동평균(50·150·200일), 52주 고가·저가, 기간 수익률, 시가총액, 거래량, 실적 성장률
       - RS 백분위 계산, 트렌드 템플릿 7개 조건 + 유동성 필터
2단계  1단계 통과 종목만 일봉 이력(FinanceDataReader)을 받아
       - 조건 3(200일선 1개월 이상 상승) 확인, VCP 탐지, 차트용 데이터 저장
3단계  docs/data/ 아래에 latest.json, 날짜별 기록, 종목별 차트 JSON, TradingView 관심목록 txt 저장

실행:  python screener/run_tv.py            (저장소 최상위 폴더에서)
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from tradingview_screener import Query, col

sys.path.insert(0, os.path.dirname(__file__))
from vcp import CFG, detect_vcp  # noqa: E402

KST = timezone(timedelta(hours=9))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "data")

SETTINGS = dict(
    min_marcap=500e8,      # 시가총액 500억 이상
    min_turnover=30e8,     # 30일 평균 거래대금 30억 이상
    rs_min=70,             # RS 70 이상
    chart_bars=260,        # 차트에 저장할 일봉 수 (약 1년)
    keep_history=60,       # 날짜별 기록 보관 개수
)

TV_FIELDS = [
    "name", "description", "close", "change", "SMA50", "SMA150", "SMA200",
    "price_52_week_high", "price_52_week_low", "Perf.3M", "Perf.6M", "Perf.Y",
    "market_cap_basic", "average_volume_30d_calc", "sector", "industry",
    "earnings_per_share_diluted_yoy_growth_fq", "total_revenue_yoy_growth_fq",
    "operating_margin", "earnings_release_next_date",
]


# ───────────────────────── 1단계: TradingView ─────────────────────────
def fetch_tv():
    for attempt in range(3):
        try:
            n, df = (Query().set_markets("korea").select(*TV_FIELDS)
                     .where(col("type") == "stock", col("subtype") == "common",
                            col("market_cap_basic") >= SETTINGS["min_marcap"])
                     .set_property("options", {"lang": "ko"})          # 종목명을 한글로
                     .limit(5000).get_scanner_data())
            return df
        except Exception as e:
            print("TradingView 조회 실패, 재시도:", e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("TradingView 스크리너 조회 실패")


def clean_universe(df):
    df = df.copy()
    df["code"] = df["name"].astype(str)
    df["description"] = df["description"].str.replace(r"\s*보통주$", "", regex=True).str.strip()
    df = df[df["code"].str.fullmatch(r"\d{5}0")]                 # 일반 보통주(우선주 코드 제외)
    df = df[~df["description"].str.contains("SPAC|스팩|REIT|리츠", case=False, na=False)]
    return df.reset_index(drop=True)


def rs_percentile(df):
    # IBD 방식 근사: 최근 3개월 40%, 6개월 30%, 1년 30% 가중 (TradingView에 9개월 수익률이 없어 단순화)
    raw = 0.4 * df["Perf.3M"].fillna(0) + 0.3 * df["Perf.6M"].fillna(0) + 0.3 * df["Perf.Y"].fillna(0)
    return (raw.rank(pct=True) * 98 + 1).round().astype(int)


def stage1(df):
    c = df["close"]
    df["turnover"] = df["average_volume_30d_calc"] * c
    df["RS"] = rs_percentile(df)
    conds = pd.DataFrame({
        "c1": (c > df["SMA150"]) & (c > df["SMA200"]),
        "c2": df["SMA150"] > df["SMA200"],
        "c4": (df["SMA50"] > df["SMA150"]) & (df["SMA50"] > df["SMA200"]),
        "c5": c > df["SMA50"],
        "c6": c >= df["price_52_week_low"] * 1.30,
        "c7": c >= df["price_52_week_high"] * 0.75,
        "c8": df["RS"] >= SETTINGS["rs_min"],
    })
    df = pd.concat([df, conds], axis=1)
    df["liq_ok"] = df["turnover"] >= SETTINGS["min_turnover"]
    df["pass7"] = conds.all(axis=1) & df["liq_ok"]
    return df


# ───────────────────────── 실적 등급 ─────────────────────────
def fund_grade(eps, rev):
    """최근 분기 전년 동기 대비: A 매출+20%·EPS+25% / B 매출+10%·EPS+10% / C 그 외 / N/A 데이터 없음"""
    if pd.isna(eps) and pd.isna(rev):
        return "N/A"
    e = -999 if pd.isna(eps) else eps
    r = -999 if pd.isna(rev) else rev
    if r >= 20 and e >= 25:
        return "A"
    if r >= 10 and e >= 10:
        return "B"
    return "C"


# ───────────────────────── 2단계: 일봉 이력 ─────────────────────────
def fetch_hist(code, start):
    for attempt in range(3):
        try:
            d = fdr.DataReader(code, start)
            d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
            return code, d[d["Volume"] > 0]
        except Exception:
            time.sleep(1 + attempt)
    return code, None


def ma200_up_days(close):
    ma = close.rolling(200).mean()
    up = (ma > ma.shift(21)).iloc[::-1]
    return int(up.cumprod().sum()) if len(up) and up.iloc[0] else 0


def chart_payload(d):
    d = d.copy()
    for p in (50, 150, 200):
        d[f"ma{p}"] = d["Close"].rolling(p).mean()
    d = d.iloc[-SETTINGS["chart_bars"]:]
    r = lambda v: None if pd.isna(v) else round(float(v), 2)
    return [[t.strftime("%Y-%m-%d"), r(o), r(h), r(l), r(c), int(v), r(m1), r(m2), r(m3)]
            for t, o, h, l, c, v, m1, m2, m3 in
            zip(d.index, d.Open, d.High, d.Low, d.Close, d.Volume, d.ma50, d.ma150, d.ma200)]


def market_condition():
    """지수 대신 지수 ETF로 판단 (종목과 같은 데이터 출처라 날짜가 어긋나지 않음)"""
    out = []
    start = (datetime.now(KST) - timedelta(days=500)).strftime("%Y-%m-%d")
    for sym, name in [("069500", "코스피200 (KODEX 200)"), ("229200", "코스닥150 (KODEX 코스닥150)")]:
        try:
            c = fdr.DataReader(sym, start)["Close"].dropna()
            m50, m200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
            last = c.iloc[-1]
            state = "상승 추세" if last > m50 > m200 else "하락 추세" if last < m50 < m200 else "혼조"
            out.append(dict(name=name, close=round(float(last), 2), ma50=round(float(m50), 2),
                            ma200=round(float(m200), 2), from_high=round(float(last / c.iloc[-252:].max() - 1), 4),
                            state=state, date=c.index[-1].strftime("%Y-%m-%d")))
        except Exception as e:
            out.append(dict(name=name, state=f"조회 실패: {e}"))
    return out


def num(v, nd=4):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), nd)
    return v


# ───────────────────────── 메인 ─────────────────────────
def main():
    t0 = time.time()
    now = datetime.now(KST)
    tv = stage1(clean_universe(fetch_tv()))
    cand = tv[tv["pass7"]].copy()
    print(f"TradingView 조회 {len(tv)}종목 → 1단계 통과 {len(cand)}종목")

    start = (now - timedelta(days=420)).strftime("%Y-%m-%d")
    with ThreadPoolExecutor(max_workers=8) as ex:
        hist = dict(ex.map(lambda c: fetch_hist(c, start), cand["code"].tolist()))

    os.makedirs(os.path.join(OUT, "charts"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "history"), exist_ok=True)
    rows, last_dates = [], []
    for _, r in cand.iterrows():
        d = hist.get(r["code"])
        if d is None or len(d) < CFG["min_bars"]:
            continue
        last_dates.append(d.index[-1])
        up_days = ma200_up_days(d["Close"])
        if up_days < 1:                                    # 조건 3 미충족
            continue
        v = detect_vcp(d)
        depths = v.pop("수축", [])
        grade = fund_grade(r["earnings_per_share_diluted_yoy_growth_fq"], r["total_revenue_yoy_growth_fq"])
        rows.append(dict(
            code=r["code"], name=r["description"], tv=f"KRX:{r['code']}",
            sector=r["sector"], industry=r["industry"],
            close=num(r["close"], 2), change=num(r["change"], 2), rs=int(r["RS"]),
            from_high=num(r["close"] / r["price_52_week_high"] - 1),
            from_low=num(r["close"] / r["price_52_week_low"] - 1),
            ma200_up_days=up_days, marcap_eok=num(r["market_cap_basic"] / 1e8, 0),
            turnover_eok=num(r["turnover"] / 1e8, 1),
            eps_yoy=num(r["earnings_per_share_diluted_yoy_growth_fq"], 1),
            rev_yoy=num(r["total_revenue_yoy_growth_fq"], 1),
            op_margin=num(r["operating_margin"], 1), grade=grade,
            next_earnings=(datetime.fromtimestamp(r["earnings_release_next_date"], KST).strftime("%Y-%m-%d")
                           if pd.notna(r["earnings_release_next_date"]) else None),
            vcp=bool(v.get("VCP")), status=v.get("상태", ""),
            contractions=" → ".join(f"{x*100:.0f}%" for x in depths), n_contr=len(depths),
            pivot=num(v.get("피벗"), 2), stop=num(v.get("손절가"), 2),
            pivot_dist=num(v.get("피벗거리")), stop_pct=num(v.get("손절폭")),
            base_days=num(v.get("베이스일수")), vol_dry=num(v.get("거래량수축비"), 2),
            vol_today=num(v.get("당일거래량배수"), 2), note=v.get("비고", ""),
        ))
        with open(os.path.join(OUT, "charts", f"{r['code']}.json"), "w", encoding="utf-8") as f:
            json.dump(chart_payload(d), f, separators=(",", ":"))

    asof = max(last_dates).strftime("%Y-%m-%d") if last_dates else now.strftime("%Y-%m-%d")
    order = {"돌파(거래량 동반)": 0}
    gorder = {"A": 0, "B": 1, "C": 2, "N/A": 3}
    rows.sort(key=lambda x: (not x["vcp"], order.get(x["status"], 1), gorder[x["grade"]], -x["rs"]))
    result = dict(
        asof=asof, generated_at=now.strftime("%Y-%m-%d %H:%M KST"), source="TradingView screener + FinanceDataReader",
        settings=SETTINGS, market=market_condition(),
        counts=dict(universe=int(len(tv)), stage1=int(len(cand)), trend=len(rows), vcp=sum(x["vcp"] for x in rows)),
        stocks=rows,
    )
    with open(os.path.join(OUT, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "history", f"{asof.replace('-', '')}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))

    # 날짜별 기록 목록 정리
    hist_dir = os.path.join(OUT, "history")
    files = sorted(x for x in os.listdir(hist_dir) if x.endswith(".json") and x != "index.json")
    for old in files[:-SETTINGS["keep_history"]]:
        os.remove(os.path.join(hist_dir, old))
    files = files[-SETTINGS["keep_history"]:]
    with open(os.path.join(hist_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump([x[:-5] for x in files], f)

    # 오늘 목록에 없는 종목의 차트 파일 삭제
    keep = {x["code"] for x in rows}
    for fn in os.listdir(os.path.join(OUT, "charts")):
        if fn[:-5] not in keep:
            os.remove(os.path.join(OUT, "charts", fn))

    # TradingView 관심목록 가져오기용 txt (섹션 구분 ###)
    vcp_syms = [x["tv"] for x in rows if x["vcp"]]
    other = [x["tv"] for x in rows if not x["vcp"]]
    with open(os.path.join(OUT, "watchlist_tradingview.txt"), "w", encoding="utf-8") as f:
        f.write(",".join((["###VCP후보"] + vcp_syms if vcp_syms else []) + ["###트렌드템플릿"] + other))

    print(f"기준일 {asof}: 트렌드 템플릿 {len(rows)}개, VCP 후보 {result['counts']['vcp']}개 "
          f"({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
