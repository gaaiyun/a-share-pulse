"""一次性：用新浪逐只日线（原始价 + 后复权）生成近一年的每日个股快照 data/spot/日期.csv.gz，
给持仓成本、均线、距高点回撤这些需要个股历史的指标当起点；之后每天由 collect.py 追加当天截面。

    python scripts/bootstrap_spot.py --start 20251001 --end 20261104 [--limit 30]

新浪历史接口没有 PE/PB/总市值：PE、PB 留空，总市值用“最新总市值/最新流通市值”的比例从流通市值折算。
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'spot'


def fetch(code, start, end):
    sym = ('sh' if code.startswith('6') else 'sz') + code
    for k in range(4):
        try:
            raw = ak.stock_zh_a_daily(symbol=sym, start_date=start, end_date=end, adjust='')
            hfq = ak.stock_zh_a_daily(symbol=sym, start_date=start, end_date=end, adjust='hfq')
            if raw.empty or hfq.empty:
                return None
            x = raw.merge(hfq[['date', 'close']].rename(columns={'close': 'hc'}), on='date')
            x['code'] = code
            return x
        except Exception:  # noqa: BLE001  网络抖动重试
            time.sleep(1.5 * (k + 1))
    return None


def limit_pct(code, st):
    if code.startswith(('300', '301', '302', '688', '689')):
        return 0.20
    return 0.05 if st else 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='20251001')
    ap.add_argument('--end', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    spot = pd.read_csv(ROOT / 'data' / 'raw' / 'spot_latest.csv', dtype={'code': str})
    spot = spot[spot['code'].str.match(r'^(0|3|6)')]
    codes = sorted(spot['code'])
    if a.limit:
        codes = codes[::max(len(codes) // a.limit, 1)][:a.limit]
    t0, rows, miss = time.time(), [], []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, c, a.start, a.end): c for c in codes}
        for i, f in enumerate(as_completed(futs)):
            df = f.result()
            if df is None or df.empty:
                miss.append(futs[f])
            else:
                rows.append(df)
            if i % 200 == 0:
                print(f'{i}/{len(codes)} {time.time() - t0:.0f}s 缺 {len(miss)}', flush=True)
    print('取完', len(rows), '缺', len(miss), miss[:20], f'{time.time() - t0:.0f}s', flush=True)
    g = pd.concat(rows, ignore_index=True)
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount', 'outstanding_share', 'hc']:
        g[c] = pd.to_numeric(g[c], errors='coerce')
    g['date'] = pd.to_datetime(g['date']).dt.strftime('%Y-%m-%d')
    g = g.sort_values(['code', 'date'])
    # 除权日的昨收要用复权比例还原：昨收 = 今收 / (后复权今收 / 后复权昨收)
    g['pre_close'] = (g['close'] / (g['hc'] / g.groupby('code')['hc'].shift(1))).round(3)
    g = g[g['pre_close'].notna() & (g['volume'] > 0)].copy()
    meta = spot.set_index('code')
    g['st'] = g['code'].map(meta['st'].astype(bool)).fillna(False).astype(int)
    g['fmv'] = (g['close'] * g['outstanding_share'] / 1e8).round(2)
    lim = np.array([limit_pct(c, s) for c, s in zip(g['code'], g['st'])])
    hl = np.floor(g['pre_close'] * (1 + lim) * 100 + 0.5 + 1e-9) / 100
    ll = np.floor(g['pre_close'] * (1 - lim) * 100 + 0.5 + 1e-9) / 100
    g['zt'] = ((g['close'] - hl).abs() < 0.0051).astype(int)
    g['dt'] = ((g['close'] - ll).abs() < 0.0051).astype(int)
    g['zb'] = (((g['high'] - hl).abs() < 0.0051) & (g['zt'] == 0)).astype(int)
    g['cant'] = ((g['open'] - hl).abs() < 0.0051).astype(int)
    last = g.sort_values('date').groupby('code').tail(1).set_index('code')
    ratio = (meta['mv_wan'] / 1e4 / last['fmv']).clip(lower=1, upper=20).reindex(g['code']).fillna(1).values
    g['mv'] = (g['fmv'] * ratio).round(2)
    g['pe'], g['pb'] = np.nan, np.nan
    cols = ['code', 'st', 'open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'fmv', 'mv', 'pe', 'pb', 'zt', 'dt', 'zb', 'cant']
    newest = g[g['date'] == g['date'].max()]
    print(newest[cols].head(5).to_string())
    chk = newest.merge(spot[['code', 'close', 'amount', 'ret']], on='code', suffixes=('', '_s'))
    print('与最新截面核对', chk['date'].iloc[0] if len(chk) else None, len(chk),
          '收盘差>0.01:', int(((chk['close'] - chk['close_s']).abs() > 0.011).sum()),
          '成交额相对差>1%:', int((((chk['amount'] / chk['amount_s']) - 1).abs() > 0.01).sum()),
          '涨跌幅差>0.05点:', int((((chk['close'] / chk['pre_close'] - 1) * 100 - chk['ret']).abs() > 0.05).sum()), flush=True)
    if a.limit:
        print('试跑，不写文件')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for d, x in g.groupby('date'):
        x[cols].to_csv(OUT / f'{d}.csv.gz', index=False, encoding='utf-8', compression={'method': 'gzip', 'mtime': 0})
    print('写入', g['date'].nunique(), '天', round(sum(p.stat().st_size for p in OUT.glob('*.gz')) / 1e6, 1), 'MB', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
