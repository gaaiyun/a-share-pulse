"""收盘后采集：只用在 GitHub Actions（海外机房）能稳定取到的公开接口。

    python scripts/collect.py              # 采当天（非交易日直接退出 0）
    python scripts/collect.py --date 20261030

东方财富 push2 行情接口在海外会被限流，这里一律不用：
全市场截面走新浪，涨停/跌停/炸板池走东财 push2ex，估值与宽度走乐咕乐股，两融走东财数据中心。
每个来源单独 try，某个挂了只影响对应模块，build 时按各自日期显示新旧。
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
from akshare.stock.cons import zh_sina_a_stock_count_url, zh_sina_a_stock_payload, zh_sina_a_stock_url
from akshare.utils import http, jsonish

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
RAW = DATA / 'raw'
LOG = []

INDICES = {  # 新浪代码 -> 名称
    'sh000001': '上证指数', 'sz399001': '深证成指', 'sz399006': '创业板指', 'sh000688': '科创50',
    'sh000300': '沪深300', 'sh000905': '中证500', 'sh000852': '中证1000', 'sh000985': '中证全指',
}
# 上交所宽基 ETF（份额变化常被看作“国家队”进出的线索）；深市 ETF 份额接口在海外取不到
ETF_WATCH = {
    '510300': '沪深300', '510310': '沪深300', '510330': '沪深300', '510050': '上证50',
    '510500': '中证500', '512100': '中证1000', '588000': '科创50', '588080': '科创50', '563300': '中证2000',
}
METRIC_COLS = ['date', 'n', 'up', 'down', 'amount_yi', 'med_ret', 'up5', 'down5', 'zt', 'dt', 'zb',
               'max_streak', 'n_streak2', 'zt_prem', 'promo_1to2', 'micro_ret', 'src']


def step(name):
    def deco(fn):
        def run(*a, **k):
            t = time.time()
            try:
                r = fn(*a, **k)
                LOG.append({'step': name, 'ok': True, 'sec': round(time.time() - t, 1)})
                print(f'[OK] {name} {time.time() - t:.1f}s', flush=True)
                return r
            except Exception as e:  # noqa: BLE001  单个来源失败不拖垮整体
                LOG.append({'step': name, 'ok': False, 'err': f'{type(e).__name__}: {e}'[:300]})
                print(f'[FAIL] {name}: {e}\n{traceback.format_exc()[-800:]}', flush=True)
                return None
        return run
    return deco


def retry(fn, n=3, wait=5):
    for i in range(n):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == n - 1:
                raise
            time.sleep(wait * (i + 1))


def append_rows(path, df, key):
    """按 key 去重后追加；同 key 以新数据为准。"""
    if path.exists():
        old = pd.read_csv(path, dtype={'code': str})
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=key, keep='last').sort_values(key)
    df.to_csv(path, index=False, encoding='utf-8')
    return df


# ---------------------------------------------------------------- 交易日
def trade_days():
    cal = retry(ak.tool_trade_date_hist_sina)
    return sorted(pd.to_datetime(cal['trade_date']).dt.date)


# ---------------------------------------------------------------- 全市场截面（新浪，保留市值/PE/PB）
def sina_spot():
    res = http.get(zh_sina_a_stock_count_url)
    pages = int(np.ceil(int(re.findall(r'\d+', res.text)[0]) / 80))
    frames = []
    for p in range(1, pages + 1):
        payload = dict(zh_sina_a_stock_payload, page=p)
        r = retry(lambda: http.get(zh_sina_a_stock_url, params=payload))
        frames.append(pd.DataFrame(jsonish.loads(r.text)))
        time.sleep(0.25)
    df = pd.concat(frames, ignore_index=True)
    keep = {'symbol': 'sym', 'code': 'code', 'name': 'name', 'trade': 'close', 'settlement': 'pre_close',
            'open': 'open', 'high': 'high', 'low': 'low', 'volume': 'volume', 'amount': 'amount',
            'changepercent': 'pct', 'per': 'pe', 'pb': 'pb', 'mktcap': 'mv_wan', 'nmc': 'fmv_wan',
            'turnoverratio': 'turnover', 'ticktime': 'tick'}
    df = df[list(keep)].rename(columns=keep)
    for c in ['close', 'pre_close', 'open', 'high', 'low', 'volume', 'amount', 'pct', 'pe', 'pb', 'mv_wan', 'fmv_wan',
              'turnover']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df


def limit_pct(row):
    c, name = row['code'], str(row['name'])
    if c.startswith(('300', '301', '302', '688', '689')):
        return 0.20
    if 'ST' in name.upper():
        return 0.05
    return 0.10


def round_half_up(x):
    return np.floor(x * 100 + 0.5 + 1e-9) / 100


def spot_metrics(spot, mv_rank_n=400):
    """与 seed_from_db.py 同口径：沪深两市、当日有成交。"""
    s = spot[spot['sym'].str[:2].isin(['sh', 'sz']) & (spot['volume'] > 0) & (spot['pre_close'] > 0)].copy()
    s['ret'] = s['close'] / s['pre_close'] - 1
    lim = s.apply(limit_pct, axis=1)
    s['hl'] = round_half_up(s['pre_close'] * (1 + lim))
    s['ll'] = round_half_up(s['pre_close'] * (1 - lim))
    s['zt'] = (s['close'] - s['hl']).abs() < 0.0051
    s['dt'] = (s['close'] - s['ll']).abs() < 0.0051
    s['zb'] = ((s['high'] - s['hl']).abs() < 0.0051) & ~s['zt']
    s['st'] = s['name'].str.upper().str.contains('ST') | s['name'].str.contains('退')
    s['mv_prev'] = s['mv_wan'] / (1 + s['ret'])
    micro = s[~s['st'] & s['mv_prev'].notna()].nsmallest(mv_rank_n, 'mv_prev')
    s['cant'] = (s['open'] - s['hl']).abs() < 0.0051
    m = {
        'n': len(s), 'up': int((s['ret'] > 0).sum()), 'down': int((s['ret'] < 0).sum()),
        'amount_yi': round(s['amount'].sum() / 1e8, 1), 'med_ret': round(float(s['ret'].median()) * 100, 3),
        'up5': int((s['ret'] >= 0.05).sum()), 'down5': int((s['ret'] <= -0.05).sum()),
        'zt': int(s['zt'].sum()), 'dt': int(s['dt'].sum()), 'zb': int(s['zb'].sum()),
        'micro_ret': round(float(micro['ret'].mean()) * 100, 3),
    }
    return m, s


def save_spot(d, s):
    """当天截面存成 data/spot/日期.csv.gz（约 150KB），market.py 只读这些文件，不再请求网络。"""
    x = pd.DataFrame({'code': s['code'], 'st': s['st'].astype(int), 'open': s['open'], 'high': s['high'], 'low': s['low'],
                      'close': s['close'], 'pre_close': s['pre_close'], 'volume': s['volume'], 'amount': s['amount'],
                      'fmv': (s['fmv_wan'] / 1e4).round(2), 'mv': (s['mv_wan'] / 1e4).round(2), 'pe': s['pe'], 'pb': s['pb'],
                      'zt': s['zt'].astype(int), 'dt': s['dt'].astype(int), 'zb': s['zb'].astype(int), 'cant': s['cant'].astype(int)})
    (DATA / 'spot').mkdir(exist_ok=True)
    x.to_csv(DATA / 'spot' / f'{d.isoformat()}.csv.gz', index=False, encoding='utf-8', compression={'method': 'gzip', 'mtime': 0})


# ---------------------------------------------------------------- 涨停/炸板/跌停池（东财 push2ex）
def pools(d):
    D = d.strftime('%Y%m%d')
    zt = retry(lambda: ak.stock_zt_pool_em(date=D))
    zb = retry(lambda: ak.stock_zt_pool_zbgc_em(date=D))
    dt = retry(lambda: ak.stock_zt_pool_dtgc_em(date=D))
    prev = retry(lambda: ak.stock_zt_pool_previous_em(date=D))
    return zt, zb, dt, prev


def pool_rows(d, zt, zb, dt):
    ds = d.isoformat()
    rows = []
    for _, r in zt.iterrows():
        rows.append({'date': ds, 'code': str(r['代码']).zfill(6), 'name': r['名称'], 'kind': 'zt', 'ind': r['所属行业'],
                     'streak': int(r['连板数']), 'stat': r['涨停统计'], 'first': str(r['首次封板时间']).zfill(6),
                     'last': str(r['最后封板时间']).zfill(6), 'opens': int(r['炸板次数']),
                     'seal_yi': round(float(r['封板资金']) / 1e8, 3), 'amount_yi': round(float(r['成交额']) / 1e8, 3),
                     'fmv_yi': round(float(r['流通市值']) / 1e8, 2), 'pct': round(float(r['涨跌幅']), 2)})
    for _, r in zb.iterrows():
        rows.append({'date': ds, 'code': str(r['代码']).zfill(6), 'name': r['名称'], 'kind': 'zb', 'ind': r['所属行业'],
                     'streak': None, 'stat': r['涨停统计'], 'first': str(r['首次封板时间']).zfill(6), 'last': None,
                     'opens': int(r['炸板次数']), 'seal_yi': None, 'amount_yi': round(float(r['成交额']) / 1e8, 3),
                     'fmv_yi': round(float(r['流通市值']) / 1e8, 2), 'pct': round(float(r['涨跌幅']), 2)})
    for _, r in dt.iterrows():
        rows.append({'date': ds, 'code': str(r['代码']).zfill(6), 'name': r['名称'], 'kind': 'dt', 'ind': r['所属行业'],
                     'streak': int(r['连续跌停']), 'stat': None, 'first': None, 'last': str(r['最后封板时间']).zfill(6),
                     'opens': int(r['开板次数']), 'seal_yi': round(float(r['封单资金']) / 1e8, 3),
                     'amount_yi': round(float(r['成交额']) / 1e8, 3), 'fmv_yi': round(float(r['流通市值']) / 1e8, 2),
                     'pct': round(float(r['涨跌幅']), 2)})
    return pd.DataFrame(rows)


def pool_metrics(zt, prev):
    """连板高度、昨日涨停今日溢价、首板晋级率（东财口径：连板数按个股连续涨停交易日计）。"""
    m = {'max_streak': int(zt['连板数'].max()) if len(zt) else 0,
         'n_streak2': int((zt['连板数'] >= 2).sum()) if len(zt) else 0}
    if len(prev):
        m['zt_prem'] = round(float(prev['涨跌幅'].mean()), 3)
        first = prev[prev['昨日连板数'] == 1]
        today = set(zt['代码'].astype(str))
        if len(first):
            m['promo_1to2'] = round(float(first['代码'].astype(str).isin(today).mean()) * 100, 1)
    return m


@step('涨停池回填')
def backfill_pools(days):
    """首次运行或断档时，把最近若干交易日的涨停池补齐（东财只保留近一个月左右）。"""
    path = DATA / 'pool_daily.csv'
    have = set(pd.read_csv(path, usecols=['date'])['date']) if path.exists() else set()
    rows, mets = [], []
    for d in days:
        if d.isoformat() in have:
            continue
        try:
            zt, zb, dt, prev = pools(d)
        except Exception as e:  # noqa: BLE001
            print('  回填跳过', d, e)
            continue
        if len(zt) == 0 and len(dt) == 0:
            continue
        rows.append(pool_rows(d, zt, zb, dt))
        mets.append({'date': d.isoformat(), 'zt': len(zt), 'zb': len(zb), 'dt': len(dt), **pool_metrics(zt, prev),
                     'src': 'pool'})
        time.sleep(0.6)
    if rows:
        append_rows(path, pd.concat(rows, ignore_index=True), ['date', 'kind', 'code'])
    return mets


# ---------------------------------------------------------------- 全历史类（每次整表覆盖）
@step('指数日线')
def indices(d):
    frames = []
    for sym, name in INDICES.items():
        df = retry(lambda: ak.stock_zh_index_daily(symbol=sym))
        # 盘中接口会带当天未收盘的一行，只留到数据日
        dt = pd.to_datetime(df['date'])
        df = df[(dt >= '2014-01-01') & (dt <= pd.Timestamp(d))][['date', 'close', 'volume']]
        df['sym'], df['name'] = sym, name
        frames.append(df)
        time.sleep(0.3)
    pd.concat(frames).to_csv(RAW / 'index_daily.csv', index=False, encoding='utf-8')


@step('乐咕估值与宽度')
def legu():
    srcs = {
        'pe_median': ak.stock_a_ttm_lyr, 'pb_median': ak.stock_a_all_pb, 'ebs': ak.stock_ebs_lg,
        'buffett': ak.stock_buffett_index_lg, 'congestion': ak.stock_a_congestion_lg,
        'highlow': lambda: ak.stock_a_high_low_statistics(symbol='all'),
        'below_net': lambda: ak.stock_a_below_net_asset_statistics(symbol='全部A股'),
    }
    bad = []
    for k, fn in srcs.items():
        try:
            retry(fn).to_csv(RAW / f'{k}.csv', index=False, encoding='utf-8')
        except Exception as e:  # noqa: BLE001
            bad.append(f'{k}: {e}')
        time.sleep(0.5)
    if bad:
        raise RuntimeError('; '.join(bad))


@step('两融账户')
def margin():
    retry(ak.stock_margin_account_info).to_csv(RAW / 'margin_account.csv', index=False, encoding='utf-8')


@step('国债收益率')
def bonds():
    df = retry(lambda: ak.bond_zh_us_rate(start_date='20150101'))
    df.to_csv(RAW / 'bond_yield.csv', index=False, encoding='utf-8')


# ---------------------------------------------------------------- 当日类（追加）
@step('行业与概念板块')
def boards(d):
    out = []
    for kind, ind in (('sina_ind', '新浪行业'), ('concept', '概念')):
        df = retry(lambda: ak.stock_sector_spot(indicator=ind))
        out.append(pd.DataFrame({'date': d.isoformat(), 'kind': kind, 'name': df['板块'],
                                 'n': pd.to_numeric(df['公司家数'], errors='coerce'),
                                 'pct': pd.to_numeric(df['涨跌幅'], errors='coerce').round(3),
                                 'amount_yi': (pd.to_numeric(df['总成交额'], errors='coerce') / 1e8).round(2),
                                 'leader': df['股票名称'],
                                 'leader_pct': pd.to_numeric(df['个股-涨跌幅'], errors='coerce').round(2)}))
    try:
        t = retry(lambda: ak.stock_board_industry_summary_ths())
        out.append(pd.DataFrame({'date': d.isoformat(), 'kind': 'ths_ind', 'name': t['板块'],
                                 'n': pd.to_numeric(t['上涨家数'], errors='coerce') + pd.to_numeric(t['下跌家数'], errors='coerce'),
                                 'pct': pd.to_numeric(t['涨跌幅'], errors='coerce'),
                                 'amount_yi': pd.to_numeric(t['总成交额'], errors='coerce'),
                                 'leader': t['领涨股'], 'leader_pct': pd.to_numeric(t['领涨股-涨跌幅'], errors='coerce'),
                                 'net_in_yi': pd.to_numeric(t['净流入'], errors='coerce'),
                                 'up': pd.to_numeric(t['上涨家数'], errors='coerce')}))
    except Exception as e:  # noqa: BLE001
        print('  同花顺行业失败', e)
    append_rows(DATA / 'board_daily.csv', pd.concat(out, ignore_index=True), ['date', 'kind', 'name'])


@step('龙虎榜')
def lhb(d):
    D = d.strftime('%Y%m%d')
    det = retry(lambda: ak.stock_lhb_detail_em(start_date=D, end_date=D))
    det = det.drop_duplicates('代码')
    a = pd.DataFrame({'date': d.isoformat(), 'code': det['代码'].astype(str).str.zfill(6), 'name': det['名称'],
                      'pct': pd.to_numeric(det['涨跌幅'], errors='coerce').round(2),
                      'net_yi': (pd.to_numeric(det['龙虎榜净买额'], errors='coerce') / 1e8).round(3),
                      'net_ratio': pd.to_numeric(det['净买额占总成交比'], errors='coerce').round(2),
                      'reason': det['上榜原因'], 'note': det['解读']})
    append_rows(DATA / 'lhb_daily.csv', a, ['date', 'code'])
    y = retry(lambda: ak.stock_lhb_hyyyb_em(start_date=D, end_date=D))
    b = pd.DataFrame({'date': d.isoformat(), 'seat': y['营业部名称'],
                      'buy_n': y['买入个股数'], 'sell_n': y['卖出个股数'],
                      'net_yi': (pd.to_numeric(y['总买卖净额'], errors='coerce') / 1e8).round(3),
                      'stocks': y['买入股票']})
    append_rows(DATA / 'lhb_seat_daily.csv', b, ['date', 'seat'])


@step('上交所宽基ETF份额')
def etf_shares(days):
    path = DATA / 'etf_shares.csv'
    have = set(pd.read_csv(path, usecols=['date'])['date']) if path.exists() else set()
    rows = []
    for d in days:
        if d.isoformat() in have:
            continue
        try:
            df = retry(lambda: ak.fund_etf_scale_sse(date=d.strftime('%Y%m%d')), n=2)
        except Exception as e:  # noqa: BLE001
            # 上交所 T+1 才发布，当晚取到空表是常态，只有网络错误才值得打印
            if not isinstance(e, KeyError):
                print('  ETF 份额跳过', d, str(e)[:120])
            continue
        df = df[df['基金代码'].astype(str).isin(ETF_WATCH)]
        for _, r in df.iterrows():
            rows.append({'date': d.isoformat(), 'code': str(r['基金代码']), 'name': r['基金简称'],
                         'track': ETF_WATCH[str(r['基金代码'])], 'shares_yi': round(float(r['基金份额']) / 1e8, 4)})
        time.sleep(0.4)
    if rows:
        append_rows(path, pd.DataFrame(rows), ['date', 'code'])
    # 份额乘最新价折成金额；不同 ETF 单位净值差很多，份额不能直接相加
    px = retry(lambda: ak.fund_etf_category_sina(symbol='ETF基金'))
    px['code'] = px['代码'].astype(str).str[-6:]
    px = px[px['code'].isin(ETF_WATCH)][['code', '名称', '最新价']].rename(columns={'名称': 'name', '最新价': 'price'})
    px.to_csv(RAW / 'etf_price.csv', index=False, encoding='utf-8')


@step('赚钱效应(乐咕)')
def activity(d):
    df = retry(ak.stock_market_activity_legu)
    kv = dict(zip(df['item'], df['value']))
    stamp = str(kv.get('统计日期', ''))[:10]
    if stamp != d.isoformat():
        print(f'  乐咕统计日期 {stamp}，不是 {d}，这次不记')
        return
    row = {'date': stamp, **{k: kv[k] for k in kv if k != '统计日期'}}
    append_rows(DATA / 'activity_daily.csv', pd.DataFrame([row]), ['date'])


def report_period(day):
    """最近一个已结束的报告期。"""
    y, m = day.year, day.month
    return f'{y - 1}1231' if m <= 3 else f'{y}0331' if m <= 6 else f'{y}0630' if m <= 9 else f'{y}0930'


def in_season(day):
    # 4、8、10 月是财报集中披露期，每天取；其他时候一周一次就够
    return day.month in (1, 4, 8, 10) or (day.month == 11 and day.day <= 5)


@step('财报与股东户数')
def earnings(d):
    """按申万一级汇总，只存聚合数。"""
    imap = pd.read_csv(DATA / 'industry_sw1.csv', dtype={'code': str})
    period = report_period(d)
    sp = RAW / 'earnings_stamp.json'
    stamp = json.loads(sp.read_text(encoding='utf-8')) if sp.exists() else {}
    last = datetime.strptime(stamp['at'], '%Y-%m-%d').date() if stamp.get('at') else date(2000, 1, 1)
    if stamp.get('period') == period and (d - last).days < (1 if in_season(d) else 7):
        return
    t = retry(lambda: ak.stock_yjbb_em(date=period))
    t = pd.DataFrame({'code': t['股票代码'].astype(str).str.zfill(6), 'yoy': pd.to_numeric(t['净利润-同比增长'], errors='coerce'),
                      'rev': pd.to_numeric(t['营业总收入-同比增长'], errors='coerce'), 'np': pd.to_numeric(t['净利润-净利润'], errors='coerce'),
                      'ann': t['最新公告日期'].astype(str).str[:10]}).merge(imap, on='code', how='left')
    t['sw1'] = t['sw1'].fillna('未归类')
    rows = []
    for name, g in list(t.groupby('sw1')) + [('全部A股', t)]:
        y = g['yoy'].dropna()
        rows.append({'period': period, 'sw1': name, 'n': len(g), 'med_yoy': round(y.median(), 2) if len(y) else None,
                     'pos': round((y > 0).mean() * 100, 1) if len(y) else None, 'dbl': round((y >= 50).mean() * 100, 1) if len(y) else None,
                     'crash': round((y <= -50).mean() * 100, 1) if len(y) else None, 'loss': round((g['np'] < 0).mean() * 100, 1),
                     'med_rev': round(g['rev'].median(), 2), 'last_ann': g['ann'].max()})
    append_rows(DATA / 'earnings_sector.csv', pd.DataFrame(rows), ['period', 'sw1'])
    try:
        h = retry(lambda: ak.stock_zh_a_gdhs(symbol=period), n=2)
        h = pd.DataFrame({'code': h['代码'].astype(str).str.zfill(6), 'chg': pd.to_numeric(h['股东户数-增减比例'], errors='coerce'),
                          'ret': pd.to_numeric(h['区间涨跌幅'], errors='coerce'), 'to': h['股东户数统计截止日-本次'].astype(str).str[:10]})
        h = h[h['to'] >= f'{period[:4]}-{period[4:6]}-01'].merge(imap, on='code', how='left')
        h['sw1'] = h['sw1'].fillna('未归类')
        rows = [{'period': period, 'sw1': n, 'n': len(g), 'med_chg': round(g['chg'].median(), 2), 'up': round((g['chg'] > 0).mean() * 100, 1),
                 'med_ret': round(g['ret'].median(), 2)} for n, g in list(h.groupby('sw1')) + [('全部A股', h)] if len(g)]
        append_rows(DATA / 'holders_sector.csv', pd.DataFrame(rows), ['period', 'sw1'])
    except Exception as e:  # noqa: BLE001
        print('  股东户数失败', str(e)[:160])
    sp.write_text(json.dumps({'period': period, 'at': d.isoformat()}), encoding='utf-8')


def _unlock_frame(a):
    return pd.DataFrame({'date': a['解禁时间'].astype(str).str[:10], 'n': pd.to_numeric(a['当日解禁股票家数'], errors='coerce'),
                         'mv_yi': (pd.to_numeric(a['实际解禁市值'], errors='coerce') / 1e8).round(2)})


@step('解禁')
def unlock(d):
    start, end = d + timedelta(days=1), d + timedelta(days=92)
    a = retry(lambda: ak.stock_restricted_release_summary_em(symbol='全部股票', start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d')))
    _unlock_frame(a).to_csv(RAW / 'unlock_future.csv', index=False, encoding='utf-8')
    # 过去 3 年每天的解禁市值，用来判断接下来算不算高峰；已有的不重复取
    hp = DATA / 'unlock_hist.csv'
    h0 = pd.read_csv(hp) if hp.exists() else pd.DataFrame(columns=['date', 'n', 'mv_yi'])
    since = (pd.to_datetime(h0['date'].max()).date() - timedelta(days=5)) if len(h0) else d - timedelta(days=1100)
    h = retry(lambda: ak.stock_restricted_release_summary_em(symbol='全部股票', start_date=since.strftime('%Y%m%d'), end_date=d.strftime('%Y%m%d')))
    append_rows(hp, _unlock_frame(h), ['date'])
    b = retry(lambda: ak.stock_restricted_release_detail_em(start_date=start.strftime('%Y%m%d'), end_date=(d + timedelta(days=62)).strftime('%Y%m%d')))
    imap = pd.read_csv(DATA / 'industry_sw1.csv', dtype={'code': str})
    b = pd.DataFrame({'code': b['股票代码'].astype(str).str.zfill(6), 'mv_yi': pd.to_numeric(b['实际解禁市值'], errors='coerce') / 1e8,
                      'ratio': pd.to_numeric(b['占解禁前流通市值比例'], errors='coerce')}).merge(imap, on='code', how='left')
    b['sw1'] = b['sw1'].fillna('未归类')
    g = b.groupby('sw1').agg(mv_yi=('mv_yi', 'sum'), n=('code', 'nunique'), big=('ratio', lambda x: int((x >= .05).sum()))).reset_index().sort_values('mv_yi', ascending=False)
    g['mv_yi'] = g['mv_yi'].round(1)
    g.to_csv(RAW / 'unlock_sector.csv', index=False, encoding='utf-8')


@step('新股与股息率')
def ipo_div():
    x = retry(lambda: ak.stock_xgsglb_em(symbol='全部股票'))
    y = pd.DataFrame({'list': x['上市日期'].astype(str).str[:10], 'sub': x['申购日期'].astype(str).str[:10],
                      'raise_yi': (pd.to_numeric(x['发行总数'], errors='coerce') * pd.to_numeric(x['发行价格'], errors='coerce') / 1e4).round(2),
                      'price': pd.to_numeric(x['发行价格'], errors='coerce'), 'first': pd.to_numeric(x['首日收盘价'], errors='coerce'),
                      'board': x['板块'], 'prize': pd.to_numeric(x['每中一签获利'], errors='coerce'), 'rate': pd.to_numeric(x['中签率'], errors='coerce')})
    y = y[y['sub'] >= '2023-01-01'].sort_values('sub')
    y.to_csv(RAW / 'ipo.csv', index=False, encoding='utf-8')
    retry(lambda: ak.stock_a_gxl_lg(symbol='上证A股')).to_csv(RAW / 'dividend.csv', index=False, encoding='utf-8')


@step('全市场聚合')
def market_step():
    import market
    market.update()


# ---------------------------------------------------------------- 主流程
def live_date(now, cal):
    """此刻的新浪截面对应哪个交易日：开盘前还是上一交易日收盘，盘中返回 None，收盘后是当天。
    GitHub 定时任务可能晚好几个小时，不能拿“今天”当数据日。"""
    today = now.date()
    past = [x for x in cal if x <= today]
    if past[-1] == today:
        hm = now.hour * 100 + now.minute
        if hm < 915:
            return past[-2]
        if hm < 1530:
            return None
    return past[-1]


def skip(msg):
    print(msg)
    # 不动 last_collect.json，页面底部的数据健康状况保持上次的；告诉工作流跳过构建和提交
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
            f.write('skipped=true\n')
    return 0


def already_done(d):
    """这个交易日已经完整采过（用于开盘前、非交易日的补跑：有缺项才重跑）。"""
    p = DATA / 'last_collect.json'
    if not p.exists():
        return False
    c = json.loads(p.read_text(encoding='utf-8'))
    return c.get('date') == d.isoformat() and c.get('ok') == c.get('total')


def write_status(d, cal, wave, merge=False):
    next_day = next((x for x in cal if x > d), None)
    steps = LOG
    p = DATA / 'last_collect.json'
    if merge and p.exists():  # 早间轮次只更新自己跑过的来源，其余保留昨晚的状态
        names = {x['step'] for x in LOG}
        steps = [x for x in json.loads(p.read_text(encoding='utf-8')).get('steps', []) if x['step'] not in names] + LOG
    now = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=8)
    p.write_text(json.dumps({'date': d.isoformat(), 'at': now.strftime('%Y-%m-%d %H:%M'), 'wave': wave,
                             'next': next_day.isoformat() if next_day else None,
                             'ok': sum(x['ok'] for x in steps), 'total': len(steps), 'steps': steps}, ensure_ascii=False, indent=1), encoding='utf-8')


def morning(d, cal):
    """开盘前：两融和 ETF 份额都是第二天早上才发布，只补这些，截面和涨停池不动。"""
    days = [x for x in cal if x <= d]
    margin()
    legu()
    bonds()
    etf_shares(days[-60:])
    write_status(d, cal, 'morning', merge=True)
    print('早间更新完成')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='YYYYMMDD，默认按当前时刻判断数据对应的交易日（北京时间）')
    ap.add_argument('--force', action='store_true', help='非交易日也跑全历史类来源')
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=8)
    cal = trade_days()
    live = live_date(now, cal)
    if a.date:
        d = datetime.strptime(a.date, '%Y%m%d').date()
    elif live is None:
        return skip(f'{now:%H:%M} 盘中，截面不是收盘数据，跳过')
    else:
        d = live
        if d != now.date() and already_done(d):
            return morning(d, cal)
    days = [x for x in cal if x <= d]
    if not (days and days[-1] == d) and not a.force:
        return skip(f'{d} 不是交易日，跳过')
    d = days[-1]
    recent = days[-25:]
    # 只有截面确实是 d 这天的收盘才能用；指定历史日期补数时只补按日期查询的来源（涨停池、龙虎榜等）
    spot_ok = d == live

    # 1) 全历史类
    indices(d)
    legu()
    margin()
    bonds()

    # 2) 当日情绪行（新浪截面 + 东财涨停池）
    mpath = DATA / 'daily_metrics.csv'
    metrics = pd.read_csv(mpath) if mpath.exists() else pd.DataFrame(columns=METRIC_COLS)
    # 断档的交易日先用涨停池补情绪字段（宽度字段补不了，留空）
    gap = [x for x in recent if x < d and x.isoformat() not in set(metrics['date'])]
    # 东财炸板池只留约 30 个交易日，回填窗口再往前取会白白重试
    filled = backfill_pools(sorted(set(gap) | set(days[-12:]))) or []
    for m in filled:
        if m['date'] in set(metrics['date']) or m['date'] == d.isoformat():
            continue
        metrics = pd.concat([metrics, pd.DataFrame([m])], ignore_index=True)

    row = {'date': d.isoformat(), 'src': 'live'}
    if not spot_ok:  # 补历史日期：保留这一行已有的截面字段，只更新涨停池相关字段
        old = metrics[metrics['date'] == row['date']]
        if len(old):
            row = {**old.iloc[0].dropna().to_dict(), **row}
    spot = step('新浪全市场截面')(sina_spot)() if spot_ok else None
    if spot is not None and len(spot) > 4000:
        sm, s = spot_metrics(spot)
        row.update(sm)
        save_spot(d, s)
        keep = s[['code', 'name', 'close', 'ret', 'amount', 'mv_wan', 'pe', 'pb', 'turnover', 'zt', 'dt', 'st']].copy()
        keep['ret'] = (keep['ret'] * 100).round(3)
        keep.to_csv(RAW / 'spot_latest.csv', index=False, encoding='utf-8')
    pz = step('当日涨停池')(pools)(d)
    if pz is not None:
        zt, zb, dt, prev = pz
        append_rows(DATA / 'pool_daily.csv', pool_rows(d, zt, zb, dt), ['date', 'kind', 'code'])
        row.update(pool_metrics(zt, prev))
        row['zt_pool'], row['zb_pool'], row['dt_pool'] = len(zt), len(zb), len(dt)
    if 'up' in row or 'max_streak' in row:
        metrics = pd.concat([metrics[metrics['date'] != row['date']], pd.DataFrame([row])], ignore_index=True)
    metrics = metrics.sort_values('date')
    metrics.to_csv(mpath, index=False, encoding='utf-8')

    # 3) 其他当日类
    if spot_ok:  # 板块榜也是实时截面
        boards(d)
    lhb(d)
    etf_shares(days[-60:])
    activity(d)
    earnings(d)
    unlock(d)
    ipo_div()
    market_step()

    write_status(d, cal, 'close')
    print(f"完成 {sum(x['ok'] for x in LOG)}/{len(LOG)}")
    # 截面和涨停池都挂了才算失败，其余来源失败只降级
    core = [x for x in LOG if x['step'] in ('新浪全市场截面', '当日涨停池')]
    return 0 if any(x['ok'] for x in core) else 1


if __name__ == '__main__':
    sys.exit(main())
