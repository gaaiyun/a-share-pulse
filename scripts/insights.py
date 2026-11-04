"""把聚合表算成给人看的判断：打法、持仓盈亏、风格、行业、谚语回测、解禁、财报、新股。

每一块单独容错，缺数据只影响自己。凡是“历史上类似情况之后”的统计，分档都只用当时已知的数据（滚动或扩张分位）。
"""
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from market import FACTORS, PLAYS, SIZE_NAMES, VALUE_NAMES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
RAW = DATA / 'raw'


def rd(x, n=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, n)


def lst(s, n=2):
    return [rd(x, n) for x in s]


def pct_of(s, v):
    a = pd.Series(s).dropna().values
    if len(a) == 0 or v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return rd(((a < v).sum() + 0.5 * (a == v).sum()) / len(a) * 100, 0)


def thin(n, tail=750, step=5):
    """较早的日子每 5 天取一个，最近 tail 天全取，图可以缩放到全历史，文件又不太大。"""
    head = list(range(0, max(n - tail, 0), step))
    return head + list(range(max(n - tail, 0), n))


def fwd(close, h):
    return close.shift(-h) / close - 1


def stat(x):
    x = pd.Series(x).dropna()
    if len(x) < 8:
        return None
    return {'n': int(len(x)), 'med': rd(x.median() * 100), 'win': rd((x > 0).mean() * 100, 0)}


def compound(r, n):
    """日收益（%）连乘成 n 日收益（%）。"""
    return (np.exp(np.log1p(r.fillna(0) / 100).rolling(n, min_periods=max(n // 2, 1)).sum()) - 1) * 100


def expanding_pct(s, m=250):
    return s.expanding(m).rank(pct=True) * 100


# ================================================================ 打法
def plays(md):
    rows = []
    for k, (name, desc) in PLAYS.items():
        s = md['p_' + k]
        roll = s.rolling(20, min_periods=10).mean()
        dm = s - s.mean()
        a = dm.rolling(20).mean()
        x = pd.concat([a, a.shift(-20)], axis=1).dropna().iloc[::20]
        rows.append({'key': k, 'name': name, 'def': desc, 'last': rd(s.iloc[-1]), 'q': rd(md['q_' + k].iloc[-1]),
                     'd5': rd(s.tail(5).mean()), 'd20': rd(s.tail(20).mean()), 'sum20': rd(s.tail(20).sum(), 1),
                     'win20': rd((s.tail(20).dropna() > 0).mean() * 100, 0), 'avg': rd(s.mean()), 'win': rd((s.dropna() > 0).mean() * 100, 0),
                     'p20': pct_of(roll, roll.iloc[-1]), 'n': rd(md['n_' + k].iloc[-1], 0),
                     'keep': rd(x.corr(method='spearman').iloc[0, 1])})
    rows.sort(key=lambda r: -(r['d20'] if r['d20'] is not None else -99))
    tail = md.tail(20)
    return {'rows': rows, 'days': list(tail.index), 'heat': {k: lst(tail['p_' + k]) for k in PLAYS}, 'date': md.index[-1], 'start': md.index[0]}


def funnel(md):
    out = []
    for k, lab in ((1, '首板→2板'), (2, '2板→3板'), (3, '3板→4板'), (4, '4板及以上→再板')):
        p, n = md[f'fp{k}'], md[f'fn{k}']
        # 家数加权：小样本的日子不该和 50 只的日子同权
        wavg = lambda a, b: rd((a * b).sum() / b[a.notna()].sum(), 1) if b[a.notna()].sum() else None
        out.append({'label': lab, 'today': rd(p.iloc[-1], 0), 'n': rd(n.iloc[-1], 0), 'm20': wavg(p.tail(20), n.tail(20)),
                    'y1': wavg(p.tail(250), n.tail(250)), 'all': wavg(p, n)})
    return out


# ================================================================ 持仓体感
def feel(md, c1000, extra):
    i = thin(len(md))
    pick = md.iloc[i]
    last = md.iloc[-1]
    cards = {k: {'v': rd(last[k], 1), 'p': pct_of(md[k], last[k])}
             for k in ('pl20', 'pl60', 'pl250', 'plw60', 'ma20', 'ma60', 'ma250', 'dd250', 'dd50', 'ytd_med', 'ytd_win', 'open_win', 'fade')}
    # 近 60 日买入者浮盈落在历史哪一档，之后中证1000 怎么走（分档只用当时已知数据）
    known = expanding_pct(md['pl60'])
    f5, f20 = fwd(c1000, 5).reindex(md.index), fwd(c1000, 20).reindex(md.index)
    edges = [(0, 10, '最低 10%'), (10, 30, '10–30%'), (30, 70, '30–70%'), (70, 90, '70–90%'), (90, 100.01, '最高 10%')]
    cur = known.iloc[-1]
    rows = []
    for a, b, lab in edges:
        m = (known >= a) & (known < b)
        rows.append({'label': lab, 'k5': stat(f5[m]), 'k20': stat(f20[m]), 'now': bool(a <= cur < b)})
    return {'cards': cards, 'buckets': rows, 'all': {'k5': stat(f5), 'k20': stat(f20)}, 'date': md.index[-1],
            'series': {'d': list(pick.index), **{k: lst(pick[k], 1) for k in ('pl20', 'pl60', 'pl250', 'ma20', 'ma60', 'ma250', 'dd250', 'open_win', 'fade')}},
            'extra': extra}


# ================================================================ 风格
def grid(md):
    out = {}
    cols = {f'g{s}{v}': (s, v) for s in range(6) for v in range(4)}
    for n, tag in ((1, 'd1'), (5, 'd5'), (20, 'd20')):
        cell = [[None] * 4 for _ in range(6)]
        for c, (s, v) in cols.items():
            x = md[c].tail(n).dropna()
            if len(x) >= max(n * .6, 1):
                cell[s][v] = rd((np.prod(1 + x / 100) - 1) * 100, 1)
        out[tag] = cell
    return {'sizes': SIZE_NAMES, 'values': VALUE_NAMES, **out}


def factors(md):
    rows = []
    i = thin(len(md))
    curves = {}
    for k, name in FACTORS.items():
        s = md['f_' + k]
        roll = s.rolling(20, min_periods=10).sum()
        ytd = s[s.index >= md.index[-1][:4] + '-01-01'].sum()
        rows.append({'key': k, 'name': name, 'd1': rd(s.iloc[-1]), 'd5': rd(s.tail(5).sum(), 1), 'd20': rd(s.tail(20).sum(), 1), 'd60': rd(s.tail(60).sum(), 1),
                     'ytd': rd(ytd, 1), 'p20': pct_of(roll, roll.iloc[-1]), 'ann': rd(s.mean() * 245, 0)})
        curves[k] = lst(s.fillna(0).cumsum().iloc[i], 1)
    return {'rows': rows, 'd': list(md.index[i]), 'curves': curves}


# ================================================================ 行业
def sectors(sd, md):
    sd = sd[sd['sw1'] != '未归类'].copy()
    ret = sd.pivot_table(index='date', columns='sw1', values='ret').sort_index()
    pb = sd.pivot_table(index='date', columns='sw1', values='pb').sort_index()
    pe = sd.pivot_table(index='date', columns='sw1', values='pe').sort_index()
    amt = sd.pivot_table(index='date', columns='sw1', values='amt').sort_index()
    mkt = md['cap_ret'].reindex(ret.index)
    r5, r20, r60 = (ret.apply(compound, n=n) for n in (5, 20, 60))
    m5, m20, m60 = (compound(mkt, n) for n in (5, 20, 60))
    last = sd[sd['date'] == sd['date'].max()].set_index('sw1')
    d = ret.index[-1]
    rows = []
    for n in ret.columns:
        if n not in last.index:
            continue
        a3 = amt[n].dropna().iloc[-750:]
        b10 = pb[n].dropna().iloc[-2430:]
        e10 = pe[n].dropna().iloc[-2430:]
        L = last.loc[n]
        rows.append({'name': n, 'r1': rd(L['ret']), 'eq': rd(L['eq']), 'up': rd(L['up'], 0), 'zt': rd(L['zt'], 0), 'dt': rd(L['dt'], 0),
                     'r5': rd(r5.at[d, n], 1), 'r20': rd(r20.at[d, n], 1), 'r60': rd(r60.at[d, n], 1),
                     'rs20': rd(r20.at[d, n] - m20[d], 1), 'rs60': rd(r60.at[d, n] - m60[d], 1),
                     'amt': rd(L['amt'], 1), 'amt_p': pct_of(a3, L['amt']), 'pb': rd(L['pb']), 'pb_p': pct_of(b10, L['pb']),
                     'pe': rd(L['pe'], 1), 'pe_p': pct_of(e10, L['pe'])})
    rows.sort(key=lambda r: -(r['rs20'] if r['rs20'] is not None else -999))
    recent = ret.tail(15)
    order = [r['name'] for r in rows]
    # 四象限的历史表现：只用当时已知的 PB 分位和 20 日相对强弱，看之后 20 日相对全市场的超额
    known_pb = pb.ffill().expanding(500).rank(pct=True) * 100
    rs = r20.sub(m20, axis=0)
    f20 = r20.shift(-20).sub(m20.shift(-20), axis=0)
    idx = ret.index[::5]
    quad = {}
    for lab, cheap, strong in (('便宜且走强', True, True), ('便宜但走弱', True, False), ('贵且走强', False, True), ('贵且走弱', False, False)):
        m = (known_pb < 50 if cheap else known_pb >= 50) & (rs > 0 if strong else rs <= 0)
        x = f20.where(m).loc[idx].stack()
        quad[lab] = {'n': int(len(x)), 'med': rd(x.median()), 'win': rd((x > 0).mean() * 100, 0)}
    # 成交拥挤：成交占比处在自己近三年前 10% 之后，20 日相对收益如何
    known_amt = amt.rolling(750, min_periods=250).rank(pct=True) * 100
    x_hot = f20.where(known_amt >= 90).loc[idx].stack()
    x_rest = f20.where(known_amt < 90).loc[idx].stack()
    # 轮动速度：本周行业排名和上周的相关，越低轮动越快
    r5s = r5.iloc[-260:]
    speed = [r5s.iloc[k].corr(r5s.iloc[k - 5], method='spearman') for k in range(5, len(r5s))]
    return {'date': d, 'rows': rows, 'days': list(recent.index), 'heat': {n: lst(recent[n], 1) for n in order}, 'quad': quad,
            'hot': {'n': int(len(x_hot)), 'med': rd(x_hot.median()), 'win': rd((x_hot > 0).mean() * 100, 0),
                    'rest_med': rd(x_rest.median()), 'rest_win': rd((x_rest > 0).mean() * 100, 0)},
            'speed': {'now': rd(speed[-1]), 'p': pct_of(speed, speed[-1])}}


# ================================================================ 这些说法灵不灵
def cooldown(mask, gap=10):
    out = np.zeros(len(mask), dtype=bool)
    last = -10 ** 9
    for i, v in enumerate(mask):
        if v and i - last >= gap:
            out[i], last = True, i
    return out


def evaluate(mask, f, rng, n_perm=2000):
    valid = ~np.isnan(f)
    x = f[mask & valid]
    pool = f[valid]
    if len(x) < 8:
        return {'n': int(len(x))}
    obs = np.median(x)
    base = np.median(pool)
    sims = np.median(rng.choice(pool, size=(n_perm, len(x))), axis=1)
    p = float(((np.abs(sims - sims.mean()) >= abs(obs - sims.mean())).mean()))
    return {'n': int(len(x)), 'med': rd(obs * 100), 'win': rd((x > 0).mean() * 100, 0), 'base': rd(base * 100), 'base_win': rd((pool > 0).mean() * 100, 0), 'p': rd(p, 3)}


def sayings(m, md, piv, ma, bn):
    """每条说法都写成当时就能判断的规则；同一波行情只算第一天（间隔不足 10 个交易日不重复计）。"""
    idx = m.index
    c1 = piv['中证1000'].reindex(idx).ffill()
    sh = piv['上证指数'].reindex(idx).ffill()
    hs = piv['沪深300'].reindex(idx).ffill()
    amt = m['amount_yi'].astype(float)
    r = sh.pct_change()
    down5 = (r < 0).rolling(5).sum() == 5
    up5 = (r > 0).rolling(5).sum() == 5
    q = lambda s, a: s.rolling(250, min_periods=120).quantile(a)
    bal = ma.drop_duplicates('日期', keep='last').set_index('日期')['融资余额'].reindex(idx).ffill() if ma is not None else pd.Series(np.nan, index=idx)
    mdx = md.reindex(idx)
    micro = (1 + m['micro_ret'].fillna(0) / 100).cumprod()
    rel20 = (micro / micro.shift(20) - 1) - (hs / hs.shift(20) - 1)
    hs_r = hs.pct_change()
    B = bn.drop_duplicates('date', keep='last').set_index('date')['below_net_asset_ratio'].reindex(idx).ffill() if bn is not None else pd.Series(np.nan, index=idx)
    items = [
        ('地量见地价', '成交额跌到近一年最低 5%', amt <= q(amt, .05), 1),
        ('天量见天价', '成交额冲到近一年最高 5%', amt >= q(amt, .95), -1),
        ('恐慌跌停潮要抄底', '跌停超过 150 家', m['dt'] >= 150, 1),
        ('全市场大跌后敢买', '个股中位数跌超 3%', m['med_ret'] <= -3, 1),
        ('普涨大阳线后别追', '个股中位数涨超 3%', m['med_ret'] >= 3, -1),
        ('涨停过百是短期高潮', '涨停超过 100 家', m['zt'] >= 100, -1),
        ('连板到 8 板以上要见顶', '最高连板 8 板以上', m['max_streak'] >= 8, -1),
        ('五连阴之后有反弹', '上证连跌 5 天', down5, 1),
        ('五连阳之后要回调', '上证连涨 5 天', up5, -1),
        ('融资余额新高是顶部', '融资余额创近一年新高', bal >= bal.rolling(250, min_periods=120).max(), -1),
        ('融资大撤退之后是底', '融资余额 20 日降超 5%', bal / bal.shift(20) - 1 <= -.05, 1),
        ('破净股暴增是大底', '破净占比到近五年最高 10%', B >= B.rolling(1250, min_periods=500).quantile(.9), 1),
        ('八成股票破年线是底', '站上年线的股票不到 20%', mdx['ma250'] < 20, 1),
        ('九成股票站上 60 日线要小心', '站上 60 日线超过 85%', mdx['ma60'] > 85, -1),
        ('近期买入的人全被套是底', '近 60 日买入者不到 15% 浮盈', mdx['plw60'] < 15, 1),
        ('指数涨个股跌不可持续', '沪深300 涨超 0.5%、中位数跌超 0.5%', (hs_r > .005) & (m['med_ret'] < -.5), -1),
    ]
    f5, f20 = fwd(c1, 5).values, fwd(c1, 20).values
    rng = np.random.default_rng(20261104)
    rows = []
    for name, rule, mask, expect in items:
        raw = mask.fillna(False).astype(bool).values
        first = cooldown(raw)
        a, b = evaluate(first, f5, rng), evaluate(first, f20, rng)
        verdict = '样本太少'
        if b.get('med') is not None:
            sign = 1 if b['med'] > b['base'] else -1
            sign5 = 1 if a['med'] > a['base'] else -1
            if b['p'] < .1:
                verdict = '灵' if sign == expect else '反着来'
            elif a['p'] < .1:
                verdict = '短期灵' if sign5 == expect else '短期反着来'
            else:
                verdict = '分不清'
        trig = int(np.where(raw)[0][-1]) if raw.any() else None
        rows.append({'name': name, 'rule': rule, 'expect': expect, 'k5': a, 'k20': b, 'verdict': verdict,
                     'today': bool(raw[-1]), 'recent': bool(raw[-5:].any()), 'last': idx[trig] if trig is not None else None})
    order = {'灵': 0, '短期灵': 1, '反着来': 2, '短期反着来': 3, '分不清': 4, '样本太少': 5}
    rows.sort(key=lambda r: (not r['today'], not r['recent'], order[r['verdict']], -(r['k20'].get('n') or 0)))
    return {'rows': rows, 'start': idx[0], 'count': {k: sum(r['verdict'] == k for r in rows) for k in order}}


def rel20_after(m, piv):
    """验证“小票相对强势到极端后容易风格回撤”这句话：20 日超额历史分位 ≥90 / ≤10 之后，再看 20 日超额。"""
    hs = piv['沪深300'].reindex(m.index).ffill()
    micro = (1 + m['micro_ret'].fillna(0) / 100).cumprod()
    rel = (micro / micro.shift(20) - 1) - (hs / hs.shift(20) - 1)
    known = expanding_pct(rel)
    nxt = rel.shift(-20)
    return {'hi': stat(nxt[known >= 90]), 'lo': stat(nxt[known <= 10]), 'all': stat(nxt)}


# ================================================================ 历史同期
def season(piv):
    out = {}
    for name in ('中证1000', '上证指数'):
        s = piv[name].dropna()
        s.index = pd.to_datetime(s.index)
        mr = s.resample('ME').last().pct_change().dropna()
        out[name] = [{'m': int(k), 'med': rd(v.median() * 100, 1), 'win': rd((v > 0).mean() * 100, 0), 'n': int(len(v))} for k, v in mr.groupby(mr.index.month)]
    return out


# ================================================================ 解禁
def unlock(asof):
    fp, hp = RAW / 'unlock_future.csv', DATA / 'unlock_hist.csv'
    if not fp.exists() or not hp.exists():
        return None
    f, h = pd.read_csv(fp), pd.read_csv(hp)
    f = f[f['date'] > asof].sort_values('date')
    f['w'] = pd.to_datetime(f['date']).dt.to_period('W-FRI').dt.end_time.dt.strftime('%m-%d')
    wk = f.groupby('w', sort=False).agg(mv=('mv_yi', 'sum'), n=('n', 'sum')).reset_index()
    h['date'] = h['date'].astype(str)
    h = h[h['date'] <= asof]
    hs = h.set_index(pd.to_datetime(h['date']))['mv_yi']
    roll = hs.resample('D').sum().rolling(30).sum().dropna()
    d = pd.Timestamp(asof)
    n30 = f[pd.to_datetime(f['date']) <= d + pd.Timedelta(days=30)]['mv_yi'].sum()
    sec = pd.read_csv(RAW / 'unlock_sector.csv') if (RAW / 'unlock_sector.csv').exists() else None
    return {'weeks': [{'w': r['w'], 'mv': rd(r['mv'], 0), 'n': rd(r['n'], 0)} for _, r in wk.head(13).iterrows()],
            'next30': rd(n30, 0), 'next30_p': pct_of(roll, n30), 'past30': rd(roll.iloc[-1], 0) if len(roll) else None,
            'avg30': rd(roll.median(), 0) if len(roll) else None,
            'top': sec[sec['sw1'] != '未归类'].head(8).to_dict('records') if sec is not None else None}


# ================================================================ 财报与股东
def earnings(sec_rows):
    p = DATA / 'earnings_sector.csv'
    if not p.exists():
        return None
    e = pd.read_csv(p, dtype={'period': str})
    per = e['period'].max()
    x = e[e['period'] == per]
    al = x[x['sw1'] == '全部A股'].iloc[0]
    rs = {r['name']: r for r in sec_rows}
    rows = [{'name': r['sw1'], 'yoy': rd(r['med_yoy'], 1), 'pos': rd(r['pos'], 0), 'dbl': rd(r['dbl'], 0), 'crash': rd(r['crash'], 0), 'loss': rd(r['loss'], 0), 'n': int(r['n']),
             'r60': rs.get(r['sw1'], {}).get('r60'), 'rs60': rs.get(r['sw1'], {}).get('rs60')}
            for _, r in x.iterrows() if r['sw1'] in rs]
    rows.sort(key=lambda r: -(r['yoy'] if r['yoy'] is not None else -999))
    hist = []
    for k, g in e[e['sw1'] == '全部A股'].sort_values('period').iterrows():
        hist.append({'period': g['period'], 'yoy': rd(g['med_yoy'], 1), 'pos': rd(g['pos'], 0), 'loss': rd(g['loss'], 0)})
    out = {'period': per, 'n': int(al['n']), 'yoy': rd(al['med_yoy'], 1), 'pos': rd(al['pos'], 0), 'dbl': rd(al['dbl'], 0), 'crash': rd(al['crash'], 0), 'loss': rd(al['loss'], 0),
           'rows': rows, 'hist': hist, 'last_ann': str(al['last_ann'])}
    hp = DATA / 'holders_sector.csv'
    if hp.exists():
        h = pd.read_csv(hp, dtype={'period': str})
        h = h[h['period'] == h['period'].max()]
        a = h[h['sw1'] == '全部A股']
        y = h[h['sw1'] != '全部A股'].sort_values('med_chg')
        if len(a):
            out['holders'] = {'period': h['period'].iloc[0], 'med': rd(a['med_chg'].iloc[0], 1), 'up': rd(a['up'].iloc[0], 0), 'n': int(a['n'].iloc[0]),
                              'less': [{'name': r['sw1'], 'chg': rd(r['med_chg'], 1)} for _, r in y.head(5).iterrows()],
                              'more': [{'name': r['sw1'], 'chg': rd(r['med_chg'], 1)} for _, r in y.tail(5).iloc[::-1].iterrows()]}
    return out


# ================================================================ 新股与股息
def ipo(asof):
    p = RAW / 'ipo.csv'
    if not p.exists():
        return None
    x = pd.read_csv(p)
    x['list'] = x['list'].astype(str)
    listed = x[(x['list'] <= asof) & x['first'].notna() & (x['price'] > 0)].copy()
    listed['gain'] = (listed['first'] / listed['price'] - 1) * 100
    last = listed.sort_values('list').tail(20)
    listed['m'] = listed['list'].str[:7]
    mon = listed.groupby('m').agg(n=('list', 'size'), raise_yi=('raise_yi', 'sum'), gain=('gain', 'median')).tail(12).reset_index()
    return {'last20': {'n': int(len(last)), 'gain': rd(last['gain'].median(), 0), 'broken': int((last['gain'] < 0).sum()), 'prize': rd(last['prize'].median(), 0),
                       'rate': rd(last['rate'].median(), 3), 'since': last['list'].iloc[0] if len(last) else None},
            'months': [{'m': r['m'], 'n': int(r['n']), 'raise': rd(r['raise_yi'], 0), 'gain': rd(r['gain'], 0)} for _, r in mon.iterrows()],
            'upcoming': int((x['list'].isin(['nan', 'None']) | (x['list'] > asof)).sum())}


def dividend(bond10):
    p = RAW / 'dividend.csv'
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d['日期'] = d['日期'].astype(str)
    d = d.dropna(subset=['股息率'])
    cur = float(d['股息率'].iloc[-1])
    return {'date': d['日期'].iloc[-1], 'v': rd(cur), 'p': pct_of(d['股息率'], cur), 'gap': rd(cur - bond10, 2) if bond10 is not None else None}


# ================================================================ 环境分档（四问之一）
def tiers(m, md, c1000):
    """三个当时就知道的信号：多少股票站上 60 日线、情绪温度、近 60 日买入者赚不赚。"""
    idx = md.index
    ma_p = expanding_pct(md['ma60'])
    pl_p = expanding_pct(md['pl60'])
    temp = m['temp'].reindex(idx)
    score = (ma_p >= 50).astype(int) + (temp >= 50).astype(int) + (pl_p >= 50).astype(int)
    score = score.where(ma_p.notna() & temp.notna() & pl_p.notna())
    f5, f20 = fwd(c1000, 5).reindex(idx), fwd(c1000, 20).reindex(idx)
    names = {0: '三项全弱', 1: '偏弱', 2: '偏强', 3: '三项全强'}
    rows = [{'score': k, 'label': v, 'k5': stat(f5[score == k]), 'k20': stat(f20[score == k])} for k, v in names.items()]
    now = score.iloc[-1]
    return {'rows': rows, 'now': None if pd.isna(now) else int(now), 'label': names.get(None if pd.isna(now) else int(now)),
            'ma_p': rd(ma_p.iloc[-1], 0), 'pl_p': rd(pl_p.iloc[-1], 0), 'temp': rd(temp.iloc[-1], 0), 'all': {'k5': stat(f5), 'k20': stat(f20)}}


# ================================================================ 四问
def four(D, X):
    out = []
    T = X.get('tiers')
    F = X.get('feel')
    if T and T['now'] is not None and F:
        c = F['cards']
        row = T['rows'][T['now']]
        s = f"{c['ma60']['v']:.0f}% 的股票站上 60 日线，近 60 日买入者中位浮盈 {c['pl60']['v']:+.1f}%，情绪温度 {T['temp']:.0f}。"
        if row['k20']:
            s += f" 历史上这一档之后 20 日，中证1000 中位 {row['k20']['med']:+.1f}%，上涨概率 {row['k20']['win']:.0f}%（全部交易日 {T['all']['k20']['med']:+.1f}% / {T['all']['k20']['win']:.0f}%）。"
        out.append({'q': '现在是顺风还是逆风', 'a': T['label'], 'tone': 'up' if T['now'] >= 2 else 'down', 'text': s, 'to': 's_feel'})
    P = X.get('plays')
    if P:
        rows = [r for r in P['rows'] if r['key'] != 'all' and r['d20'] is not None]
        al = next(r for r in P['rows'] if r['key'] == 'all')
        best, worst = rows[0], rows[-1]
        s = (f"按 T+1 算、扣成本后，近 20 日{best['name']}日均 {best['d20']:+.2f}%（胜率 {best['win20']:.0f}%），{worst['name']}日均 {worst['d20']:+.2f}%；"
             f"闭眼买全市场 {al['d20']:+.2f}%。")
        ban = next(r for r in rows if r['key'] == 'ban1')
        if ban['p20'] is not None:
            s += f" 首板接力的 20 日表现在历史 {ban['p20']:.0f}% 分位" + ('，这类接力打法的好坏过去会延续一阵。' if (ban['keep'] or 0) >= .15 else '。')
        out.append({'q': '什么打法在赚钱', 'a': best['name'] if best['d20'] > 0 else '都在亏，' + best['name'] + '亏得最少', 'tone': 'up' if best['d20'] > 0 else 'down', 'text': s, 'to': 's_play'})
    S = X.get('sectors')
    if S:
        top = S['rows'][:3]
        s = '近 20 日相对全市场最强：' + '、'.join(f"{r['name']} {r['rs20']:+.1f}" for r in top) + ' 个百分点。'
        crowded = [r for r in S['rows'] if (r['amt_p'] or 0) >= 90]
        if crowded:
            s += ' 成交占比到自己近三年高位的有 ' + '、'.join(r['name'] for r in crowded[:4]) + '。'
        cheap = [r for r in S['rows'] if (r['pb_p'] or 100) < 30 and (r['rs20'] or -1) > 0]
        if cheap:
            s += ' 估值低位又在走强的有 ' + '、'.join(r['name'] for r in cheap[:4]) + '。'
        out.append({'q': '钱在往哪走', 'a': top[0]['name'], 'tone': 'up', 'text': s, 'to': 's6'})
    risks = []
    lv = D.get('leverage')
    if lv and lv.get('chg20_pct') is not None and lv['chg20_pct'] <= -3:
        risks.append(f"融资盘 20 日撤出 {abs(lv['chg20']):.0f} 亿（{lv['chg20_pct']:.1f}%）")
    if lv and (lv.get('guarantee_p') or 50) < 20:
        risks.append(f"维持担保比例 {lv['guarantee']:.0f}%，近三年 {lv['guarantee_p']:.0f}% 分位，急跌时强平压力大")
    U = X.get('unlock')
    if U and (U.get('next30_p') or 0) >= 80:
        risks.append(f"未来 30 天解禁 {U['next30']:.0f} 亿，处在近三年 {U['next30_p']:.0f}% 分位")
    if S:
        hot = [r for r in S['rows'] if (r['amt_p'] or 0) >= 95]
        if hot:
            risks.append(hot[0]['name'] + f"成交占比 {hot[0]['amt']:.0f}%，到了近三年前 5%")
    if F and (F['cards']['dd50']['p'] or 0) >= 80:
        risks.append(f"{F['cards']['dd50']['v']:.0f}% 的股票较一年高点跌超一半")
    E = X.get('earnings')
    if E and E.get('crash') is not None and E['crash'] >= 25:
        risks.append(f"{E['period'][:4]} 三季报净利下滑超 50% 的公司占 {E['crash']:.0f}%")
    if not risks:
        risks.append('杠杆、解禁、拥挤度都没有到历史高风险区')
    out.append({'q': '要提防什么', 'a': f'{len(risks)} 项' if len(risks) > 1 else '1 项', 'tone': 'amber', 'text': '；'.join(risks[:4]) + '。', 'to': 's5'})
    return out


# ================================================================ 每节开头的一句话结论
def verdicts(D, X):
    v = {}
    P = X.get('plays')
    if P:
        rows = [r for r in P['rows'] if r['key'] != 'all' and r['d20'] is not None]
        al = next(r for r in P['rows'] if r['key'] == 'all')
        pos = [r for r in rows if r['d20'] > 0]
        best, worst = rows[0], rows[-1]
        s = (f"近 20 日有 {len(pos)} 种打法扣成本后赚钱，最好的是{best['name']}（日均 {best['d20']:+.2f}%），" if pos else
             f"近 20 日没有一种打法扣成本后赚钱，亏得最少的是{best['name']}（日均 {best['d20']:+.2f}%），")
        s += f"最差的是{worst['name']}（日均 {worst['d20']:+.2f}%，胜率 {worst['win20']:.0f}%）；闭眼买全市场日均 {al['d20']:+.2f}%。"
        q = sorted((r for r in rows if r['q'] is not None), key=lambda r: -r['q'])
        if q:
            s += f" 今天开盘买入、收盘还拿着的，{q[0]['name']}浮盈最高（{q[0]['q']:+.2f}%），{q[-1]['name']}最低（{q[-1]['q']:+.2f}%）。"
        v['play'] = s
    F = X.get('feel')
    if F:
        c = F['cards']
        v['feel'] = (f"近 60 日买入的人，一半浮亏超过 {abs(c['pl60']['v']):.1f}%，只有 {c['plw60']['v']:.0f}% 的股票高于这段时间的平均成本；"
                     f"一半的股票比一年内高点低了 {abs(c['dd250']['v']):.0f}% 以上，{c['dd50']['v']:.0f}% 的股票跌超一半；今年到现在只有 {c['ytd_win']['v']:.0f}% 的股票赚钱。") \
            if c['pl60']['v'] is not None and c['pl60']['v'] < 0 else \
            (f"近 60 日买入的人中位浮盈 {c['pl60']['v']:+.1f}%，{c['plw60']['v']:.0f}% 的股票高于这段时间的平均成本；"
             f"一半的股票比一年内高点低了 {abs(c['dd250']['v']):.0f}% 以上；今年到现在 {c['ytd_win']['v']:.0f}% 的股票赚钱。")
    G = X.get('factors')
    if G:
        r = sorted((x for x in G['rows'] if x['d20'] is not None), key=lambda x: -x['d20'])
        v['style'] = (f"近 20 日{r[0]['name']}最占优（多空差 {r[0]['d20']:+.1f} 个百分点），{r[-1]['name']}最吃亏（{r[-1]['d20']:+.1f}）。"
                      f"长期年化最高的是{max(G['rows'], key=lambda x: x['ann'] or -999)['name']}。")
    S = X.get('sectors')
    if S:
        q = S['quad']
        cheap = [r['name'] for r in S['rows'] if (r['pb_p'] or 100) < 50 and (r['rs20'] or -1) > 0]
        s = f"历史上“便宜且走强”的行业，之后 20 日相对全市场中位 {q['便宜且走强']['med']:+.2f}%、跑赢概率 {q['便宜且走强']['win']:.0f}%；“贵且走弱”是 {q['贵且走弱']['med']:+.2f}% / {q['贵且走弱']['win']:.0f}%。"
        if cheap:
            s += ' 现在落在便宜且走强象限的：' + '、'.join(cheap[:6]) + '。'
        v['sector'] = s
    Y = X.get('sayings')
    if Y:
        c = Y['count']
        s = f"{len(Y['rows'])} 条老说法里，{c['灵'] + c['短期灵']} 条经得起检验，{c['反着来'] + c['短期反着来']} 条实际反着来，{c['分不清']} 条分不清。"
        now = [r['name'] for r in Y['rows'] if r['today']]
        if now:
            s += ' 今天触发：' + '、'.join(now) + '。'
        v['say'] = s
    return v


# ================================================================ 组装
def safe(name, fn, errs):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001  一块出错不影响其他块
        errs.append({'block': name, 'err': f'{type(e).__name__}: {e}'[:200]})
        print(f'[FAIL] 判断块 {name}: {e}\n{traceback.format_exc()[-600:]}', flush=True)
        return None


def build(D, m, piv):
    """m: 带温度的 daily_metrics（索引为日期）；piv: 指数收盘价。"""
    errs = []
    X = {}
    md_p, sd_p = DATA / 'market_daily.csv', DATA / 'sector_daily.csv'
    if md_p.exists():
        md = pd.read_csv(md_p).set_index('date').sort_index()
        c1000 = piv['中证1000']
        extra = {}
        lp = RAW / 'market_latest.json'
        if lp.exists():
            extra = json.loads(lp.read_text(encoding='utf-8'))
        X['plays'] = safe('打法', lambda: plays(md), errs)
        X['funnel'] = safe('晋级', lambda: funnel(md), errs)
        X['feel'] = safe('持仓', lambda: feel(md, c1000, extra), errs)
        X['grid'] = safe('风格矩阵', lambda: grid(md), errs)
        X['factors'] = safe('风格因子', lambda: factors(md), errs)
        X['tiers'] = safe('环境分档', lambda: tiers(m, md, c1000), errs)
        if sd_p.exists():
            X['sectors'] = safe('行业', lambda: sectors(pd.read_csv(sd_p), md), errs)
        ma = pd.read_csv(RAW / 'margin_account.csv') if (RAW / 'margin_account.csv').exists() else None
        if ma is not None:
            ma['日期'] = ma['日期'].astype(str)
        bn = pd.read_csv(RAW / 'below_net.csv') if (RAW / 'below_net.csv').exists() else None
        if bn is not None:
            bn['date'] = bn['date'].astype(str)
        X['sayings'] = safe('说法', lambda: sayings(m, md, piv, ma, bn), errs)
        X['rel20'] = safe('风格回撤', lambda: rel20_after(m, piv), errs)
    X['season'] = safe('历史同期', lambda: season(piv), errs)
    X['unlock'] = safe('解禁', lambda: unlock(D['asof']), errs)
    X['earnings'] = safe('财报', lambda: earnings((X.get('sectors') or {}).get('rows', [])), errs)
    X['ipo'] = safe('新股', lambda: ipo(D['asof']), errs)
    X['dividend'] = safe('股息', lambda: dividend((D.get('bond') or {}).get('cn10')), errs)
    X['four'] = safe('四问', lambda: four(D, X), errs)
    X['v'] = safe('结论', lambda: verdicts(D, X), errs)
    X['errors'] = errs
    return X
