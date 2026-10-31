"""把 data/ 里的原始数据算成指标和判断，写 site/data.json 并渲染 site/index.html。

    python scripts/build.py

所有“历史上类似情形之后”的统计都只用当时已知的数据分档（滚动/扩张分位），避免用未来数据给过去打标签。
"""
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
RAW = DATA / 'raw'
SITE = ROOT / 'site'
WIN = 250  # 情绪分位用的滚动窗口（约一年）
# 新浪“概念”里混着持仓/指数/财务类标签，不算题材
NOT_THEME = '超大盘|大盘|小盘|重仓|融资融券|股通|MSCI|标普|富时|罗素|转融|预盈|预增|预亏|预减|送转|破净|次新|成份|成分|指数|ST|基金|社保|QFII|养老金|证金|汇金|机构'


def rd(x, n=2):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, n)


def read(name, **kw):
    p = RAW / name if (RAW / name).exists() else DATA / name
    return pd.read_csv(p, **kw) if p.exists() else None


def pct_rank_last(s):
    """窗口内最后一个值的分位（0-1），并列取中间。"""
    a = np.asarray(s, dtype=float)
    v = a[-1]
    a = a[~np.isnan(a)]
    if np.isnan(v) or len(a) < 20:
        return np.nan
    return ((a < v).sum() + 0.5 * ((a == v).sum() - 1)) / max(len(a) - 1, 1)


def hist_pct(series, value):
    a = pd.Series(series).dropna().values
    if len(a) == 0 or value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return rd(((a < value).sum() + 0.5 * (a == value).sum()) / len(a) * 100, 1)


def fmt_pct(x, n=1, sign=True):
    if x is None:
        return '—'
    return f'{x:+.{n}f}%' if sign else f'{x:.{n}f}%'


# ================================================================ 情绪
FEATS = {  # 名称, 方向（-1 表示越小越热）
    'up_ratio': ('上涨家数占比', 1), 'net_limit': ('涨停减跌停', 1), 'zb_rate': ('炸板率', -1),
    'zt_prem': ('昨日涨停今日溢价', 1), 'max_streak': ('连板高度', 1), 'promo_1to2': ('首板晋级率', 1),
}


def sentiment(m):
    m = m.copy()
    m['up_ratio'] = m['up'] / (m['up'] + m['down'])
    m['net_limit'] = m['zt'] - m['dt']
    m['zb_rate'] = m['zb'] / (m['zt'] + m['zb'])
    for k, (_, sgn) in FEATS.items():
        r = m[k].astype(float).rolling(WIN, min_periods=60).apply(pct_rank_last, raw=True)
        m['p_' + k] = r if sgn > 0 else 1 - r
    pc = m[['p_' + k for k in FEATS]]
    m['temp'] = (pc.mean(axis=1) * 100).where(pc.notna().sum(axis=1) >= 3)
    m['temp_ma5'] = m['temp'].rolling(5, min_periods=3).mean()
    m['dtemp5'] = m['temp'] - m['temp'].shift(5)
    m['amount_p'] = m['amount_yi'].rolling(WIN, min_periods=60).apply(pct_rank_last, raw=True) * 100
    m['stage'] = [stage(t, d) for t, d in zip(m['temp'], m['dtemp5'])]
    return m


def stage(t, d):
    if t is None or np.isnan(t):
        return None
    d = 0 if d is None or np.isnan(d) else d
    if t <= 20:
        return '冰点'
    if t >= 80:
        return '高潮' if d >= 0 else '高位分歧'
    if d <= -12:
        return '退潮'
    if d >= 12:
        return '修复' if t < 50 else '发酵'
    return '低位震荡' if t < 40 else ('中位震荡' if t < 60 else '高位震荡')


STAGE_NOTE = {
    '冰点': '亏钱效应集中释放，短线资金普遍空仓或止损。历史上冰点常在 1–3 天内出现修复，但修复的第一天往往只有少数核心股走强。',
    '退潮': '前期强势股开始补跌，连板高度下移、炸板增多。退潮期追高的胜率最低，资金倾向于切换到低位或防守方向。',
    '修复': '情绪从低位回升，首板和低位股先活跃。能否转为发酵，关键看连板高度能否打开、昨日涨停能否继续给出正溢价。',
    '发酵': '赚钱效应扩散，接力意愿增强。需要留意主线是否集中、成交额是否同步放大——缩量发酵持续性通常一般。',
    '高潮': '涨停家数和连板高度都处于一年高位，人气最旺的阶段往往也是分歧的起点。',
    '高位分歧': '情绪仍在高位但开始走弱，高标股分化，次日溢价和晋级率通常最先回落。',
    '低位震荡': '情绪偏冷但没有进一步恶化，市场缺乏主线，适合观察而不是试错。',
    '中位震荡': '情绪平稳，没有明显的方向性信号，结构比指数重要。',
    '高位震荡': '情绪偏热但没有继续抬升，资金在高位反复换手。',
}


def forward(series_close, h):
    return series_close.shift(-h) / series_close - 1


def base_rate(mask, fwd):
    x = fwd[mask].dropna()
    if len(x) < 15:
        return None
    return {'n': int(len(x)), 'win': rd((x > 0).mean() * 100, 0), 'med': rd(x.median() * 100, 2),
            'q25': rd(x.quantile(.25) * 100, 2), 'q75': rd(x.quantile(.75) * 100, 2)}


# ================================================================ 主流程
def main():
    m = pd.read_csv(DATA / 'daily_metrics.csv').sort_values('date').reset_index(drop=True)
    m = sentiment(m)
    last = m.iloc[-1]
    asof = last['date']

    idx = read('index_daily.csv')
    idx['date'] = idx['date'].astype(str)
    piv = idx.pivot_table(index='date', columns='name', values='close').sort_index()

    # 微盘股指数：最小市值 400 只等权，每日换仓；涨停池补的断档日没有截面，用中证1000 当日涨跌代替
    mm = m.set_index('date')
    csi1000_ret = piv['中证1000'].pct_change() * 100
    micro = mm['micro_ret'].copy()
    filled_days = [d for d in micro.index if np.isnan(micro[d]) and d in csi1000_ret.index and d >= '2016-01-05']
    micro.loc[filled_days] = csi1000_ret.reindex(filled_days)
    micro_idx = (1 + micro.fillna(0) / 100).cumprod() * 1000

    # 前瞻收益（只用于统计历史基准率）
    allA = piv['中证1000'].reindex(mm.index)  # 新浪的中证全指历史不全，用中证1000 代表中小盘
    fwd = {
        'k5': forward(allA, 5), 'k20': forward(allA, 20),
        'micro5': forward(micro_idx, 5), 'micro20': forward(micro_idx, 20),
    }
    t_now = last['temp']
    stg = last['stage']
    tb = int(min(t_now // 20, 4)) if not np.isnan(t_now) else None
    temp_s = mm['temp']
    in_bucket = (temp_s // 20).clip(upper=4) == tb
    same_stage = mm['stage'] == stg
    rates = {
        'bucket': f'{tb * 20}–{tb * 20 + 20}' if tb is not None else None,
        'by_temp': {k: base_rate(in_bucket, v) for k, v in fwd.items()},
        'by_stage': {k: base_rate(same_stage, v) for k, v in fwd.items()},
        'all': {k: base_rate(temp_s.notna(), v) for k, v in fwd.items()},
    }

    # 最像今天的交易日（情绪六项分位的欧氏距离，排除最近 30 个交易日，彼此至少间隔 10 天）
    pcols = ['p_' + k for k in FEATS]
    X = mm[pcols]
    today_vec = X.iloc[-1]
    ok_cols = today_vec.dropna().index
    dist = np.sqrt(((X[ok_cols] - today_vec[ok_cols]) ** 2).sum(axis=1)).where(X[ok_cols].notna().all(axis=1))
    dist = dist.iloc[:-30].dropna().sort_values()
    picked = []
    pos = {d: i for i, d in enumerate(mm.index)}
    for d, v in dist.items():
        if all(abs(pos[d] - pos[p]) >= 10 for p in picked):
            picked.append(d)
        if len(picked) == 8:
            break
    similar = [{'date': d, 'dist': rd(dist[d], 3), 'temp': rd(mm.at[d, 'temp'], 0), 'stage': mm.at[d, 'stage'],
                'zt': rd(mm.at[d, 'zt'], 0), 'dt': rd(mm.at[d, 'dt'], 0), 'streak': rd(mm.at[d, 'max_streak'], 0),
                'up': rd(mm.at[d, 'up_ratio'] * 100 if not np.isnan(mm.at[d, 'up_ratio']) else None, 0),
                'k5': rd(fwd['k5'][d] * 100), 'k20': rd(fwd['k20'][d] * 100),
                'micro5': rd(fwd['micro5'][d] * 100), 'micro20': rd(fwd['micro20'][d] * 100)} for d in picked]
    sim_stats = {}
    for k in ('k5', 'k20', 'micro5', 'micro20'):
        v = [s[k] for s in similar if s[k] is not None]
        sim_stats[k] = {'med': rd(np.median(v)) if v else None, 'win': len([x for x in v if x > 0]), 'n': len(v)}

    # ---------------- 指数
    ind_rows = []
    for name in ['上证指数', '深证成指', '创业板指', '科创50', '沪深300', '中证500', '中证1000']:
        s = piv[name].dropna()
        c = s.iloc[-1]
        y0 = s[s.index < asof[:4] + '-01-01']
        hi, lo = s.iloc[-250:].max(), s.iloc[-250:].min()
        ind_rows.append({'name': name, 'close': rd(c), 'date': s.index[-1],
                         'r1': rd((c / s.iloc[-2] - 1) * 100), 'r5': rd((c / s.iloc[-6] - 1) * 100),
                         'r20': rd((c / s.iloc[-21] - 1) * 100),
                         'ytd': rd((c / y0.iloc[-1] - 1) * 100) if len(y0) else None,
                         'pos250': rd((c - lo) / (hi - lo) * 100, 0) if hi > lo else None})
    span = piv.index[-500:]
    style = {'dates': list(span)}
    base = span[0]
    for name in ['沪深300', '中证1000']:
        style[name] = [rd(v / piv.at[base, name] * 100) for v in piv.loc[span, name]]
    mi = micro_idx.reindex(span).ffill()
    style['微盘400'] = [rd(v / mi.iloc[0] * 100) for v in mi]
    # 微盘相对沪深300 的 20 日超额，及其历史分位
    rel20_hist = (micro_idx / micro_idx.shift(20) - 1) - (piv['沪深300'].reindex(micro_idx.index).pipe(lambda s: s / s.shift(20) - 1))
    rel20 = rel20_hist.iloc[-1] * 100
    style['rel20'] = rd(rel20)
    style['rel20_p'] = hist_pct(rel20_hist * 100, rel20)
    style['rel60'] = rd(((micro_idx.iloc[-1] / micro_idx.iloc[-61] - 1) -
                         (piv['沪深300'].iloc[-1] / piv['沪深300'].iloc[-61] - 1)) * 100)
    style['micro_ytd'] = rd((micro_idx.iloc[-1] / micro_idx[micro_idx.index < asof[:4] + '-01-01'].iloc[-1] - 1) * 100)
    style['filled_days'] = filled_days[-10:]

    # 指数涨、中位数跌：“赚了指数不赚钱”
    hs300_r = piv['沪深300'].pct_change().reindex(mm.index) * 100
    div = ((hs300_r > 0.3) & (mm['med_ret'] < -0.3)) | ((hs300_r < -0.3) & (mm['med_ret'] > 0.3))
    divergence = {'last20': int(div.iloc[-20:].sum()), 'hs300_r': rd(hs300_r.iloc[-1]), 'med_r': rd(mm['med_ret'].iloc[-1]),
                  'med_r20': rd(mm['med_ret'].iloc[-20:].sum())}

    # ---------------- 估值
    val = {}
    ebs = read('ebs.csv')
    if ebs is not None:
        ebs['日期'] = ebs['日期'].astype(str)
        ebs = ebs.dropna(subset=['股债利差']).reset_index(drop=True)
        e = ebs['股债利差'] * 100
        cur = e.iloc[-1]
        expanding_p = e.expanding(min_periods=250).apply(lambda a: (a[:-1] < a[-1]).mean() if len(a) > 1 else np.nan, raw=True)
        fwd250 = ebs['沪深300指数'].shift(-250) / ebs['沪深300指数'] - 1
        q_now = expanding_p.iloc[-1]
        lo_q, hi_q = math.floor(q_now * 5) / 5, math.floor(q_now * 5) / 5 + 0.2
        mask = (expanding_p >= lo_q) & (expanding_p < hi_q + (1e-9 if hi_q >= 1 else 0))
        ten = ebs['日期'] >= str(int(asof[:4]) - 10) + asof[4:]
        val['ebs'] = {'date': ebs['日期'].iloc[-1], 'value': rd(cur), 'p_all': hist_pct(e, cur),
                      'p10': hist_pct(e[ten], cur), 'band': f'{int(lo_q * 100)}–{int(min(hi_q, 1) * 100)}',
                      'fwd1y': base_rate(mask, fwd250),
                      'fwd1y_all': base_rate(expanding_p.notna(), fwd250),
                      'series': {'d': list(ebs['日期'].iloc[::5]), 'v': [rd(x) for x in e.iloc[::5]],
                                 'idx': [rd(x, 0) for x in ebs['沪深300指数'].iloc[::5]]}}
    pe = read('pe_median.csv')
    if pe is not None:
        r = pe.iloc[-1]
        val['pe'] = {'date': str(r['date']), 'median': rd(r['middlePETTM']), 'p10': rd(r['quantileInRecent10YearsMiddlePeTtm'] * 100, 0),
                     'p_all': rd(r['quantileInAllHistoryMiddlePeTtm'] * 100, 0),
                     'series': {'d': list(pe['date'].astype(str)), 'v': [rd(x) for x in pe['middlePETTM']]}}
    pb = read('pb_median.csv')
    if pb is not None:
        r = pb.iloc[-1]
        val['pb'] = {'date': str(r['date']), 'median': rd(r['middlePB']), 'p10': rd(r['quantileInRecent10YearsMiddlePB'] * 100, 0),
                     'series': {'d': list(pb['date'].astype(str)), 'v': [rd(x) for x in pb['middlePB']]}}
    bn = read('below_net.csv')
    if bn is not None:
        r = bn.iloc[-1]
        ratio = bn['below_net_asset_ratio'] * 100
        val['below_net'] = {'date': str(r['date']), 'n': int(r['below_net_asset']), 'ratio': rd(r['below_net_asset_ratio'] * 100, 1),
                            'p_all': hist_pct(ratio, ratio.iloc[-1]),
                            'series': {'d': list(bn['date'].astype(str).iloc[::5]), 'v': [rd(x, 1) for x in ratio.iloc[::5]]}}
    bf = read('buffett.csv')
    if bf is not None:
        b = bf['总市值'] / bf['GDP'] * 100
        ten = bf['日期'].astype(str) >= str(int(asof[:4]) - 10) + asof[4:]
        val['buffett'] = {'date': str(bf['日期'].iloc[-1]), 'value': rd(b.iloc[-1], 0), 'p10': hist_pct(b[ten], b.iloc[-1]),
                          'p_all': hist_pct(b, b.iloc[-1])}
    cg = read('congestion.csv')
    if cg is not None:
        c2 = cg.dropna(subset=['congestion'])
        val['congestion'] = {'date': str(c2['date'].iloc[-1]), 'value': rd(c2['congestion'].iloc[-1] * 100, 1),
                             'p_all': hist_pct(c2['congestion'], c2['congestion'].iloc[-1])}

    # ---------------- 新高新低
    hl = read('highlow.csv')
    breadth = None
    if hl is not None:
        r = hl.iloc[-1]
        net20 = hl['high20'] - hl['low20']
        f5 = hl['close'].shift(-5) / hl['close'] - 1
        extreme = hl['low20'] >= hl['low20'].quantile(.9)
        breadth = {'date': str(r['date']), 'high20': int(r['high20']), 'low20': int(r['low20']), 'high60': int(r['high60']),
                   'low60': int(r['low60']), 'high120': int(r['high120']), 'low120': int(r['low120']),
                   'low20_p': hist_pct(hl['low20'], r['low20']), 'net20_p': hist_pct(net20, net20.iloc[-1]),
                   'after_extreme_low': base_rate(extreme, f5),
                   'series': {'d': list(hl['date'].astype(str)), 'h': [int(x) for x in hl['high20']], 'l': [int(x) for x in hl['low20']]}}

    # ---------------- 两融
    lev = None
    ma = read('margin_account.csv')
    if ma is not None:
        ma['日期'] = ma['日期'].astype(str)
        ma = ma.dropna(subset=['融资余额'])
        amt = mm['amount_yi']
        ma['ratio'] = ma['融资买入额'] / ma['日期'].map(amt) * 100
        r = ma.iloc[-1]
        bal = ma['融资余额']
        lev = {'date': r['日期'], 'balance': rd(r['融资余额'], 0), 'buy': rd(r['融资买入额'], 0),
               'chg5': rd(bal.iloc[-1] - bal.iloc[-6], 0), 'chg20': rd(bal.iloc[-1] - bal.iloc[-21], 0),
               'chg20_pct': rd((bal.iloc[-1] / bal.iloc[-21] - 1) * 100),
               'ratio': rd(ma['ratio'].dropna().iloc[-1]) if ma['ratio'].notna().any() else None,
               'ratio_date': ma.dropna(subset=['ratio'])['日期'].iloc[-1] if ma['ratio'].notna().any() else None,
               'ratio_p': hist_pct(ma['ratio'], ma['ratio'].dropna().iloc[-1]) if ma['ratio'].notna().any() else None,
               'guarantee': rd(r['平均维持担保比例'], 1),
               'guarantee_p': hist_pct(ma['平均维持担保比例'].iloc[-750:], r['平均维持担保比例']),
               'bal_p': hist_pct(bal, bal.iloc[-1]),
               'series': {'d': list(ma['日期'].iloc[-750:]), 'bal': [rd(x, 0) for x in bal.iloc[-750:]],
                          'ratio': [rd(x) for x in ma['ratio'].iloc[-750:]]}}

    # ---------------- 利率
    rates_bond = None
    by = read('bond_yield.csv')
    if by is not None:
        by = by.dropna(subset=['中国国债收益率10年'])
        r = by.iloc[-1]
        yago = by[by['日期'].astype(str) <= str(int(str(r['日期'])[:4]) - 1) + str(r['日期'])[4:10]]
        rates_bond = {'date': str(r['日期'])[:10], 'cn10': rd(r['中国国债收益率10年'], 3), 'cn2': rd(r['中国国债收益率2年'], 3),
                      'cn30': rd(r['中国国债收益率30年'], 3),
                      'cn10_1y': rd((r['中国国债收益率10年'] - yago['中国国债收益率10年'].iloc[-1]) * 100, 0) if len(yago) else None,
                      'us10': rd(r['美国国债收益率10年'], 2)}

    # ---------------- 涨停结构
    pool = read('pool_daily.csv', dtype={'code': str})
    limit = None
    if pool is not None and len(pool):
        pdays = sorted(pool['date'].unique())
        pd_last = pdays[-1]
        today = pool[pool['date'] == pd_last]
        zt = today[today['kind'] == 'zt'].sort_values(['streak', 'first'], ascending=[False, True])
        zbp = today[today['kind'] == 'zb']
        dtp = today[today['kind'] == 'dt']
        ladder = []
        for k, g in zt.groupby('streak', sort=False):
            ladder.append({'streak': int(k), 'n': len(g), 'stocks': [
                {'name': r['name'], 'ind': r['ind'], 'first': r['first'][:2] + ':' + r['first'][2:4] if isinstance(r['first'], str) else '',
                 'opens': int(r['opens']), 'seal': rd(r['seal_yi']), 'oneword': r['first'] == '092500' and r['opens'] == 0,
                 'stat': r['stat']}
                for _, r in g.head(12).iterrows()]})
        ladder.sort(key=lambda x: -x['streak'])
        last5 = pdays[-5:]
        p5 = pool[pool['date'].isin(last5) & (pool['kind'] == 'zt')]
        ind5 = p5.groupby(['ind', 'date']).size().unstack(fill_value=0).reindex(columns=last5, fill_value=0)
        ind5['sum'] = ind5.sum(axis=1)
        ind5['days'] = (ind5[last5] > 0).sum(axis=1)
        ind5 = ind5.sort_values(['sum', 'days'], ascending=False).head(12)
        dt_ind = dtp.groupby('ind').size().sort_values(ascending=False).head(6)
        limit = {
            'date': pd_last, 'zt': len(zt), 'zb': len(zbp), 'dt': len(dtp),
            'seal_rate': rd(len(zt) / (len(zt) + len(zbp)) * 100, 0) if len(zt) + len(zbp) else None,
            'oneword': int(((zt['first'] == '092500') & (zt['opens'] == 0)).sum()),
            'first_board': int((zt['streak'] == 1).sum()), 'ladder': ladder,
            'mainline': {'days': last5, 'rows': [{'ind': i, 'by_day': [int(r[d]) for d in last5], 'sum': int(r['sum']),
                                                  'days': int(r['days'])} for i, r in ind5.iterrows()]},
            'dt_ind': [{'ind': i, 'n': int(v)} for i, v in dt_ind.items()],
            'dt_top': [{'name': r['name'], 'ind': r['ind'], 'streak': int(r['streak']) if not np.isnan(r['streak']) else 1}
                       for _, r in dtp.sort_values('streak', ascending=False).head(8).iterrows()],
        }

    # ---------------- 行业（申万一级，按总市值加权）与概念
    spot = read('spot_latest.csv', dtype={'code': str})
    industries = None
    movers = None
    if spot is not None:
        imap = pd.read_csv(DATA / 'industry_sw1.csv', dtype={'code': str})
        s = spot.merge(imap, on='code', how='left')
        s['sw1'] = s['sw1'].fillna('未归类')
        s['w'] = s['mv_wan'] / (1 + s['ret'] / 100)
        tot_amt = s['amount'].sum()
        g = []
        for name, x in s.groupby('sw1'):
            if name == '未归类' or len(x) < 5:
                continue
            g.append({'name': name, 'n': len(x), 'ret': rd((x['ret'] * x['w']).sum() / x['w'].sum()),
                      'med': rd(x['ret'].median()), 'up': rd((x['ret'] > 0).mean() * 100, 0),
                      'amt_share': rd(x['amount'].sum() / tot_amt * 100, 1), 'zt': int(x['zt'].sum()), 'dt': int(x['dt'].sum()),
                      'pe_med': rd(x.loc[x['pe'] > 0, 'pe'].median(), 1)})
        industries = sorted(g, key=lambda r: -(r['ret'] or 0))
        z = s[~s['st'].astype(bool)]
        big = z[z['mv_wan'] >= 1e6]  # 百亿市值以上
        movers = {
            'gain': z.nlargest(10, 'ret')[['name', 'ret', 'sw1', 'amount']].to_dict('records'),
            'lose': z.nsmallest(10, 'ret')[['name', 'ret', 'sw1', 'amount']].to_dict('records'),
            'amount': s.nlargest(10, 'amount')[['name', 'ret', 'sw1', 'amount']].to_dict('records'),
            'big_gain': big.nlargest(8, 'ret')[['name', 'ret', 'sw1', 'amount']].to_dict('records'),
            'big_lose': big.nsmallest(8, 'ret')[['name', 'ret', 'sw1', 'amount']].to_dict('records'),
        }
        for v in movers.values():
            for r in v:
                r['ret'] = rd(r['ret'])
                r['amount'] = rd(r['amount'] / 1e8, 1)
    boards = read('board_daily.csv')
    concepts = None
    ths = None
    if boards is not None:
        bl = boards[boards['date'] == boards['date'].max()]
        c = bl[(bl['kind'] == 'concept') & (bl['n'] >= 8) & ~bl['name'].str.contains(NOT_THEME)].sort_values('pct', ascending=False)
        concepts = {'date': bl['date'].iloc[0],
                    'top': c.head(10)[['name', 'pct', 'n', 'leader', 'leader_pct']].to_dict('records'),
                    'bottom': c.tail(10).iloc[::-1][['name', 'pct', 'n', 'leader', 'leader_pct']].to_dict('records')}
        t = bl[bl['kind'] == 'ths_ind'].dropna(subset=['net_in_yi']).sort_values('net_in_yi', ascending=False)
        if len(t):
            ths = {'in': t.head(6)[['name', 'pct', 'net_in_yi']].to_dict('records'),
                   'out': t.tail(6).iloc[::-1][['name', 'pct', 'net_in_yi']].to_dict('records')}

    # ---------------- ETF 份额
    etf = None
    es = read('etf_shares.csv', dtype={'code': str})
    if es is not None and len(es):
        # 份额变化 × 最新价 ≈ 净申赎金额（亿元）；没有价格时退回份额
        price = read('etf_price.csv', dtype={'code': str})
        pmap = dict(zip(price['code'], price['price'])) if price is not None else {}
        sh = es.pivot_table(index='date', columns='code', values='shares_yi').sort_index().ffill()
        amt = sh.mul(pd.Series({c: pmap.get(c, np.nan) for c in sh.columns}), axis=1)
        flow = amt - amt.iloc[0]
        track_of = es.drop_duplicates('code').set_index('code')['track']
        etf = {'dates': list(sh.index), 'priced': bool(pmap), 'tracks': {}}
        for tr, codes in track_of.groupby(track_of).groups.items():
            a = amt[list(codes)].sum(axis=1, min_count=1)
            sv = sh[list(codes)].sum(axis=1)
            d = lambda k: rd(a.iloc[-1] - a.iloc[-1 - k], 1) if len(a) > k else None
            etf['tracks'][tr] = {'shares': rd(sv.iloc[-1]), 'aum': rd(a.iloc[-1], 0), 'd1': d(1), 'd5': d(5), 'd20': d(20),
                                 'series': [rd(x, 1) for x in flow[list(codes)].sum(axis=1, min_count=1)]}
        if pmap:
            etf['d5_total'] = rd(sum((x['d5'] or 0) for x in etf['tracks'].values()), 0)
            etf['d20_total'] = rd(sum((x['d20'] or 0) for x in etf['tracks'].values()), 0)

    # ---------------- 龙虎榜
    lhb = None
    ld = read('lhb_daily.csv', dtype={'code': str})
    ls = read('lhb_seat_daily.csv')
    if ld is not None and len(ld):
        x = ld[ld['date'] == ld['date'].max()].sort_values('net_yi', ascending=False)
        lhb = {'date': x['date'].iloc[0], 'n': len(x),
               'buy': x.head(8)[['name', 'pct', 'net_yi', 'reason', 'note']].to_dict('records'),
               'sell': x.tail(5).iloc[::-1][['name', 'pct', 'net_yi', 'reason', 'note']].to_dict('records')}
        if ls is not None and len(ls):
            y = ls[ls['date'] == ls['date'].max()]
            inst = y[y['seat'].str.contains('机构专用')]['net_yi'].sum()
            hk = y[y['seat'].str.contains('股通专用')]['net_yi'].sum()
            hot = y[~y['seat'].str.contains('机构专用|股通专用')].sort_values('net_yi', ascending=False)
            lhb.update({'inst_net': rd(inst), 'hk_net': rd(hk),
                        'seats': [{'seat': r['seat'].replace('股份有限公司', '').replace('有限责任公司', '').replace('证券营业部', ''),
                                   'net': rd(r['net_yi']), 'stocks': str(r['stocks'])[:40]} for _, r in hot.head(6).iterrows()]})

    # ---------------- 情绪时间序列（给图）
    tail = m.iloc[-500:]
    sent_series = {'d': list(tail['date']), 'temp': [rd(x, 0) for x in tail['temp']], 'zt': [rd(x, 0) for x in tail['zt']],
                   'dt': [rd(x, 0) for x in tail['dt']], 'zb': [rd(x, 0) for x in tail['zb']],
                   'streak': [rd(x, 0) for x in tail['max_streak']], 'prem': [rd(x) for x in tail['zt_prem']],
                   'up': [rd(x * 100, 1) for x in tail['up_ratio']], 'amount': [rd(x, 0) for x in tail['amount_yi']],
                   'promo': [rd(x, 1) for x in tail['promo_1to2']]}
    comps = {k: {'label': lab, 'p': rd(last['p_' + k] * 100, 0), 'raw': rd(last[k] * (100 if k in ('up_ratio', 'zb_rate') else 1), 1)}
             for k, (lab, _) in FEATS.items()}

    today = {k: rd(last[k], 2) for k in ['n', 'up', 'down', 'amount_yi', 'med_ret', 'up5', 'down5', 'zt', 'dt', 'zb',
                                         'max_streak', 'n_streak2', 'zt_prem', 'promo_1to2', 'micro_ret', 'temp', 'dtemp5', 'amount_p']}
    today['stage'] = stg
    today['amount_5avg'] = rd(m['amount_yi'].iloc[-5:].mean(), 0)
    today['amount_chg'] = rd(m['amount_yi'].iloc[-1] - m['amount_yi'].iloc[-2], 0)

    D = {
        'asof': asof, 'built': (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M'),
        'today': today, 'stage_note': STAGE_NOTE.get(stg, ''), 'components': comps, 'rates': rates,
        'similar': similar, 'sim_stats': sim_stats, 'indices': ind_rows, 'style': style, 'divergence': divergence,
        'valuation': val, 'breadth': breadth, 'leverage': lev, 'bond': rates_bond, 'limit': limit,
        'industries': industries, 'movers': movers, 'concepts': concepts, 'ths': ths, 'etf': etf, 'lhb': lhb,
        'sent_series': sent_series, 'history_start': m['date'].iloc[0],
        'collect': json.loads((DATA / 'last_collect.json').read_text(encoding='utf-8')) if (DATA / 'last_collect.json').exists() else None,
    }
    D['takeaways'] = takeaways(D)
    D['headline'] = headline(D)
    SITE.mkdir(exist_ok=True)
    payload = json.dumps(D, ensure_ascii=False, separators=(',', ':'), allow_nan=False, default=lambda o: None)
    (SITE / 'data.json').write_text(payload, encoding='utf-8')
    tpl = (ROOT / 'scripts' / 'template.html').read_text(encoding='utf-8')
    html = tpl.replace('/*__DATA__*/null', payload.replace('</', '<\\/'))
    (SITE / 'index.html').write_text(html, encoding='utf-8')
    print('built', asof, 'temp', today['temp'], stg, f'{len(html) / 1024:.0f}KB')


# ================================================================ 文字判断
def headline(D):
    """把几个最有方向性的信号拼成一句判断；信号不明确的就不说。"""
    t, parts, money = D['today'], [], []
    stg = t.get('stage')
    if stg:
        parts.append({'冰点': '短线情绪到了冰点', '退潮': '短线情绪退潮', '修复': '短线情绪在修复', '发酵': '赚钱效应在扩散',
                      '高潮': '短线情绪处在高潮', '高位分歧': '情绪高位分歧'}.get(stg, '情绪' + stg))
    low = [r for r in D['indices'] if r['pos250'] is not None and r['pos250'] <= 3]
    if len(low) >= 4:
        parts.append(f'{len(low)} 个主要指数收在近一年最低点附近')
    b = D['breadth']
    if b and b['low20_p'] and b['low20_p'] >= 95:
        parts.append(f"创 20 日新低的股票有 {b['low20']} 家")
    lv, et = D['leverage'], D['etf']
    if lv and lv['chg20_pct'] is not None:
        if lv['chg20_pct'] <= -3:
            money.append(f"融资盘 20 日撤出 {abs(lv['chg20']):.0f} 亿")
        elif lv['chg20_pct'] >= 3:
            money.append(f"融资盘 20 日加仓 {lv['chg20']:.0f} 亿")
    flow = (et or {}).get('d5_total') or 0
    if et and abs(flow) >= 20:
        money.append(f"上交所宽基 ETF 5 日{'净申购' if flow > 0 else '净赎回'}约 {abs(flow):.0f} 亿元")
    tail = '。'
    if lv and lv['chg20_pct'] is not None and lv['chg20_pct'] <= -3 and flow >= 20:
        tail = '——托底的钱在进，杠杆的钱在走。'
    elif lv and lv['chg20_pct'] is not None and lv['chg20_pct'] >= 3 and flow <= -20:
        tail = '——杠杆在加，长线资金在借机减仓。'
    e = D['valuation'].get('ebs')
    val = ''
    if e and e['p_all'] is not None:
        if e['p_all'] >= 80:
            val = f"股债利差在历史 {e['p_all']:.0f}% 分位，拉长看股票相对债券不贵。"
        elif e['p_all'] <= 20:
            val = f"股债利差只在历史 {e['p_all']:.0f}% 分位，股票相对债券偏贵。"
    s = '，'.join(parts)
    s += ('；' + '，'.join(money) + tail) if money else '。'
    return s + val


def takeaways(D):
    t, out = D['today'], []
    r = D['rates']

    def br(x):
        return f"{x['n']} 次，随后 5 日中证1000 上涨概率 {x['win']:.0f}%，中位数 {fmt_pct(x['med'], 2)}" if x else '样本不足'

    if t['temp'] is not None:
        head = f"情绪温度 {t['temp']:.0f}（5 日 {t['dtemp5']:+.0f}），处在「{t['stage']}」阶段。"
        a5 = r['by_temp'].get('k5')
        base = r['all'].get('k5')
        if a5 and base:
            edge = a5['med'] - base['med']
            lean = '略好于' if edge > 0.3 else ('略差于' if edge < -0.3 else '与')
            tail = '无条件基准' if lean != '与' else '无条件基准接近'
            head += f" {D['history_start'][:4]} 年以来温度落在 {r['bucket']} 的交易日共 {br(a5)}，{lean}{tail}（{fmt_pct(base['med'], 2)}）。"
        out.append({'tag': '情绪', 'text': head})

    lm = D['limit']
    if lm:
        top = lm['ladder'][0] if lm['ladder'] else None
        s = f"涨停 {lm['zt']} 家、炸板 {lm['zb']} 家、跌停 {lm['dt']} 家，封板率 {lm['seal_rate']:.0f}%。"
        if top:
            names = '、'.join(x['name'] for x in top['stocks'][:3])
            s += f" 最高 {top['streak']} 连板（{names}）。"
        if t['promo_1to2'] is not None and t['zt_prem'] is not None:
            s += f" 昨日涨停股今天平均 {fmt_pct(t['zt_prem'], 2)}，首板晋级二板 {t['promo_1to2']:.0f}%。"
            if t['zt_prem'] < 0 and t['promo_1to2'] < 20:
                s += ' 接力资金亏钱，打板次日难有溢价。'
            elif t['zt_prem'] > 2 and t['promo_1to2'] > 35:
                s += ' 接力有溢价，短线资金愿意追。'
        main = lm['mainline']['rows'][:3]
        if main:
            s += ' 近 5 日涨停最集中的方向：' + '、'.join(f"{x['ind']}（{x['sum']} 家/{x['days']} 天）" for x in main) + '。'
            if main[0]['days'] <= 2:
                s += ' 题材持续性差，轮动快，没有形成连续几天的主线。'
        out.append({'tag': '涨停', 'text': s})

    dv = D['divergence']
    if dv['hs300_r'] is not None and dv['med_r'] is not None:
        s = f"沪深300 {fmt_pct(dv['hs300_r'], 2)}，全市场个股涨跌幅中位数 {fmt_pct(dv['med_r'], 2)}，上涨 {int(t['up'] or 0)} 家、下跌 {int(t['down'] or 0)} 家。"
        if dv['last20'] >= 5:
            s += f" 近 20 个交易日有 {dv['last20']} 天指数和中位数方向相反，指数不能代表多数股票的体感。"
        out.append({'tag': '宽度', 'text': s})

    st = D['style']
    if st.get('rel20') is not None:
        side = '微盘股跑赢' if st['rel20'] > 0 else '微盘股跑输'
        s = f"{side}沪深300 {abs(st['rel20']):.1f} 个百分点（20 日），处于 2016 年以来 {st['rel20_p']:.0f}% 分位；今年以来微盘 400 累计 {fmt_pct(st['micro_ytd'], 1)}。"
        if st['rel20_p'] is not None and st['rel20_p'] >= 90:
            s += ' 小票相对强势已到极端，历史上这种位置容易出现风格回撤。'
        elif st['rel20_p'] is not None and st['rel20_p'] <= 10:
            s += ' 小票相对弱势到极端，常见于流动性收缩或监管冲击。'
        out.append({'tag': '风格', 'text': s})

    v = D['valuation']
    if 'ebs' in v:
        e = v['ebs']
        s = f"股债利差（沪深300 盈利收益率 − 10 年国债）{e['value']:.2f}%，处于 2005 年以来 {e['p_all']:.0f}% 分位、近十年 {e['p10']:.0f}% 分位。"
        if e['fwd1y'] and e['fwd1y_all']:
            s += (f" 按当时已知数据分档，同处 {e['band']} 分位区间时，沪深300 未来一年上涨概率 {e['fwd1y']['win']:.0f}%、中位数 {fmt_pct(e['fwd1y']['med'], 1)}"
                  f"（全样本 {e['fwd1y_all']['win']:.0f}% / {fmt_pct(e['fwd1y_all']['med'], 1)}；样本按日重叠，独立年份远少于 {e['fwd1y']['n']}）。")
        if 'pe' in v:
            s += f" 全 A 市盈率中位数 {v['pe']['median']}，近十年 {v['pe']['p10']:.0f}% 分位"
        if 'below_net' in v:
            s += f"；破净股 {v['below_net']['n']} 家（{v['below_net']['ratio']}%）。"
        out.append({'tag': '估值', 'text': s})

    lv = D['leverage']
    if lv:
        s = f"融资余额 {lv['balance'] / 1e4:.2f} 万亿（{lv['date']}），20 日 {lv['chg20']:+.0f} 亿（{fmt_pct(lv['chg20_pct'], 1)}）。"
        if lv['ratio'] is not None:
            s += f" 融资买入占两市成交 {lv['ratio']:.1f}%，处于 2016 年以来 {lv['ratio_p']:.0f}% 分位。"
        s += f" 平均维持担保比例 {lv['guarantee']}%（近三年 {lv['guarantee_p']:.0f}% 分位）"
        s += '，偏低时急跌更容易触发强平。' if (lv['guarantee_p'] or 50) < 20 else '。'
        out.append({'tag': '杠杆', 'text': s})

    b = D['breadth']
    if b:
        s = f"{b['date']} 创 20 日新高 {b['high20']} 家、新低 {b['low20']} 家（新低数处于近两年 {b['low20_p']:.0f}% 分位）。"
        if b['low20_p'] and b['low20_p'] >= 90 and b['after_extreme_low']:
            x = b['after_extreme_low']
            s += f" 近两年新低数处在前 10% 的 {x['n']} 天里，上证此后 5 日上涨概率 {x['win']:.0f}%、中位数 {fmt_pct(x['med'], 2)}——普跌之后短线反抽并不少见，但样本只有两年。"
        out.append({'tag': '新高新低', 'text': s})

    et = D['etf']
    if et and et.get('d5_total') is not None:
        tr = sorted(et['tracks'].items(), key=lambda kv: -abs(kv[1]['d5'] or 0))
        s = f"上交所主要宽基 ETF 5 日净申购约 {et['d5_total']:+.0f} 亿元、20 日 {et['d20_total']:+.0f} 亿元（份额变化×最新价估算），"
        s += '其中' + '、'.join(f"{k} {v['d5']:+.0f} 亿" for k, v in tr[:3] if v['d5'] is not None) + '。'
        hs = D['divergence']['hs300_r']
        if et['d5_total'] > 30 and hs is not None and D['indices'][4]['r5'] is not None and D['indices'][4]['r5'] < 0:
            s += '下跌中大额申购宽基 ETF，市场通常理解为中长期资金在托底，主要护的是大盘权重。'
        elif et['d5_total'] < -30:
            s += '宽基 ETF 在被净赎回，托底资金可能在退出。'
        out.append({'tag': 'ETF', 'text': s})
    for x in out:
        x['text'] = x['text'].replace('。 ', '。').replace('） ', '）')
    return out


if __name__ == '__main__':
    main()
