"""把每日个股快照（data/spot/日期.csv.gz）算成全市场聚合指标。页面只用聚合结果，不展示个股。

    python scripts/market.py            # 把快照里还没算过的交易日追加到 data/market_daily.csv、data/sector_daily.csv

历史回填和每日增量走同一套函数（Block + day_stats + sector_stats），口径不会分叉。
打法收益按 A 股 T+1 算：信号日 S 收盘选股 → S+1 开盘买入（开盘即涨停买不进）→ S+2 开盘卖出，扣 0.2% 成本。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
SPOT = DATA / 'spot'
COLS = ['code', 'st', 'open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'fmv', 'mv', 'pe', 'pb', 'zt', 'dt', 'zb', 'cant']
COST = 0.2      # 一买一卖的佣金、印花税、滑点，%
WINDOW = 270    # 每日增量读多少天快照，够算 250 日指标
KEEP = 300      # 仓库里最多留多少天快照
SIZE_EDGES = [30, 50, 100, 300, 1000]  # 亿元
SIZE_NAMES = ['30亿以下', '30-50亿', '50-100亿', '100-300亿', '300-1000亿', '千亿以上']
VALUE_NAMES = ['亏损', '低PE', '中PE', '高PE']
PLAYS = {  # 打法：键 -> (名称, 定义)
    'ban1': ('首板接力', '昨天首次涨停'),
    'ban2': ('连板接力', '昨天 2 连板以上'),
    'zb': ('炸板反包', '昨天摸过涨停没封住'),
    'chase': ('追涨', '昨天涨 5% 以上、没涨停'),
    'brk': ('放量突破', '昨天创 60 日新高、成交额翻倍'),
    'dip': ('抄底大跌', '昨天跌 5% 以上、没跌停'),
    'dt': ('跌停博反弹', '昨天跌停'),
    'over': ('超跌低吸', '近 5 日累计跌超 15%'),
    'hot': ('热门股', '昨天成交额前 100'),
    'micro': ('微盘', '市值最小 400 只'),
    'large': ('大票', '市值最大 300 只'),
    'lowpe': ('低估值', 'PE 最低 500 只（盈利股）'),
    'all': ('全市场', '所有非 ST、上市满 20 日的股票'),
}
FACTORS = {'size': '小市值', 'value': '低估值', 'lowvol': '低波动', 'mom': '强势股', 'rev': '超跌反转', 'lowturn': '低换手', 'lowprice': '低价股'}


def nanmean(a, axis=None):
    """全 NaN 时不报警告。"""
    m = ~np.isnan(a)
    n = m.sum(axis=axis)
    s = np.where(m, a, 0).sum(axis=axis)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


class Block:
    """若干交易日的个股面板：行 = 交易日，列 = 股票。"""

    def __init__(self, df):
        self.dates = np.sort(df['date'].unique())
        self.codes, cidx = np.unique(df['code'].astype(str).values, return_inverse=True)
        tidx = np.searchsorted(self.dates, df['date'].values)
        shape = (len(self.dates), len(self.codes))
        for c in COLS[1:]:
            a = np.full(shape, np.nan, dtype=np.float32)
            a[tidx, cidx] = pd.to_numeric(df[c], errors='coerce').values.astype(np.float32)
            setattr(self, c, a)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.ret = self.close / self.pre_close - 1
            r = np.nan_to_num(self.close.astype(np.float64) / self.pre_close, nan=1.0, posinf=1.0, neginf=1.0)
        self.adj = np.cumprod(np.clip(r, 0.2, 5), axis=0)
        self.adj[np.isnan(self.close)] = np.nan
        # 上市（出现在快照里）的累计天数：用来排除次新
        self.age = np.cumsum(~np.isnan(self.close), axis=0)
        st = np.zeros(shape, dtype=np.int16)
        z = np.nan_to_num(self.zt) > 0
        for t in range(shape[0]):
            st[t] = (st[t - 1] + 1) * z[t] if t else z[t]
        self.streak = st

    def elig(self, i, age=20):
        return ~np.isnan(self.close[i]) & (np.nan_to_num(self.st[i]) == 0) & (self.age[i] >= age)

    def win(self, a, i, n):
        return a[max(0, i - n + 1):i + 1]


def _rank(a, n, largest=False):
    """取最小（或最大）的 n 个，返回布尔掩码；NaN 不参与。"""
    m = np.zeros(a.shape, dtype=bool)
    idx = np.where(~np.isnan(a))[0]
    if len(idx) == 0:
        return m
    order = idx[np.argsort(a[idx])]
    m[order[-n:] if largest else order[:n]] = True
    return m


def play_masks(b, s):
    """信号日 s 的打法选股。"""
    ok = b.elig(s)
    ret, zt, dt, zb = b.ret[s], np.nan_to_num(b.zt[s]) > 0, np.nan_to_num(b.dt[s]) > 0, np.nan_to_num(b.zb[s]) > 0
    adj, amt = b.adj, b.amount
    m = {'ban1': zt & (b.streak[s] == 1), 'ban2': zt & (b.streak[s] >= 2), 'zb': zb, 'dt': dt,
         'chase': (ret >= .05) & ~zt, 'dip': (ret <= -.05) & ~dt}
    w60 = b.win(adj, s, 60)
    with np.errstate(invalid='ignore', divide='ignore'):
        hi60 = np.nanmax(np.where(np.isnan(w60), -np.inf, w60), axis=0)
        amt20 = nanmean(b.win(amt, s - 1, 20).astype(np.float64), axis=0) if s >= 1 else np.full(ret.shape, np.nan)
        m['brk'] = (adj[s] >= hi60 - 1e-9) & (amt[s] >= 2 * amt20) & (b.age[s] >= 60)
        m['over'] = (adj[s] / adj[s - 5] - 1 <= -.15) if s >= 5 else np.zeros(ret.shape, dtype=bool)
    m['hot'] = _rank(np.where(ok, amt[s], np.nan), 100, largest=True)
    mv = np.where(ok, b.mv[s], np.nan)
    m['micro'], m['large'] = _rank(mv, 400), _rank(mv, 300, largest=True)
    pe = np.where(ok & (b.pe[s] > 0), b.pe[s], np.nan)
    m['lowpe'] = _rank(pe, 500)
    m['all'] = np.ones(ret.shape, dtype=bool)
    return {k: v & ok for k, v in m.items()}


def _spread(sig, ret, ok):
    m = ok & ~np.isnan(sig) & ~np.isnan(ret)
    n = int(m.sum())
    if n < 200:
        return np.nan
    s, r = sig[m], ret[m]
    k, order = n // 5, np.argsort(s)
    return float(r[order[-k:]].mean() - r[order[:k]].mean()) * 100


def day_stats(b, i):
    """交易日 i 的全市场聚合。所有分组都只用 i 之前（含信号日）已知的数据。"""
    rec = {'date': str(b.dates[i])}
    O, H, C, P = b.open[i], b.high[i], b.close[i], b.pre_close[i]
    ok = b.elig(i)
    with np.errstate(invalid='ignore', divide='ignore'):
        iret = C / O - 1
        rec['open_win'] = float(np.mean(iret[ok] > 0)) * 100 if ok.any() else np.nan
        rec['open_ret'] = float(nanmean(iret[ok])) * 100
        rec['fade'] = float(nanmean((C / H - 1)[ok])) * 100

    # ---- 打法（T+1）：s = i-2 选股，b1 = i-1 开盘买，i 开盘卖
    if i >= 2:
        s, b1 = i - 2, i - 1
        with np.errstate(invalid='ignore', divide='ignore'):
            r = O * (b.close[b1] / P) / b.open[b1] - 1  # 卖出价按除权比例还原
        can = (np.nan_to_num(b.cant[b1]) == 0) & (b.open[b1] > 0) & ~np.isnan(r)
        for k, m in play_masks(b, s).items():
            x = r[m & can]
            rec['p_' + k] = float(x.mean()) * 100 - COST if len(x) >= 3 else np.nan
            rec['n_' + k] = int(len(x))

    # ---- 今天开盘买、收盘浮盈：昨天选股，反映今天的体感（还没扣成本，也还不能卖）
    if i >= 1:
        with np.errstate(invalid='ignore', divide='ignore'):
            q = C / O - 1
        can = (np.nan_to_num(b.cant[i]) == 0) & (O > 0) & ~np.isnan(q)
        for k, m in play_masks(b, i - 1).items():
            x = q[m & can]
            rec['q_' + k] = float(x.mean()) * 100 if len(x) >= 3 else np.nan

    # ---- 连板晋级：昨天 k 板 -> 今天还封住的比例
    if i >= 1:
        yes = b.streak[i - 1]
        oky = b.elig(i - 1, age=60)
        z = np.nan_to_num(b.zt[i]) > 0
        for k in (1, 2, 3, 4):
            mk = oky & (yes == k if k < 4 else yes >= 4)
            rec[f'fn{k}'] = int(mk.sum())
            rec[f'fp{k}'] = float(z[mk].mean()) * 100 if mk.any() else np.nan

    # ---- 全市场站上均线、持仓成本、距高点回撤（只看非 ST、上市满 20 日）
    adj, A = b.adj, b.adj[i]
    for n in (20, 60, 250):
        w = b.win(adj, i, n)
        enough = (~np.isnan(w)).sum(axis=0) >= max(int(n * .8), 2)
        ma = nanmean(w, axis=0)
        mk = ok & enough & ~np.isnan(ma)
        with np.errstate(invalid='ignore'):
            rec[f'ma{n}'] = float((A[mk] > ma[mk]).mean()) * 100 if mk.sum() >= 200 else np.nan
        # 近 n 日买入者的平均成本 = 成交额之和 ÷ 复权成交量之和
        wa = b.win(b.amount, i, n).astype(np.float64)
        wc = b.win(b.close, i, n).astype(np.float64)
        wv = b.win(b.volume, i, n).astype(np.float64)
        with np.errstate(invalid='ignore', divide='ignore'):
            shares = wv * wc / w  # 除以复权系数 adj/close
            cost = np.nansum(wa, axis=0) / np.nansum(shares, axis=0)
            pl = A / cost - 1
        mk2 = mk & np.isfinite(pl)
        rec[f'pl{n}'] = float(np.median(pl[mk2])) * 100 if mk2.sum() >= 200 else np.nan
        rec[f'plw{n}'] = float((pl[mk2] > 0).mean()) * 100 if mk2.sum() >= 200 else np.nan
    w250 = b.win(adj, i, 250)
    with np.errstate(invalid='ignore', divide='ignore'):
        hi = np.nanmax(np.where(np.isnan(w250), -np.inf, w250), axis=0)
        dd = A / hi - 1
    mk = ok & np.isfinite(dd) & (b.age[i] >= 120)
    if mk.sum() >= 200:
        rec['dd250'] = float(np.median(dd[mk])) * 100
        rec['dd50'] = float((dd[mk] <= -.5).mean()) * 100
    # 年内：相对上年最后一个交易日
    yr = str(b.dates[i])[:4]
    j = int(np.searchsorted(b.dates, yr + '-01-01'))
    if 0 < j <= i:
        with np.errstate(invalid='ignore', divide='ignore'):
            yt = A / adj[j - 1] - 1
        mk = ok & np.isfinite(yt)
        if mk.sum() >= 200:
            rec['ytd_med'] = float(np.median(yt[mk])) * 100
            rec['ytd_win'] = float((yt[mk] > 0).mean()) * 100

    # ---- 分组、因子：按昨天已知的市值和估值分组，看今天的涨跌
    if i >= 1:
        p = i - 1
        okp = b.elig(p)
        r = b.ret[i]
        mv, pe = b.mv[p], b.pe[p]
        with np.errstate(invalid='ignore'):
            sz = np.searchsorted(SIZE_EDGES, np.nan_to_num(mv, nan=-1), side='right')
            pos = pe > 0
            v = np.full(r.shape, -1, dtype=int)
            v[pe <= 0] = 0
            if pos.sum() >= 300:
                q1, q2 = np.nanpercentile(pe[pos], [100 / 3, 200 / 3])
                v[pos & (pe <= q1)] = 1
                v[pos & (pe > q1) & (pe <= q2)] = 2
                v[pos & (pe > q2)] = 3
        good = okp & ~np.isnan(r) & ~np.isnan(mv)
        for si in range(6):
            for vi in range(4):
                mk = good & (sz == si) & (v == vi)
                rec[f'g{si}{vi}'] = float(r[mk].mean()) * 100 if mk.sum() >= 5 else np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            ep = np.where(pe > 0, 1 / pe, np.where(pe <= 0, -1.0, np.nan))
            vol = np.nanstd(b.win(b.ret, p, 20).astype(np.float64), axis=0) if p >= 19 else np.full(r.shape, np.nan)
            mom = adj[p] / adj[p - 20] - 1 if p >= 20 else np.full(r.shape, np.nan)
            rev = adj[p] / adj[p - 5] - 1 if p >= 5 else np.full(r.shape, np.nan)
            tn = nanmean(b.win(b.amount, p, 20).astype(np.float64), axis=0) / (b.fmv[p].astype(np.float64) * 1e8)
        for k, sig in (('size', -mv), ('value', ep), ('lowvol', -vol), ('mom', mom), ('rev', -rev), ('lowturn', -tn), ('lowprice', -b.close[p])):
            rec['f_' + k] = _spread(sig, r, okp)
        with np.errstate(invalid='ignore'):
            w = np.where(good, mv, np.nan)
        rec['cap_ret'] = float(np.nansum(w * r) / np.nansum(w)) * 100 if np.nansum(w) > 0 else np.nan
    return rec


def sector_stats(b, i, sec, names):
    """申万一级聚合：市值加权与等权收益、成交占比、上涨占比、涨跌停、整体法 PE/PB。"""
    C, r = b.close[i], b.ret[i]
    p = max(i - 1, 0)
    w = np.nan_to_num(b.mv[p]).astype(np.float64)
    valid = ~np.isnan(C) & ~np.isnan(r) & (sec >= 0)
    tot_amt = np.nansum(b.amount[i][sec >= 0].astype(np.float64))
    rows = []
    for k, name in enumerate(names):
        m = valid & (sec == k)
        if m.sum() < 5:
            continue
        ww = w[m]
        mv = np.nan_to_num(b.mv[i][m]).astype(np.float64)
        pe, pb = b.pe[i][m].astype(np.float64), b.pb[i][m].astype(np.float64)
        ok_pe, ok_pb = ~np.isnan(pe) & (pe != 0), ~np.isnan(pb) & (pb > 0)
        e = (mv[ok_pe] / pe[ok_pe]).sum()
        bk = (mv[ok_pb] / pb[ok_pb]).sum()
        rows.append({'date': str(b.dates[i]), 'sw1': name, 'n': int(m.sum()),
                     'ret': float((r[m] * ww).sum() / ww.sum()) * 100 if ww.sum() > 0 else np.nan,
                     'eq': float(r[m].mean()) * 100, 'up': float((r[m] > 0).mean()) * 100,
                     'amt': float(np.nansum(b.amount[i][m].astype(np.float64)) / tot_amt) * 100 if tot_amt > 0 else np.nan,
                     'zt': int(np.nan_to_num(b.zt[i][m]).sum()), 'dt': int(np.nan_to_num(b.dt[i][m]).sum()),
                     'pe': mv[ok_pe].sum() / e if e > 0 else np.nan, 'pb': mv[ok_pb].sum() / bk if bk > 0 else np.nan,
                     'mv': mv.sum()})
    return rows


def latest_extra(b, i):
    """只给最新一天的分布：年内涨跌、距高点回撤。"""
    ok = b.elig(i)
    A = b.adj[i]
    out = {'date': str(b.dates[i])}
    yr = str(b.dates[i])[:4]
    j = int(np.searchsorted(b.dates, yr + '-01-01'))
    if 0 < j <= i:
        with np.errstate(invalid='ignore', divide='ignore'):
            yt = (A / b.adj[j - 1] - 1) * 100
        y = yt[ok & np.isfinite(yt)]
        if len(y) >= 200:
            edges = [-1e9, -30, -20, -10, 0, 10, 20, 50, 1e9]
            out['ytd'] = {'labels': ['跌超30%', '-30~-20', '-20~-10', '-10~0', '0~10', '10~20', '20~50', '涨超50%'],
                          'counts': np.histogram(y, bins=edges)[0].tolist(), 'n': int(len(y)),
                          'q': [round(float(x), 2) for x in np.percentile(y, range(0, 101))]}
    w = b.win(b.adj, i, 250)
    with np.errstate(invalid='ignore', divide='ignore'):
        dd = (A / np.nanmax(np.where(np.isnan(w), -np.inf, w), axis=0) - 1) * 100
    d = dd[ok & np.isfinite(dd) & (b.age[i] >= 120)]
    if len(d) >= 200:
        out['dd'] = {'labels': ['跌超70%', '-70~-50', '-50~-40', '-40~-30', '-30~-20', '-20~-10', '-10~0'],
                     'counts': np.histogram(d, bins=[-1e9, -70, -50, -40, -30, -20, -10, 0.001])[0].tolist(), 'n': int(len(d))}
    return out


def sector_index(codes):
    """把股票代码映射到申万一级下标；没有映射的（新股）是 -1。"""
    imap = pd.read_csv(DATA / 'industry_sw1.csv', dtype={'code': str})
    names = sorted(imap['sw1'].unique())
    pos = {n: k for k, n in enumerate(names)}
    d = dict(zip(imap['code'], imap['sw1'].map(pos)))
    return np.array([d.get(str(c), -1) for c in codes]), names


def read_spot(n=WINDOW):
    files = sorted(SPOT.glob('*.csv.gz'))[-n:]
    frames = []
    for f in files:
        x = pd.read_csv(f, dtype={'code': str})
        x['date'] = f.name[:10]
        frames.append(x)
    return pd.concat(frames, ignore_index=True) if frames else None


def round_df(d, n=3):
    num = d.select_dtypes('number').columns
    d[num] = d[num].round(n)
    return d


def update():
    """把快照里还没算过的交易日追加进聚合表；已有的行不动。"""
    df = read_spot()
    if df is None or df['date'].nunique() < 30:
        print('快照不足，跳过')
        return 0
    daily_p, sec_p = DATA / 'market_daily.csv', DATA / 'sector_daily.csv'
    old = pd.read_csv(daily_p) if daily_p.exists() else pd.DataFrame(columns=['date'])
    olds = pd.read_csv(sec_p) if sec_p.exists() else pd.DataFrame(columns=['date', 'sw1'])
    have = set(old['date'].astype(str))
    b = Block(df)
    sec, names = sector_index(b.codes)
    todo = [i for i, d in enumerate(b.dates) if str(d) not in have and i >= 2]
    if not have:  # 没有历史回填时只算窗口足够长的日子
        todo = [i for i in todo if i >= 250]
    rows, srows = [], []
    for i in todo:
        rows.append(day_stats(b, i))
        srows.extend(sector_stats(b, i, sec, names))
    if rows:
        new = round_df(pd.DataFrame(rows))
        out = pd.concat([old, new], ignore_index=True).drop_duplicates('date', keep='last').sort_values('date')
        out.to_csv(daily_p, index=False, encoding='utf-8')
        snew = round_df(pd.DataFrame(srows))
        sout = pd.concat([olds, snew], ignore_index=True).drop_duplicates(['date', 'sw1'], keep='last').sort_values(['date', 'sw1'])
        sout.to_csv(sec_p, index=False, encoding='utf-8')
    (DATA / 'raw').mkdir(exist_ok=True)
    (DATA / 'raw' / 'market_latest.json').write_text(json.dumps(latest_extra(b, len(b.dates) - 1), ensure_ascii=False), encoding='utf-8')
    for f in sorted(SPOT.glob('*.csv.gz'))[:-KEEP]:
        f.unlink()
    print('聚合追加', len(rows), '天，最新', b.dates[-1])
    return len(rows)


if __name__ == '__main__':
    sys.exit(0 if update() >= 0 else 1)
