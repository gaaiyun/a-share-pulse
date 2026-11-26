"""拟合 A 股日度纯风格因子收益。运行 python scripts/style.py 回填缺日并查看统计。"""
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from market import Block, DATA, SPOT, nanmean, sector_index

FACTORS = {
    'SIZE': '大市值',
    'NLSIZE': '非线性市值',
    'BETA': '高贝塔',
    'MOM': '强势股',
    'RESVOL': '高波动',
    'LIQ': '高换手',
    'BTOP': '低市净率',
    'EYLD': '高盈利收益率',
}
WINDOW = 250
YEAR = 245
COLS = ['date', *FACTORS, 'r2']


def ratio(num, den):
    """无效分母留空。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        value = np.asarray(num, dtype=float) / den
    return np.where(np.isfinite(value), value, np.nan)


def beta(returns, market):
    """各股在同一历史窗口内对市场收益的带截距回归。"""
    good = np.isfinite(returns) & np.isfinite(market[:, None])
    x, y = np.where(good, market[:, None], 0), np.where(good, returns, 0)
    n = np.maximum(1, good.sum(axis=0))
    var = (x * x).sum(axis=0) - x.sum(axis=0) ** 2 / n
    cov = (x * y).sum(axis=0) - x.sum(axis=0) * y.sum(axis=0) / n
    return np.where((n >= 120) & (var > 0), ratio(cov, var), np.nan)


def winsor(x):
    mid = np.median(x)
    mad = np.median(np.abs(x - mid))
    return np.clip(x, mid - 3 * mad, mid + 3 * mad)


def orthogonal(x, controls, cap):
    root = np.sqrt(cap / cap.mean())
    design = np.column_stack((np.ones(len(x)), *controls))
    coef = np.linalg.lstsq(design * root[:, None], x * root, rcond=None)[0]
    return x - design @ coef


def exposures(b, i, sector):
    """收益日 i 的暴露与可用样本只使用 i-1 及更早的面板。"""
    e = np.full((len(b.codes), len(FACTORS)), np.nan)
    s = i - 1
    if s < WINDOW:
        return e
    book, earnings = ratio(1, b.pb[s]), ratio(1, b.pe[s])
    if not (np.isfinite(book) & np.isfinite(earnings)).any():
        return e
    ret = b.ret[s - WINDOW + 1:i].astype(float)
    prev_cap = b.fmv[s - WINDOW:s].astype(float)
    good = np.isfinite(ret) & np.isfinite(prev_cap) & (prev_cap > 0)
    weight = np.where(good, prev_cap, 0)
    mkt = ratio((np.where(good, ret, 0) * weight).sum(axis=1), weight.sum(axis=1))
    fmv = b.fmv[s].astype(float)
    mv = b.mv[s].astype(float)
    turn = ratio(b.amount[i - 120:i], np.where(b.fmv[i - 120:i] > 0, b.fmv[i - 120:i] * 1e8, np.nan))
    r60 = np.where(np.isfinite(ret[-60:]), ret[-60:], np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        size = np.log(np.where(mv > 0, mv, np.nan))
        mom = ratio(b.adj[s - 20], b.adj[s - WINDOW]) - 1
        vol = np.sqrt(nanmean((r60 - nanmean(r60, axis=0)) ** 2, axis=0))
        vol[np.isfinite(r60).sum(axis=0) < 2] = np.nan
        liq = (np.log(nanmean(turn[-20:], axis=0)) + np.log(nanmean(turn, axis=0))) / 2
    raw = np.column_stack((size, size ** 3, beta(ret, mkt), mom, vol, liq, book, earnings))
    ok = b.elig(s, age=60) & (b.st[s] == 0) & (sector >= 0) & (fmv > 0) & np.isfinite(fmv)
    ok &= np.isfinite(raw).all(axis=1)
    if not ok.any():
        return e

    x, cap = raw[ok], fmv[ok]
    for j in (0, 2, 3, 6, 7):
        x[:, j] = winsor(x[:, j])
    # 在残差上缩尾，再投影一次，保留缩尾后的截面正交性。
    for j, controls in ((1, (x[:, 0],)), (4, (x[:, 2], x[:, 0])), (5, (x[:, 0],))):
        resid = orthogonal(x[:, j], controls, cap)
        x[:, j] = orthogonal(winsor(resid), controls, cap)
    scale = x.std(axis=0)
    if np.all(scale > np.finfo(float).eps):
        e[ok] = (x - np.average(x, axis=0, weights=cap)) / scale
    return e


def fit_wls(returns, exposure, cap, sector):
    """回归权重为流通市值平方根，行业约束按流通市值加权。"""
    groups, idx = np.unique(sector, return_inverse=True)
    if len(groups) == 0 or len(returns) <= len(groups) + exposure.shape[1]:
        return None
    share = np.bincount(idx, weights=cap)
    ref = int(share.argmax())
    keep = np.arange(len(groups)) != ref
    dummy = idx[:, None] == np.arange(len(groups))
    industry_design = dummy[:, keep] - dummy[:, [ref]] * (share[keep] / share[ref])
    design = np.column_stack((np.ones(len(returns)), industry_design, exposure))
    weight = np.sqrt(cap)
    root = np.sqrt(weight / weight.mean())
    coef, _, rank, _ = np.linalg.lstsq(design * root[:, None], returns * root, rcond=None)
    if rank < design.shape[1]:
        return None
    ind = np.zeros(len(groups))
    ind[keep] = coef[1:len(groups)]
    ind[ref] = -np.dot(ind[keep], share[keep]) / share[ref]
    err = returns - design @ coef
    tot = np.dot(weight, (returns - np.average(returns, weights=weight)) ** 2)
    if tot <= 0:
        return None
    return coef[0], ind, coef[len(groups):], 1 - np.dot(weight, err ** 2) / tot


def day_stats(b, i, sector):
    e = exposures(b, i, sector)
    ret = b.ret[i].astype(float)
    cap = b.fmv[i - 1].astype(float)
    used = np.isfinite(e).all(axis=1) & np.isfinite(ret)
    fit = fit_wls(ret[used], e[used], cap[used], sector[used])
    if fit is not None:
        _, _, style, r2 = fit
        return dict(zip(COLS, [str(b.dates[i]), *(style * 100), r2]))


def update(spot=SPOT, batch_size=40):
    """回填和增量用相同窗口及回归，保留已写日期。"""
    path = DATA / 'style_daily.csv'
    old = pd.read_csv(path) if path.exists() else None
    have = set(old['date']) if old is not None else set()
    files = sorted(spot.glob('????-??-??.csv.gz'))
    todo = [i for i in range(WINDOW + 1, len(files)) if files[i].name[:10] not in have]
    rows = []
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        first = batch[0] - WINDOW - 1
        frames = [pd.read_csv(p, dtype={'code': str}).assign(date=p.name[:10])
                  for p in files[first:batch[-1] + 1]]
        b = Block(pd.concat(frames, ignore_index=True))
        del frames
        sector, _ = sector_index(b.codes)
        for i in batch:
            row = day_stats(b, i - first, sector)
            if row is not None:
                rows.append(row)
    if rows:
        new = pd.DataFrame(rows, columns=COLS)
        out = new if old is None else pd.concat((old, new), ignore_index=True)
        with NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=DATA, delete=False) as stream:
            out.sort_values('date').to_csv(stream, index=False, float_format='%.8f', lineterminator='\n')
        Path(stream.name).replace(path)
        print('风格因子追加', len(rows), '天')
    return len(rows)


def recent_stats(daily, days=YEAR):
    """最近若干有效日，按 245 日线性年化，t 用 5 阶 Newey–West。"""
    x = daily.tail(days)[list(FACTORS)].to_numpy(dtype=float)
    n = len(x)
    avg = x.mean(axis=0)
    u = x - avg
    long_var = (u * u).mean(axis=0)
    for lag in range(1, min(5, n - 1) + 1):
        long_var += 2 * (1 - lag / 6) * (u[lag:] * u[:-lag]).sum(axis=0) / n
    se = np.sqrt(np.maximum(long_var, 0) / n)
    return avg * YEAR, ratio(avg, se)


if __name__ == '__main__':
    update()
    path = DATA / 'style_daily.csv'
    if path.exists():
        daily = pd.read_csv(path)
        annual, t = recent_stats(daily)
        print(f"{daily['date'].iloc[0]} 至 {daily['date'].iloc[-1]}，{len(daily)} 日，平均 R2 {daily['r2'].mean():.2%}")
        print(f'近一年可得 {min(len(daily), YEAR)} 日，年化 / Newey-West t')
        for name, a, v in zip(FACTORS, annual, t):
            print(f'{name:7} {a:+7.2f}% / {v:+.2f}')
