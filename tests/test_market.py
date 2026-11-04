"""market.py 的口径测试：用手算得出答案的小面板核对。  python tests/test_market.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import insights  # noqa: E402
import market  # noqa: E402


def frame(date, code, o, c, pre, zt=0, dt=0, zb=0, cant=0, st=0, mv=100.0, pe=20.0, vol=1000.0):
    return {'date': date, 'code': code, 'st': st, 'open': o, 'high': max(o, c), 'low': min(o, c), 'close': c, 'pre_close': pre, 'volume': vol,
            'amount': vol * (o + c) / 2, 'fmv': mv, 'mv': mv, 'pe': pe, 'pb': 2.0, 'zt': zt, 'dt': dt, 'zb': zb, 'cant': cant}


def panel():
    rows, days = [], [f'2026-01-{d:02d}' for d in range(1, 31)]
    # 26 只普通股票每天不动，保证“上市满 20 日”和样本数够
    for i in range(26):
        for d in days:
            rows.append(frame(d, f'{600000 + i}', 10, 10, 10, mv=100 + i))
    # 三只股票：第 25 天（下标 24）涨停，下标 25 开盘买，下标 26 开盘卖
    for k, code in enumerate(['000001', '000002', '000003']):
        for j, d in enumerate(days):
            if j == 24:
                rows.append(frame(d, code, 10, 11, 10, zt=1))
            elif j == 25:
                # 000003 开盘就涨停，买不进
                rows.append(frame(d, code, 12.1 if k == 2 else 11, 12, 11, cant=1 if k == 2 else 0))
            elif j == 26:
                # 000002 当天除权 1 拆 2：昨收 12 -> 前收盘 6，开盘 6.6 相当于未除权 13.2
                rows.append(frame(d, code, 6.6 if k == 1 else 13.2, 6.6 if k == 1 else 13.2, 6 if k == 1 else 12))
            else:
                px = 11 if j < 24 else (6.6 if k == 1 else 13.2)
                rows.append(frame(d, code, 10 if j < 24 else px, 10 if j < 24 else px, 10 if j < 24 else px))
    return pd.DataFrame(rows), days


def test_t_plus_1():
    df, days = panel()
    b = market.Block(df)
    r = market.day_stats(b, 26)
    assert b.streak[24][list(b.codes).index('000001')] == 1
    # 买入价 11，卖出 13.2，收益 20%，000002 除权还原后同样 20%；000003 买不进不计入。样本不足 3 只时应为 NaN
    assert r['n_ban1'] == 2, r['n_ban1']
    assert np.isnan(r['p_ban1'])
    # 放宽样本后核对数值：再加一只同样的股票
    extra = df[df['code'] == '000001'].copy()
    extra['code'] = '000004'
    b2 = market.Block(pd.concat([df, extra], ignore_index=True))
    r2 = market.day_stats(b2, 26)
    assert r2['n_ban1'] == 3
    assert abs(r2['p_ban1'] - (20 - market.COST)) < 1e-3, r2['p_ban1']


def test_cost_and_ma():
    rows = []
    days = [f'2026-{m:02d}-{d:02d}' for m in range(1, 5) for d in range(1, 21)]
    for i in range(210):
        for j, d in enumerate(days):
            # 前 40 天 10 元，之后 11 元；成交量相同，60 日成本可手算
            c = 10.0 if j < 40 else 11.0
            pre = 10.0 if j <= 40 else 11.0
            rows.append(frame(d, f'{600000 + i}', c, c, pre, mv=50 + i))
    b = market.Block(pd.DataFrame(rows))
    r = market.day_stats(b, 69)
    # 最近 60 天（下标 10~69）：前 30 天 10 元，后 30 天 11 元，成交量相等 -> 平均成本 10.5
    cost = (10 * 30 + 11 * 30) / 60
    assert abs(r['pl60'] - (11 / cost - 1) * 100) < 0.01, r['pl60']
    assert r['ma20'] == 0 and r['ma60'] == 100, (r['ma20'], r['ma60'])
    assert 'dd250' not in r  # 上市不足 120 天的不算回撤


def test_cooldown():
    m = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)
    got = np.where(insights.cooldown(m, gap=10))[0].tolist()
    assert got == [0, 11], got


def test_size_grid():
    rows = []
    days = [f'2026-01-{d:02d}' for d in range(1, 25)]
    for i in range(400):
        for j, d in enumerate(days):
            c = 10.0 if j < 23 else (11.0 if i < 200 else 10.0)
            rows.append(frame(d, f'{600000 + i}', 10, c, 10, mv=20 if i < 200 else 2000, pe=10 + i * 0.1))
    b = market.Block(pd.DataFrame(rows))
    r = market.day_stats(b, 23)
    # 小票（第 0 档）今天都涨 10%，千亿以上（第 5 档）不动
    vals = [r[f'g0{v}'] for v in range(4) if not np.isnan(r[f'g0{v}'])]
    assert vals and all(abs(x - 10) < 1e-3 for x in vals), vals
    assert all(abs(r[f'g5{v}']) < 1e-3 for v in range(4) if not np.isnan(r[f'g5{v}']))
    assert abs(r['f_size'] - 10) < 1e-3, r['f_size']


if __name__ == '__main__':
    fails = 0
    for k, fn in list(globals().items()):
        if k.startswith('test_'):
            try:
                fn()
                print('[OK]', k)
            except AssertionError as e:
                fails += 1
                print('[FAIL]', k, e)
    sys.exit(fails)
