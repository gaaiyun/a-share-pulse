"""盘中快照：交易时段每 10 分钟取一次全市场截面，写 live.json 推到 live 分支并触发页面部署。

    python scripts/intraday.py          # 一直跑到 15:03
    python scripts/intraday.py --once   # 只跑一轮（测试用，非交易时段也出结果，不推送）

这个任务坏了不影响收盘数据：live.json 只在页面的“盘中”版块使用，收盘数据进来后版块自动隐藏。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

from collect import retry, sina_spot, spot_metrics, trade_days
import market

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
REPO = os.environ.get('GITHUB_REPOSITORY', 'gaaiyun/a-share-pulse')
WATCH = {'上证指数': 'sh000001', '深证成指': 'sz399001', '创业板指': 'sz399006', '科创50': 'sh000688', '沪深300': 'sh000300', '中证500': 'sh000905', '中证1000': 'sh000852'}
PLAY_SHOW = ['ban1', 'ban2', 'zb', 'chase', 'dip', 'over', 'hot', 'micro', 'large', 'all']


def now_bj():
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=8)


def yesterday_masks():
    """用快照算昨天收盘选出的打法名单（只算一次）。"""
    df = market.read_spot(70)
    if df is None:
        return None, None
    b = market.Block(df)
    return b.codes, market.play_masks(b, len(b.dates) - 1)


def indices():
    x = retry(ak.stock_zh_index_spot_sina, n=2)
    x = x.set_index('代码')
    out = []
    for name, code in WATCH.items():
        if code in x.index:
            out.append({'name': name, 'close': round(float(x.at[code, '最新价']), 2), 'ret': round(float(x.at[code, '涨跌幅']), 2)})
    return out


def snapshot(codes, masks, imap):
    spot = sina_spot()
    if len(spot) < 4000:
        raise RuntimeError(f'截面只有 {len(spot)} 行')
    m, s = spot_metrics(spot)
    s = s[s['open'] > 0].copy()
    s['iret'] = s['close'] / s['open'] - 1
    t = now_bj().strftime('%H:%M')
    row = {'t': t, 'up': m['up'], 'down': m['down'], 'zt': m['zt'], 'dt': m['dt'], 'zb': m['zb'], 'med': round(m['med_ret'], 2),
           'amount': m['amount_yi'], 'up_ratio': round(m['up'] / max(m['up'] + m['down'], 1) * 100, 1),
           'open_win': round(float((s['iret'] > 0).mean()) * 100, 1)}
    plays = []
    if masks is not None:
        pos = pd.Series(np.arange(len(codes)), index=codes)
        ir = s.set_index('code')['iret']
        cant = s.set_index('code')['cant']
        for k in PLAY_SHOW:
            chosen = codes[masks[k]]
            x = ir.reindex(chosen).dropna()
            x = x[~cant.reindex(x.index).fillna(False).astype(bool)]
            if len(x) >= 3:
                plays.append({'key': k, 'name': market.PLAYS[k][0], 'v': round(float(x.mean()) * 100, 2), 'n': len(x)})
        plays.sort(key=lambda r: -r['v'])
    x = s.merge(imap, on='code', how='left').dropna(subset=['sw1'])
    x['w'] = x['mv_wan'] / (1 + x['ret'])
    sec = [{'name': n, 'ret': round(float((g['ret'] * g['w']).sum() / g['w'].sum()) * 100, 2), 'up': round(float((g['ret'] > 0).mean()) * 100)}
           for n, g in x.groupby('sw1') if len(g) >= 5]
    sec.sort(key=lambda r: -r['ret'])
    return row, plays, sec[:5] + sec[-3:]


def read_live():
    """取 live 分支上已有的文件：当天的接着追加，昨天的留作“同刻对比”。"""
    try:
        out = subprocess.run(['curl', '-fsSL', f'https://raw.githubusercontent.com/{REPO}/live/live.json?{int(time.time())}'], capture_output=True, timeout=30)
        return json.loads(out.stdout) if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def publish(payload):
    token = os.environ.get('GH_TOKEN')
    if not token:
        print('没有 GH_TOKEN，只写本地')
        return
    with tempfile.TemporaryDirectory() as d:
        Path(d, 'live.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        env = {**os.environ, 'GIT_AUTHOR_NAME': 'a-share-pulse-bot', 'GIT_AUTHOR_EMAIL': 'a-share-pulse-bot@users.noreply.github.com',
               'GIT_COMMITTER_NAME': 'a-share-pulse-bot', 'GIT_COMMITTER_EMAIL': 'a-share-pulse-bot@users.noreply.github.com'}
        run = lambda *c: subprocess.run(c, cwd=d, env=env, check=True, capture_output=True)
        run('git', 'init', '-q', '-b', 'live')
        run('git', 'add', 'live.json')
        run('git', 'commit', '-q', '-m', 'chore(盘中): 快照')
        run('git', 'push', '-q', '-f', f'https://x-access-token:{token}@github.com/{REPO}.git', 'live')
    subprocess.run(['gh', 'workflow', 'run', 'pages.yml', '-R', REPO, '--ref', 'main'], check=True, capture_output=True)


def round_once(state, codes, masks, imap, push=True):
    now = now_bj()
    row, plays, sec = snapshot(codes, masks, imap)
    try:
        idx = indices()
    except Exception as e:  # noqa: BLE001
        print('指数失败', str(e)[:120])
        idx = state.get('indices', [])
    today = now.date().isoformat()
    if state.get('date') != today:
        state.update({'prev': state.get('series', []), 'date': today, 'series': []})
    state['series'] = [x for x in state['series'] if x['t'] != row['t']] + [row]
    vs = None
    prev = [x for x in state.get('prev', []) if x['t'] <= row['t']]
    if prev and prev[-1]['amount']:
        vs = round((row['amount'] / prev[-1]['amount'] - 1) * 100, 1)
    state.update({'at': now.strftime('%H:%M'), 'plays': plays, 'sectors': sec, 'indices': idx, 'amount_vs': vs})
    if push:
        publish(state)
    print(state['at'], row, 'plays', plays[:2], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    now = now_bj()
    cal = trade_days()
    imap = pd.read_csv(DATA / 'industry_sw1.csv', dtype={'code': str})
    codes, masks = yesterday_masks()
    state = read_live() or {}
    if a.once:
        round_once(state, codes, masks, imap, push=False)
        print(json.dumps(state, ensure_ascii=False)[:1500])
        return 0
    if now.date() not in cal:
        print('今天不是交易日')
        return 0
    fails = 0
    while True:
        now = now_bj()
        hm = now.hour * 100 + now.minute
        if hm >= 1503:
            break
        if 1132 <= hm < 1259 or hm < 930:  # 午休和开盘前不取
            time.sleep(60)
            continue
        t0 = time.time()
        try:
            round_once(state, codes, masks, imap)
            fails = 0
        except Exception as e:  # noqa: BLE001  单轮失败下一轮再来
            fails += 1
            print('本轮失败', type(e).__name__, str(e)[:200], flush=True)
            if fails >= 6:
                print('连续失败 6 轮，退出')
                return 1
        time.sleep(max(600 - (time.time() - t0), 30))
    return 0


if __name__ == '__main__':
    sys.exit(main())
