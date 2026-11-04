"""临时：探测几个公开接口在 Actions 上能不能取到、返回什么列。用完删除。"""
import time

import akshare as ak


def t(name, fn):
    t0 = time.time()
    try:
        df = fn()
        print(f'[OK] {name} {time.time() - t0:.1f}s shape={df.shape}\n  cols={list(df.columns)}\n{df.tail(3).to_string()[:900]}\n', flush=True)
    except Exception as e:  # noqa: BLE001
        print(f'[FAIL] {name} {time.time() - t0:.1f}s {type(e).__name__}: {str(e)[:200]}\n', flush=True)


t('解禁汇总', lambda: ak.stock_restricted_release_summary_em(symbol='全部股票', start_date='20261105', end_date='20270105'))
t('解禁明细', lambda: ak.stock_restricted_release_detail_em(start_date='20261105', end_date='20261231'))
t('业绩报表Q3', lambda: ak.stock_yjbb_em(date='20260930'))
t('业绩预告', lambda: ak.stock_yjyg_em(date='20261231'))
t('增减持', lambda: ak.stock_ggcg_em(symbol='全部'))
t('股息率沪深300', lambda: ak.stock_a_gxl_lg(symbol='沪深300'))
t('股息率上证A', lambda: ak.stock_a_gxl_lg(symbol='上证A股'))
t('回购', lambda: ak.stock_repurchase_em())
t('新股', lambda: ak.stock_xgsglb_em(symbol='全部股票'))
t('两融沪市', lambda: ak.stock_margin_sse(start_date='20261101', end_date='20261104'))
t('股东户数', lambda: ak.stock_zh_a_gdhs(symbol='20260930'))
t('板块资金', lambda: ak.stock_fund_flow_industry(symbol='即时'))
t('大宗', lambda: ak.stock_dzjy_sctj())
