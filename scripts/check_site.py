"""部署前自检：数据能解析、关键块在、页面脚本语法没问题。不通过就不提交不部署，线上继续显示上一版。

    python scripts/check_site.py
"""
import csv
import json
import re
import subprocess
import sys
import unicodedata
from datetime import date
from html import unescape
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / 'site'



LAB_FORBIDDEN = {
    '证券代码': r'(?<!\d)\d{6}(?!\d)',
    '本机路径或用户名': (
        r'(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\[\w.-]+\\|(?:[/\\](?:Users|home)[/\\])'
        r'|(?:^|[\s/\\])~[/\\]|\b(?:gaaiy(?:un)?|(?:user(?:name)?|用户名)\s*[:=]\s*\w+)\b'
    ),
    '设备或算力': (
        r'\b(?:GPU|CUDA|VRAM|TPU|ROCm|RTX|GTX|GeForce|Tesla|Radeon|Intel|NVIDIA|AMD)\b'
        r'|(?<![\d.])\b(?:[345]\d[56789]\d\s*(?:Ti|SUPER)?|[AHBL]\s*(?:20|40|80|100|200)|MI\d{3})\b'
        r'|显卡|算力|显存'
    ),
    '模型或接口供应商': (
        r'\b(?:OpenAI|ChatGPT|GPT(?:[- ._]\d+)?|Claude|Anthropic|DeepSeek|V4[.\d-]*(?:Flash|Pro)'
        r'|Qwen|Gemini|Google|Grok|Kimi|Moonshot(?:AI)?|GLM|Zhipu|SiliconFlow|OpenRouter'
        r'|ByteDance|Seed-OSS|Doubao|Llama|Meta AI|Mistral|NVIDIA|Azure|Cohere|Baidu|ERNIE|Hunyuan|Tencent'
        r'|Alibaba|DashScope|Eastmoney|AkShare|Tushare|Wind|Choice|JQData)\b'
        r'|东方财富|东财|硅基流动|通义千问|智谱|火山引擎|豆包|文心一言|混元|月之暗面|聚宽'
    ),
    '凭据或用量': (
        r'\b(?:tokens?|api[_ -]?key|access[_ -]?key|password|secret|authorization|bearer|'
        r'sk-[a-z0-9_-]{8,}|(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[a-z0-9]{16,}|'
        r'github_pat_[a-z0-9_]{16,}|AIza[a-z0-9_-]{15,})\b'
        r'|-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----|密钥|口令'
    ),
    '持仓明细字段': (
        r'\b(?:tranches|holdings|positions|stock_codes|stock_names|ticker|symbol|security_code)\b'
        r'|持仓名单|持仓明细|持仓列表'
    ),
    '完整因子表达式': (
        r'(?<!\w)(?:CsRank|CsDemean|CsZScore|TsRank|TsMin|TsMax|IfElse|Mul|Div|Sub|Add|'
        r'Delta|Delay|Corr|Cov|TsMean|Mean|Rank|ZScore|Shift|Where|Rolling)\s*\('
        r'|\$[a-z_]\w*|\b(?:main_flow\w*|ins_net60|sue_kf)\b'
    ),
    '外部地址': r'\b(?:https?|ftp|ssh|file)://',
}
LAB_TOP = 'schema strategies accounts models annual factors experiments dataset'
LAB_MODEL = 'name delta_pp nw_t wins excess_pct conclusion'
LAB_MODEL_DETAIL = 'name delta_pp nw_t wins conclusion category structure features target seeds date note'
LAB_FM = 'name evaluated historical_admitted admitted invalid fitness_median incremental_ic_median validation_2024_median'
LAB_RECORD = 'id conclusion metrics'
LAB_YEARS = range(2019, 2027)


def lab_keys(row, keys):
    if type(row) is not dict or set(row) != set(keys.split()):
        raise ValueError('字段结构')


def lab_num(value):
    if value is not None and (type(value) not in (int, float) or not isfinite(value)):
        raise ValueError('应为有限数值或空')


def lab_points(points):
    previous = ''
    for day, value in points:
        date.fromisoformat(day)
        lab_num(value)
        if day <= previous or value is None or value <= 0:
            raise ValueError('净值序列无序或含无效值')
        previous = day


def validate_lab(d):
    lab_keys(d, LAB_TOP)
    if d['schema'] != 2 or not all(d[key] for key in LAB_TOP.split()[1:]):
        raise ValueError('版本或分区缺失')
    for row in d['strategies']:
        lab_keys(row, 'name group rule period paths annual_pct sharpe calmar drawdown_pct worst_year_pct')
        for k in ('paths', 'annual_pct', 'sharpe', 'calmar', 'drawdown_pct', 'worst_year_pct'):
            lab_num(row[k])
        for day in row['period']:
            date.fromisoformat(day)
    for row in d['accounts']:
        lab_keys(row, 'name start end latest_nav return_pct series benchmarks')
        lab_points(row['series'])
        date.fromisoformat(row['start'])
        date.fromisoformat(row['end'])
        lab_num(row['latest_nav'])
        lab_num(row['return_pct'])
        if not row['series'] or row['start'] > row['series'][0][0] or row['series'][-1] != [row['end'], row['latest_nav']]:
            raise ValueError('模拟账户日期或最新净值不一致')
        if [b['name'] for b in row['benchmarks']] != ['沪深300', '中证1000', '中证2000']:
            raise ValueError('基准指数缺失')
        for b in row['benchmarks']:
            lab_keys(b, 'name return_pct series')
            lab_points(b['series'])
            lab_num(b['return_pct'])
            if b['series'] and (b['series'][0] != [row['start'], 1] or b['series'][-1][0] > row['end']):
                raise ValueError('指数与账户期间不一致')
    model_names = set()
    for row in d['models']:
        lab_keys(row, LAB_MODEL_DETAIL)
        for k in ('delta_pp', 'nw_t'):
            lab_num(row[k])
        for k in ('name', 'category', 'structure', 'features', 'target', 'conclusion'):
            if not isinstance(row[k], str) or not row[k]:
                raise ValueError('模型说明缺失')
        date.fromisoformat(row['date'])
        if row['seeds'] is not None and (type(row['seeds']) is not int or row['seeds'] < 0):
            raise ValueError('种子数无效')
        if row['wins'] is not None and not re.fullmatch(r'\d+/\d+', row['wins']):
            raise ValueError('胜年无效')
        if row['name'] in model_names:
            raise ValueError('模型重复')
        model_names.add(row['name'])
    required_models = {'NN3T1', 'NN3W20', 'PVPLUS', 'PVW', 'PVO', 'PVPLUSB', 'PVWB', 'PVT', 'PVTB',
                       'PVENS', 'B_PV2', 'ARCH3', 'MT1', 'EV11', 'EV12', 'SEQ1', 'LIST1', 'LIST2',
                       'TDW1', 'TDW2', 'TDW3', 'HUBER', 'HUBERM', 'FMA1', 'FMB1',
                       'v1', 'P14', 'LGB', 'T5', 'NN3', 'NN3S', 'C1',
                       'ABLCORE', 'ABLPV', 'ABLMOD', 'ABLFUND', 'ABLF2', 'ONLYPV', 'ONLYFUND', 'ONLYMOD',
                       *('R3_IRANK_' + suffix for suffix in ('G', '25', '50', '100', '50P'))}
    if not required_models.issubset(model_names):
        raise ValueError('模型一览缺失')

    annual_names = set()
    for row in d['annual']:
        lab_keys(row, 'name group end values')
        end = date.fromisoformat(row['end'])
        if len(row['values']) != len(LAB_YEARS) or row['group'] not in {'回测', '10 万配置', '模拟账户', '指数参照'}:
            raise ValueError('逐年收益区间无效')
        key = (row['group'], row['name'])
        if key in annual_names:
            raise ValueError('逐年收益策略重复')
        annual_names.add(key)
        observed = [year for year, value in zip(LAB_YEARS, row['values']) if value is not None]
        if not observed or observed[-1] != end.year:
            raise ValueError('逐年收益末日不一致')
        for value in (x for x in row['values'] if x is not None):
            lab_keys(value, 'return_pct band_pct')
            mid = value['return_pct']
            lab_num(mid)
            if mid is None or mid <= -100:
                raise ValueError('逐年收益无效')
            if value['band_pct'] is not None:
                low, high = value['band_pct']
                lab_num(low)
                lab_num(high)
                if not -100 < low <= mid <= high:
                    raise ValueError('收益分位区间无效')
    required_annual = {('回测', 'NN3T1 候选池 100 只'),
                       *(('10 万配置' if row['group'] == '10 万配置' else '回测', row['name']) for row in d['strategies']),
                       *(('模拟账户', row['name']) for row in d['accounts'])}
    if not required_annual.issubset(annual_names):
        raise ValueError('逐年收益缺少策略或账户')
    f = d['factors']
    lab_keys(f, 'rounds families models events')
    lab_num(f['rounds'])
    for row in f['families']:
        lab_keys(row, LAB_FM + ' construction void')
        if type(row['void']) is not bool or (row['void'] and (row['admitted'] or row['invalid'] != row['evaluated'])):
            raise ValueError('作废因子统计不一致')
    for row in f['models']:
        lab_keys(row, LAB_FM)
        if not re.fullmatch(r'模型 \d{2}|未记模型', row['name']):
            raise ValueError('模型未匿名')
    for row in f['families'] + f['models']:
        for k in LAB_FM.split()[1:]:
            lab_num(row[k])
        if not 0 <= row['admitted'] <= row['historical_admitted'] <= row['evaluated']:
            raise ValueError('入库数不一致')
    for row in f['events']:
        lab_keys(row, LAB_MODEL + ' construction coverage_pct residual_ic')
        for k in ('delta_pp', 'nw_t', 'excess_pct', 'coverage_pct', 'residual_ic'):
            lab_num(row[k])
    e = d['experiments']
    lab_keys(e, 'total species bans gates')
    for s in e['species']:
        lab_keys(s, 'name status representative records')
        if s['status'] not in {'活跃', '休眠', '禁区'}:
            raise ValueError('物种状态无效')
        if s['representative'] not in s['records']:
            raise ValueError('代表实验未登记')
        for row in s['records']:
            lab_keys(row, LAB_RECORD)
    if e['total'] != sum(len(s['records']) for s in e['species']):
        raise ValueError('登记总数不一致')
    for rows, width in ((e['bans'], 3), (e['gates'], 4)):
        if not all(len(row) == width and all(isinstance(x, str) for x in row) for row in rows):
            raise ValueError('门检或禁区记录不完整')
    t = d['dataset']
    lab_keys(t, 'samples period unavailable_days fields review_count agreement_fields agreement event_windows events predictions')
    for row in t['agreement']:
        lab_keys(row, 'pair rates kappa')
        for values in (row['rates'], row['kappa']):
            if values is not None:
                if len(values) != len(t['agreement_fields']):
                    raise ValueError('一致性表字段不完整')
                for value in values:
                    lab_num(value)
    for row in t['events']:
        lab_keys(row, 'name n windows')
        if len(row['windows']) != len(t['event_windows']):
            raise ValueError('事件窗口不完整')
        for change, stat, n in row['windows']:
            for value in (change, stat, n):
                lab_num(value)
    for row in t['predictions']:
        lab_keys(row, 'target name n positive weekly_auc weekly_delta pooled_auc pooled_delta hit_pct hit_delta_pp')
        for k in set(row) - {'name', 'target'}:
            lab_num(row[k])


def lab_strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from lab_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from lab_strings(item)
    elif isinstance(value, str):
        yield value


def normalize(text):
    return unicodedata.normalize('NFKC', unescape(text)).replace('\u200b', '').replace('\ufeff', '')


def stock_name_pattern():
    names = set()
    for path in (ROOT / 'data/raw/spot_latest.csv', ROOT / 'data/lhb_daily.csv'):
        if path.is_file():
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    name = re.sub(r'\s+', '', normalize(row.get('name', '')))
                    if name:
                        names.add(name)
    if not names:
        raise ValueError('缺少证券名称参照数据')
    return re.compile('|'.join(map(re.escape, sorted(names, key=lambda x: (-len(x), x)))), re.I)


def check_lab(html, payload, home=''):
    problems = []
    pages = (('lab.html', html), ('首页研究板块', home))
    sources = (*pages, ('lab.json', '\n'.join(lab_strings(payload))))
    for name, source in sources:
        for label, pattern in LAB_FORBIDDEN.items():
            if re.search(pattern, normalize(source), re.I):
                problems.append(f'{name} 含禁用内容（{label}）')
    construction = [r['construction'] for key in ('families', 'events') for r in payload['factors'][key]]
    if any(re.search(r'\b[A-Za-z]\w*\s*[+*/^=-]\s*[A-Za-z0-9_]\w*\b', s) for s in construction):
        problems.append('lab.json 含禁用内容（完整因子表达式）')
    names = stock_name_pattern()
    # 再扫描去标签后的可见文本，以识别被内部标签分开的简称。
    for name, source in sources:
        visible = re.sub(r'<(script|style)\b[^>]*>.*?</\1>', '', source, flags=re.I | re.S)
        visible = re.sub(r'</?(?:span|b|i|em|strong)\b[^>]*>', '', visible, flags=re.I)
        visible = re.sub(r'<[^>]+>', '\n', visible)
        source += '\n' + visible
        if any(names.search(re.sub(r'[ \t\u3000]+', '', line)) for line in normalize(source).splitlines()):
            problems.append(f'{name} 含禁用内容（证券名称）')
    return problems


def check_script(html, label, problems):
    scripts = re.findall(r'<script>(.*?)</script>', html, re.S)
    if not scripts:
        problems.append(f'{label} 缺少内联脚本')
        return
    for script in scripts:
        run = subprocess.run(['node', '--check', '-'], input=script.encode(), capture_output=True)
        if run.returncode:
            problems.append(f'{label} 脚本语法错误')


def check_style_data(style, problems):
    """仅检查风格收益的数据。"""
    code = LAB_FORBIDDEN['证券代码']
    path = r'[A-Za-z]:[\\/]|\\\\|(?:^|[\s=:(])(?:~|[\w.-]+)?[/\\][\w.-]+'
    if any(re.search(code, normalize(value)) for value in lab_strings(style)) or re.search(
        r'(?:\[|:|,)\d{6}(?=[,}\]])', json.dumps(style, separators=(',', ':'))
    ):
        problems.append('风格因子收益含六位证券代码')
    if any(re.search(path, normalize(value)) for value in lab_strings(style)):
        problems.append('风格因子收益含路径')


def check_lab_artifacts(index, problems):
    model = re.search(r'<section\b[^>]*\bid="model_sec"[^>]*>(.*?)</section>', index, re.S)
    home = model[1].partition('<div id="m_lab"')[2] if model else ''
    for id_ in ('m_lab_strategies', 'm_lab_annual_table', 'm_lab_chart', 'm_lab_accounts',
                'm_lab_models', 'm_lab_factors'):
        if f'id="{id_}"' not in home:
            problems.append(f'首页研究板块缺 {id_}')
    if '<a class="vault-link" href="lab.html"><b>完整研究看板 →</b></a>' not in home:
        problems.append('选股模型缺少完整研究看板链接')
    if '<!--__LAB_' in home:
        problems.append('首页研究数据没有进入构建产物')
    nav = re.search(r'<nav\b[^>]*\bid="toc"[^>]*>(.*?)</nav>', index, re.S)
    if '<a href="lab.html">研究</a>' not in index:
        problems.append('首页缺少研究导航')
    if not nav or 'href="#model_sec"' not in nav[1]:
        problems.append('首页模型导航配置错误')
    try:
        html = (SITE / 'lab.html').read_text(encoding='utf-8')
        payload = json.loads((SITE / 'lab.json').read_text(encoding='utf-8'))
        if html != (ROOT / 'scripts/lab.html').read_text(encoding='utf-8'):
            problems.append('lab.html 与构建模板不一致')
        if not re.search(r"""fetch\(\s*['"]\./lab\.json['"]\s*\)""", html):
            problems.append('lab.html 缺少数据加载入口')
        for id_ in ('annual-chart', 'annual-table', 'annual-picker', 'model-table', 'model-picker'):
            if f'id="{id_}"' not in html:
                problems.append('lab.html 缺少模型或逐年收益控件')
        validate_lab(payload)
        problems.extend(check_lab(html, payload, home))
        check_script(html, 'lab.html', problems)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        # 不回显输入值，避免在失败日志中暴露私有数据。
        problems.append('研究产物缺失、不可读取或字段校验失败（' + type(exc).__name__ + '）')


def main():
    D = json.loads((SITE / 'data.json').read_text(encoding='utf-8'))
    problems = []
    for k in ('asof', 'today', 'indices', 'sent_series', 'x'):
        if not D.get(k):
            problems.append(f'缺 {k}')
    X = D.get('x') or {}
    check_style_data(X.get('style_returns'), problems)
    for k in ('plays', 'feel', 'sectors', 'sayings', 'four'):
        if not X.get(k):
            problems.append(f'判断块 {k} 为空')
    if X.get('errors'):
        # 单块出错只影响那一块，页面还能用：提示但不拦截
        print('[WARN] 判断块出错：', X['errors'])
    html = (SITE / 'index.html').read_text(encoding='utf-8')
    if '/*__DATA__*/null' in html:
        problems.append('数据没有嵌进页面')
    if (ROOT / 'data' / 'factor_research.json').exists():
        research = D.get('factor_research')
        if not research or not research.get('batches') or not research.get('overview') or 'research_comparison' not in html:
            problems.append('Agent 因子研究没有进入构建产物')
        elif any(re.search(p, json.dumps(research, ensure_ascii=False), re.I) for p in (
            r'(?<![A-Za-z])[A-Za-z]:[\\/]', r'\b\d{6}\.(SH|SZ|BJ|XSHG|XSHE)\b',
            r'\b(sh|sz|bj)\d{6}\b', r'COSMOS_\d+', r'QA2_\d+', r'sk-[A-Za-z0-9]{8,}',
            r'5070|GPU|CUDA|tokens|\bHTTP\b|断点|重放|worker|supervisor')):
            problems.append('Agent 研究产物没有脱敏')
    check_script(html, 'index.html', problems)
    check_lab_artifacts(html, problems)
    for p in problems:
        print('[FAIL]', p)
    print('自检', '不通过' if problems else '通过', D.get('asof'))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
