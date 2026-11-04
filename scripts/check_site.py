"""部署前自检：数据能解析、关键块在、页面脚本语法没问题。不通过就不提交不部署，线上继续显示上一版。

    python scripts/check_site.py
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / 'site'


def main():
    D = json.loads((SITE / 'data.json').read_text(encoding='utf-8'))
    problems = []
    for k in ('asof', 'today', 'indices', 'sent_series', 'x'):
        if not D.get(k):
            problems.append(f'缺 {k}')
    X = D.get('x') or {}
    for k in ('plays', 'feel', 'sectors', 'sayings', 'four'):
        if not X.get(k):
            problems.append(f'判断块 {k} 为空')
    if X.get('errors'):
        # 单块出错只影响那一块，页面还能用：提示但不拦截
        print('[WARN] 判断块出错：', X['errors'])
    html = (SITE / 'index.html').read_text(encoding='utf-8')
    if '/*__DATA__*/null' in html:
        problems.append('数据没有嵌进页面')
    script = re.findall(r'<script>(.*?)</script>', html, re.S)[-1]
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(script)
    r = subprocess.run(['node', '--check', f.name], capture_output=True, text=True)
    if r.returncode != 0:
        problems.append('页面脚本语法错误：' + r.stderr[:300])
    for p in problems:
        print('[FAIL]', p)
    print('自检', '不通过' if problems else '通过', D.get('asof'))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
