"""自托管字体子集：宋体只收页面用到的字；Bodoni 只收数字和符号。国内访问不依赖外部字体服务。

页面加了新字（尤其是标题）后重跑。源字体放 FONTS 目录，没有就从下面两个地址下载：
  https://raw.githubusercontent.com/notofonts/noto-cjk/main/Serif/SubsetOTF/SC/NotoSerifSC-SemiBold.otf
  https://raw.githubusercontent.com/google/fonts/main/ofl/bodonimoda/BodoniModa%5Bopsz%2Cwght%5D.ttf
需要 pip install fonttools brotli。
"""
import re
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

FONTS = Path(r'G:\dev-cache\fonts')
R = Path(__file__).resolve().parent.parent
OUT = R / 'site' / 'assets'
text = ''
for f in ['scripts/template.html', 'scripts/vault.html', 'scripts/insights.py', 'scripts/market.py', 'data/industry_sw1.csv']:
    text += (R / f).read_text(encoding='utf-8')
cjk = set(re.findall(r'[\u3400-\u9fff\u3000-\u303f\uff00-\uffef“”‘’—…·]', text))
print('字数', len(cjk))
base = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz +-−%.,:/()~≥≤×÷↓↑'
opts = subset.Options(flavor='woff2', layout_features=['*'], desubroutinize=True, hinting=False, notdef_outline=True, name_IDs=[1, 2])
font = TTFont(FONTS / 'NotoSerifSC-SemiBold.otf')
s = subset.Subsetter(opts)
s.populate(text=''.join(sorted(cjk)) + base)
s.subset(font)
font.flavor = 'woff2'
font.save(OUT / 'serif-sc.woff2')
print('serif-sc', round((OUT / 'serif-sc.woff2').stat().st_size / 1024), 'KB')

v = TTFont(FONTS / 'BodoniModa.ttf')
print('轴', [(a.axisTag, a.minValue, a.maxValue) for a in v['fvar'].axes])
st = instancer.instantiateVariableFont(v, {'opsz': 28, 'wght': 500})
s = subset.Subsetter(subset.Options(flavor='woff2', layout_features=['lnum', 'tnum', 'kern'], hinting=False, name_IDs=[1, 2]))
s.populate(text='0123456789+-−%.,~:/A')
s.subset(st)
st.flavor = 'woff2'
st.save(OUT / 'bodoni-num.woff2')
print('bodoni', round((OUT / 'bodoni-num.woff2').stat().st_size / 1024), 'KB')
