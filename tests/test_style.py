"""离线验证因子收益的回归、约束、时点和页面汇总。"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import insights
import market
import style


class StyleTests(unittest.TestCase):
    def test_wls_recovers_market_industry_and_style(self):
        rng = np.random.default_rng(11)
        exposure = rng.normal(size=(160, len(style.FACTORS)))
        cap = rng.lognormal(size=160)
        sector = np.repeat([100, 200, 300, 400], 40)
        shares = cap.reshape(4, 40).sum(axis=1)
        industry = np.array([.01, -.03, .025, .009])
        industry -= np.average(industry, weights=shares)
        known = np.linspace(-.008, .006, len(style.FACTORS))
        realized = .008 + np.repeat(industry, 40) + exposure @ known
        intercept, coefficients, premia, r2 = style.fit_wls(realized, exposure, cap, sector)
        np.testing.assert_allclose(intercept, .008, atol=1e-12)
        np.testing.assert_allclose(coefficients, industry, atol=1e-12)
        np.testing.assert_allclose(premia, known, atol=1e-12)
        np.testing.assert_allclose(r2, 1, atol=1e-12)
        np.testing.assert_allclose(coefficients @ shares, 0, atol=1e-12)
        # 独立的等式约束解验证有残差时的 WLS 权重。
        realized += rng.normal(0, .01, 160)
        intercept, coefficients, premia, _ = style.fit_wls(realized, exposure, cap, sector)
        design = np.column_stack((np.ones(160), np.repeat(np.eye(4), 40, axis=0), exposure))
        constraint = np.r_[0, shares / shares.sum(), np.zeros(len(known))]
        weighted = design.T * np.sqrt(cap)
        equations = np.block([[weighted @ design, constraint[:, None]], [constraint, 0]])
        solution = np.linalg.solve(equations, np.r_[weighted @ realized, 0])[:-1]
        np.testing.assert_allclose(np.r_[intercept, coefficients, premia], solution, atol=1e-12)
        np.testing.assert_allclose(coefficients @ shares, 0, atol=1e-12)

    def test_exposure_is_lagged_and_normalized(self):
        rng = np.random.default_rng(12)
        n, days = 80, 256
        shape = (days, n)
        caps = rng.lognormal(5, 1, size=n)
        ret = rng.normal(0, .012, (days, 1)) * rng.uniform(.5, 1.6, n) + rng.normal(0, .015, shape)
        dates = pd.bdate_range('2025-01-01', periods=days).strftime('%Y-%m-%d')
        df = pd.DataFrame({'date': np.repeat(dates, n), 'code': np.tile(np.arange(100, 100 + n), days)})
        for c in market.COLS[1:]:
            df[c] = 0.0
        df['close'] = (10 * (1 + ret)).ravel()
        df['pre_close'] = 10.0
        df['mv'] = np.tile(caps, days)
        df['fmv'] = df['mv'] * .8
        df['amount'] = df['fmv'] * 1e8 * rng.uniform(.001, .1, days * n)
        df['pb'] = rng.uniform(.5, 5, days * n)
        df['pe'] = rng.choice([-1, 1], days * n) * rng.uniform(10, 45, days * n)
        sector = np.arange(n) % 4
        i = 252
        block = market.Block(df)
        exposure = style.exposures(block, i, sector)
        self.assertTrue(np.isfinite(exposure).all())
        self.assertTrue(np.isnan(style.exposures(block, 250, sector)).all())
        np.testing.assert_allclose(np.average(exposure, weights=caps, axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(exposure.std(axis=0), 1, atol=1e-12)
        cov = (exposure.T * caps) @ exposure / caps.sum()
        np.testing.assert_allclose(cov[[1, 4, 4, 5], [0, 0, 2, 0]], 0, atol=1e-7)
        changed = df.copy()
        future = changed['date'] >= dates[i]
        changed.loc[future, ['close', 'pre_close', 'st', 'fmv', 'mv', 'amount', 'pb', 'pe']] = 2
        np.testing.assert_array_equal(style.exposures(market.Block(changed), i, sector), exposure)
        clipped = df[df['date'] < dates[i]].copy()
        np.testing.assert_array_equal(style.exposures(market.Block(clipped), i, sector), exposure)

    def test_coverage_and_monthly_rank(self):
        dates = pd.bdate_range('2026-01-01', periods=21).strftime('%Y-%m-%d')
        daily = pd.DataFrame(dict.fromkeys(style.FACTORS, np.ones(len(dates))))
        daily['date'] = dates
        daily['SIZE'] = -2
        full = insights.style_returns(daily, dates, dates[-1])
        self.assertEqual(full['rows'][0]['d20'], -40)
        self.assertEqual(full['rows'][0]['p20'], 50)
        self.assertEqual(full['curves']['SIZE'][-1], -42)
        self.assertIn('大市值风格跑输 40.00%', full['summary'])
        short = insights.style_returns(daily, dates, dates[9])
        self.assertIsNone(short['rows'][0]['d20'])
        self.assertIsNone(short['rows'][0]['p20'])
        self.assertEqual(short['rows'][0]['d5'], -10)
        self.assertEqual(short['ytd_n'], 10)


if __name__ == '__main__':
    unittest.main()
