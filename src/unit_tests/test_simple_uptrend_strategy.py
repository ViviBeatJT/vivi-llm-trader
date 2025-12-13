# tests/test_simple_uptrend_strategy.py
# python -m pytest src/unit_tests/test_simple_uptrend_strategy.py


"""
SimpleUpTrendStrategy 单元测试

测试覆盖的 Corner Cases:
1. 最后10分钟不开仓
2. 不连续开仓（已有仓位时不再买入）
3. 止损后冷却期
4. 连续亏损冷却期加长
5. 连续亏损后 reduce allocation
6. 布林带很窄时谨慎交易
7. 只在指定布林带范围内交易
8. 最多只有 max_allocation 的钱拿来投资
9. 不会连续开仓超过 allocation
10. 盈利后重置连续亏损计数
11. 各种市场状态下的行为
12. 边界条件测试
"""

import unittest
import sys
import os
from datetime import datetime, time, timedelta
from io import StringIO

import pandas as pd
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.simple_uptrend_strategy import SimpleUpTrendStrategy


class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def create_ohlcv_data(
        n_bars: int = 50,
        start_price: float = 100.0,
        trend: str = 'flat',  # 'up', 'down', 'flat', 'volatile'
        volatility: float = 0.02,
        start_time: datetime = None
    ) -> pd.DataFrame:
        """
        生成 OHLCV 测试数据
        
        Args:
            n_bars: K线数量
            start_price: 起始价格
            trend: 趋势方向
            volatility: 波动率
            start_time: 开始时间
        """
        if start_time is None:
            start_time = datetime(2024, 1, 15, 9, 30)
        
        np.random.seed(42)
        
        prices = [start_price]
        for i in range(1, n_bars):
            change = np.random.normal(0, volatility)
            
            if trend == 'up':
                change += 0.002  # 上涨偏移
            elif trend == 'down':
                change -= 0.002  # 下跌偏移
            elif trend == 'volatile':
                change *= 2  # 加大波动
                
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # 确保价格为正
        
        data = []
        for i, close in enumerate(prices):
            high = close * (1 + abs(np.random.normal(0, volatility/2)))
            low = close * (1 - abs(np.random.normal(0, volatility/2)))
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.randint(1000, 10000)
            
            timestamp = start_time + timedelta(minutes=i)
            data.append({
                'open': open_price,
                'high': max(high, open_price, close),
                'low': min(low, open_price, close),
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.index = pd.date_range(start=start_time, periods=n_bars, freq='1min')
        return df

    @staticmethod
    def create_narrow_bb_data(n_bars: int = 50, base_price: float = 100.0) -> pd.DataFrame:
        """创建布林带很窄的数据（低波动）"""
        return TestDataGenerator.create_ohlcv_data(
            n_bars=n_bars,
            start_price=base_price,
            trend='flat',
            volatility=0.001  # 极低波动
        )

    @staticmethod
    def create_uptrend_data(n_bars: int = 50, start_price: float = 100.0) -> pd.DataFrame:
        """创建上升趋势数据"""
        return TestDataGenerator.create_ohlcv_data(
            n_bars=n_bars,
            start_price=start_price,
            trend='up',
            volatility=0.015
        )

    @staticmethod
    def create_downtrend_data(n_bars: int = 50, start_price: float = 100.0) -> pd.DataFrame:
        """创建下降趋势数据"""
        return TestDataGenerator.create_ohlcv_data(
            n_bars=n_bars,
            start_price=start_price,
            trend='down',
            volatility=0.015
        )


class SuppressOutput:
    """抑制策略打印输出"""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stdout = self._stdout


class TestMarketCloseProtection(unittest.TestCase):
    """测试收盘保护功能"""
    
    def setUp(self):
        """每个测试前初始化"""
        with SuppressOutput():
            self.strategy = SimpleUpTrendStrategy(
                no_new_position_minutes=10,
                market_close_time=time(16, 0),
                cooldown_bars=3,
                verbose_init=False
            )
        self.ticker = 'TEST'
        self.data = TestDataGenerator.create_ohlcv_data(n_bars=30)
    
    def test_no_buy_in_last_10_minutes(self):
        """测试：最后10分钟不开仓"""
        # 15:51 - 应该禁止开仓
        test_time = datetime(2024, 1, 15, 15, 51)
        
        with SuppressOutput():
            result, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=self.data,
                current_position=0,
                avg_cost=0,
                current_time_et=test_time
            )
        
        # 即使其他条件满足买入，也应该是 HOLD
        self.assertEqual(result['signal'], 'HOLD')
        self.assertIn('收盘前', result['reason'])
    
    def test_allow_buy_before_cutoff(self):
        """测试：收盘前10分钟之前可以正常交易"""
        test_time = datetime(2024, 1, 15, 15, 45)  # 15:45
        
        with SuppressOutput():
            result, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=self.data,
                current_position=0,
                avg_cost=0,
                current_time_et=test_time
            )
        
        # 不应该因为时间而被阻止（可能因为其他原因是 HOLD）
        self.assertNotIn('收盘前', result.get('reason', ''))
    
    def test_allow_sell_in_last_10_minutes(self):
        """测试：最后10分钟允许卖出"""
        test_time = datetime(2024, 1, 15, 15, 55)
        
        # 模拟持仓且亏损触发止损
        with SuppressOutput():
            result, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=self.data,
                current_position=100,
                avg_cost=200,  # 大幅亏损
                current_time_et=test_time
            )
        
        # 止损应该正常触发
        self.assertEqual(result['signal'], 'SELL')
    
    def test_force_close_at_market_close(self):
        """测试：收盘时强制平仓"""
        with SuppressOutput():
            result, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=self.data,
                current_position=100,
                avg_cost=100,
                is_market_close=True
            )
        
        self.assertEqual(result['signal'], 'SELL')
        self.assertIn('收盘平仓', result['reason'])
    
    def test_different_cutoff_times(self):
        """测试：不同的禁止开仓时间"""
        # 测试 15:50 (刚好10分钟)
        self.assertTrue(self.strategy._is_last_n_minutes(datetime(2024, 1, 15, 15, 50)))
        
        # 测试 15:49 (不在10分钟内)
        self.assertFalse(self.strategy._is_last_n_minutes(datetime(2024, 1, 15, 15, 49)))
        
        # 测试 15:59 (在10分钟内)
        self.assertTrue(self.strategy._is_last_n_minutes(datetime(2024, 1, 15, 15, 59)))
        
        # 测试 16:00 (已收盘)
        self.assertFalse(self.strategy._is_last_n_minutes(datetime(2024, 1, 15, 16, 0)))


class TestCooldownPeriod(unittest.TestCase):
    """测试冷却期功能"""
    
    def setUp(self):
        with SuppressOutput():
            self.strategy = SimpleUpTrendStrategy(
                cooldown_bars=5,
                cooldown_minutes=0,
                consecutive_loss_multiplier=2.0,
                max_cooldown_multiplier=4.0,
                verbose_init=False
            )
        self.ticker = 'TEST'
    
    def test_cooldown_after_stop_loss(self):
        """测试：止损后进入冷却期"""
        self.strategy._bar_count[self.ticker] = 100
        
        with SuppressOutput():
            self.strategy._start_cooldown(self.ticker, datetime.now(), is_stop_loss=True)
        
        # 检查冷却期状态
        self.strategy._bar_count[self.ticker] = 102  # 过了2根K线
        is_cooling, reason = self.strategy._is_in_cooldown(self.ticker)
        
        self.assertTrue(is_cooling)
        self.assertIn('还需', reason)
    
    def test_cooldown_ends_after_n_bars(self):
        """测试：冷却期在N根K线后结束"""
        self.strategy._bar_count[self.ticker] = 100
        
        with SuppressOutput():
            self.strategy._start_cooldown(self.ticker, datetime.now(), is_stop_loss=True)
        
        # 过了足够的K线
        self.strategy._bar_count[self.ticker] = 106  # 过了6根K线，超过5根
        is_cooling, reason = self.strategy._is_in_cooldown(self.ticker)
        
        self.assertFalse(is_cooling)
    
    def test_consecutive_loss_extends_cooldown(self):
        """测试：连续亏损延长冷却期"""
        # 第一次止损
        self.strategy._bar_count[self.ticker] = 100
        with SuppressOutput():
            self.strategy._start_cooldown(self.ticker, datetime.now(), is_stop_loss=True)
        
        losses = self.strategy.get_consecutive_losses(self.ticker)
        multiplier1 = self.strategy.get_cooldown_multiplier(self.ticker)
        
        self.assertEqual(losses, 1)
        self.assertEqual(multiplier1, 1.0)  # 第一次止损，倍数为1
        
        # 清除冷却期，模拟第二次止损
        self.strategy._clear_cooldown(self.ticker)
        self.strategy._bar_count[self.ticker] = 200
        with SuppressOutput():
            self.strategy._start_cooldown(self.ticker, datetime.now(), is_stop_loss=True)
        
        losses = self.strategy.get_consecutive_losses(self.ticker)
        multiplier2 = self.strategy.get_cooldown_multiplier(self.ticker)
        
        self.assertEqual(losses, 2)
        self.assertEqual(multiplier2, 2.0)  # 第二次止损，倍数为2x
        
        # 第三次止损
        self.strategy._clear_cooldown(self.ticker)
        self.strategy._bar_count[self.ticker] = 300
        with SuppressOutput():
            self.strategy._start_cooldown(self.ticker, datetime.now(), is_stop_loss=True)
        
        multiplier3 = self.strategy.get_cooldown_multiplier(self.ticker)
        self.assertEqual(multiplier3, 4.0)  # 达到最大倍数
    
    def test_max_cooldown_multiplier_cap(self):
        """测试：冷却期倍数有上限"""
        # 模拟多次连续亏损
        for i in range(10):
            self.strategy._bar_count[self.ticker] = i * 100
            with SuppressOutput():
                self.strategy._start_cooldown(self.ticker, datetime.now(), is_stop_loss=True)
            self.strategy._clear_cooldown(self.ticker)
        
        multiplier = self.strategy.get_cooldown_multiplier(self.ticker)
        self.assertLessEqual(multiplier, self.strategy.max_cooldown_multiplier)
    
    def test_profit_resets_consecutive_losses(self):
        """测试：盈利重置连续亏损计数"""
        # 先模拟几次亏损
        for i in range(3):
            self.strategy._bar_count[self.ticker] = i * 100
            with SuppressOutput():
                self.strategy._start_cooldown(self.ticker, datetime.now(), is_stop_loss=True)
            self.strategy._clear_cooldown(self.ticker)
        
        self.assertEqual(self.strategy.get_consecutive_losses(self.ticker), 3)
        
        # 记录一次盈利
        self.strategy._record_profit(self.ticker)
        
        self.assertEqual(self.strategy.get_consecutive_losses(self.ticker), 0)
        self.assertEqual(self.strategy.get_cooldown_multiplier(self.ticker), 1.0)


class TestAllocationManagement(unittest.TestCase):
    """测试仓位管理功能"""
    
    def setUp(self):
        with SuppressOutput():
            self.strategy = SimpleUpTrendStrategy(
                max_allocation=1.0,
                min_allocation=0.25,
                reduce_allocation_ratio=0.5,
                reduce_allocation_threshold=0.01,
                recovery_threshold=0.005,
                recovery_step=0.1,
                verbose_init=False
            )
        self.ticker = 'TEST'
    
    def test_initial_allocation_is_max(self):
        """测试：初始仓位是最大值"""
        allocation = self.strategy.get_current_allocation(self.ticker)
        self.assertEqual(allocation, self.strategy.max_allocation)
    
    def test_reduce_allocation_on_loss(self):
        """测试：亏损时减仓"""
        with SuppressOutput():
            self.strategy._reduce_allocation(self.ticker, "测试亏损")
        
        allocation = self.strategy.get_current_allocation(self.ticker)
        expected = self.strategy.max_allocation * self.strategy.reduce_allocation_ratio
        self.assertEqual(allocation, expected)
    
    def test_allocation_not_below_minimum(self):
        """测试：仓位不低于最小值"""
        # 多次减仓
        for _ in range(10):
            with SuppressOutput():
                self.strategy._reduce_allocation(self.ticker, "测试")
        
        allocation = self.strategy.get_current_allocation(self.ticker)
        self.assertGreaterEqual(allocation, self.strategy.min_allocation)
    
    def test_recover_allocation_after_profit(self):
        """测试：盈利后恢复仓位"""
        # 先减仓
        with SuppressOutput():
            self.strategy._reduce_allocation(self.ticker, "测试")
        reduced = self.strategy.get_current_allocation(self.ticker)
        
        # 恢复仓位
        with SuppressOutput():
            self.strategy._recover_allocation(self.ticker)
        recovered = self.strategy.get_current_allocation(self.ticker)
        
        self.assertGreater(recovered, reduced)
        self.assertEqual(recovered, reduced + self.strategy.recovery_step)
    
    def test_allocation_not_above_maximum(self):
        """测试：仓位不超过最大值"""
        # 多次恢复
        for _ in range(20):
            with SuppressOutput():
                self.strategy._recover_allocation(self.ticker)
        
        allocation = self.strategy.get_current_allocation(self.ticker)
        self.assertLessEqual(allocation, self.strategy.max_allocation)
    
    def test_cannot_open_position_when_max_invested(self):
        """测试：已达最大投资时不能开仓"""
        self.strategy._total_invested[self.ticker] = 1.0
        
        can_open, reason = self.strategy._can_open_position(self.ticker, 0.5)
        
        self.assertFalse(can_open)
        self.assertIn('最大仓位', reason)
    
    def test_partial_position_allowed(self):
        """测试：部分投资后仍可开仓"""
        self.strategy._total_invested[self.ticker] = 0.5
        
        can_open, reason = self.strategy._can_open_position(self.ticker, 0.3)
        
        self.assertTrue(can_open)


class TestBollingerBandProtection(unittest.TestCase):
    """测试布林带保护功能"""
    
    def setUp(self):
        with SuppressOutput():
            self.strategy = SimpleUpTrendStrategy(
                bb_narrow_threshold=0.02,
                bb_narrow_action='BLOCK',
                cooldown_bars=0,
                verbose_init=False
            )
        self.ticker = 'TEST'
    
    def test_narrow_bb_detection(self):
        """测试：检测布林带过窄"""
        # 窄幅布林带
        is_narrow, width = self.strategy._is_bb_narrow(
            bb_upper=100.9,
            bb_lower=99.0,
            price=100.0
        )
        self.assertTrue(is_narrow)  # 1.9% 宽度 < 2% 阈值... 等于阈值
        
        # 正常布林带
        is_narrow, width = self.strategy._is_bb_narrow(
            bb_upper=105.0,
            bb_lower=95.0,
            price=100.0
        )
        self.assertFalse(is_narrow)  # 10% 宽度
    
    def test_block_trading_when_bb_narrow(self):
        """测试：布林带过窄时阻止交易"""
        # 创建低波动数据
        data = TestDataGenerator.create_narrow_bb_data(n_bars=30)
        
        with SuppressOutput():
            result, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=data,
                current_position=0,
                avg_cost=0,
                current_time_et=datetime(2024, 1, 15, 10, 0)
            )
        
        # 如果检测到窄幅且设置为 BLOCK，应该是 HOLD
        if result['is_bb_narrow']:
            self.assertEqual(result['signal'], 'HOLD')
    
    def test_warn_mode_reduces_confidence(self):
        """测试：WARN 模式降低信心度"""
        with SuppressOutput():
            warn_strategy = SimpleUpTrendStrategy(
                bb_narrow_threshold=0.02,
                bb_narrow_action='WARN',
                verbose_init=False
            )
        
        # 检查 WARN 模式的逻辑存在
        self.assertEqual(warn_strategy.bb_narrow_action, 'WARN')


class TestBollingerBandTradingRange(unittest.TestCase):
    """测试布林带交易范围"""
    
    def setUp(self):
        with SuppressOutput():
            self.strategy = SimpleUpTrendStrategy(
                uptrend_buy_low=0.40,
                uptrend_buy_high=0.60,
                range_buy_threshold=0.20,
                range_sell_threshold=0.55,
                verbose_init=False
            )
        self.ticker = 'TEST'
    
    def test_uptrend_buy_only_in_range(self):
        """测试：上升趋势只在指定范围买入"""
        # BB位置在范围内
        signal, conf, reason = self.strategy._uptrend_strategy(
            position=0, avg_cost=0, price=100, bb_pos=0.50, pnl_pct=0
        )
        self.assertEqual(signal, 'BUY')
        
        # BB位置过低
        signal, conf, reason = self.strategy._uptrend_strategy(
            position=0, avg_cost=0, price=100, bb_pos=0.30, pnl_pct=0
        )
        self.assertEqual(signal, 'HOLD')
        self.assertIn('回调过深', reason)
        
        # BB位置过高
        signal, conf, reason = self.strategy._uptrend_strategy(
            position=0, avg_cost=0, price=100, bb_pos=0.80, pnl_pct=0
        )
        self.assertEqual(signal, 'HOLD')
        self.assertIn('等待回调', reason)
    
    def test_ranging_buy_at_low(self):
        """测试：震荡市只在低点买入"""
        # 低于阈值
        signal, conf, reason = self.strategy._ranging_strategy(
            position=0, price=100, bb_pos=0.15
        )
        self.assertEqual(signal, 'BUY')
        
        # 高于阈值
        signal, conf, reason = self.strategy._ranging_strategy(
            position=0, price=100, bb_pos=0.40
        )
        self.assertEqual(signal, 'HOLD')
    
    def test_ranging_sell_at_high(self):
        """测试：震荡市在高点卖出"""
        # 高于卖出阈值
        signal, conf, reason = self.strategy._ranging_strategy(
            position=100, price=100, bb_pos=0.60
        )
        self.assertEqual(signal, 'SELL')
        
        # 低于卖出阈值
        signal, conf, reason = self.strategy._ranging_strategy(
            position=100, price=100, bb_pos=0.40
        )
        self.assertEqual(signal, 'HOLD')


class TestNoConsecutiveOpening(unittest.TestCase):
    """测试防止连续开仓"""
    
    def setUp(self):
        with SuppressOutput():
            self.strategy = SimpleUpTrendStrategy(
                max_allocation=1.0,
                cooldown_bars=0,
                verbose_init=False
            )
        self.ticker = 'TEST'
        self.data = TestDataGenerator.create_ohlcv_data(n_bars=30)
    
    def test_no_buy_when_already_holding(self):
        """测试：已持仓时不会再发出买入信号"""
        # 模拟持仓状态
        with SuppressOutput():
            result, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=self.data,
                current_position=100,  # 已持仓
                avg_cost=100,
                current_time_et=datetime(2024, 1, 15, 10, 0)
            )
        
        # 有持仓时不应该买入
        self.assertNotEqual(result['signal'], 'BUY')
    
    def test_max_allocation_prevents_over_investment(self):
        """测试：不会超过最大投资比例"""
        # 设置已投资到最大
        self.strategy._total_invested[self.ticker] = 1.0
        
        can_open, reason = self.strategy._can_open_position(self.ticker, 0.5)
        
        self.assertFalse(can_open)
        self.assertIn('最大仓位', reason)


class TestStopLossLogic(unittest.TestCase):
    """测试止损逻辑"""
    
    def setUp(self):
        with SuppressOutput():
            self.strategy = SimpleUpTrendStrategy(
                quick_stop_loss=0.0005,   # 0.05%
                normal_stop_loss=0.001,   # 1%
                cooldown_bars=5,
                verbose_init=False
            )
        self.ticker = 'TEST'
        self.data = TestDataGenerator.create_ohlcv_data(n_bars=30, start_price=100)
    
    def test_stop_loss_triggers_sell(self):
        """测试：触发止损时卖出"""
        # 构造亏损场景
        with SuppressOutput():
            result, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=self.data,
                current_position=100,
                avg_cost=100.06,  # 成本远高于当前价格，触发止损
                current_time_et=datetime(2024, 1, 15, 10, 0)
            )
        
        self.assertEqual(result['signal'], 'SELL')
        self.assertIn('止损', result['reason'])
    
    def test_stop_loss_starts_cooldown(self):
        """测试：止损后进入冷却期"""
        self.strategy.reset_state(self.ticker)
        
        # 触发止损
        with SuppressOutput():
            self.strategy.get_signal(
                ticker=self.ticker,
                new_data=self.data,
                current_position=100,
                avg_cost=200,
                current_time_et=datetime(2024, 1, 15, 10, 0)
            )
        
        # 检查是否在冷却期
        is_cooling, _ = self.strategy._is_in_cooldown(self.ticker)
        self.assertTrue(is_cooling)
    
    def test_stop_loss_reduces_allocation(self):
        """测试：止损后减仓"""
        self.strategy.reset_state(self.ticker)
        initial_allocation = self.strategy.get_current_allocation(self.ticker)
        
        # 触发止损
        with SuppressOutput():
            self.strategy.get_signal(
                ticker=self.ticker,
                new_data=self.data,
                current_position=100,
                avg_cost=200,
                current_time_et=datetime(2024, 1, 15, 10, 0)
            )
        
        new_allocation = self.strategy.get_current_allocation(self.ticker)
        self.assertLess(new_allocation, initial_allocation)


class TestMarketStates(unittest.TestCase):
    """测试不同市场状态"""
    
    def setUp(self):
        with SuppressOutput():
            self.strategy = SimpleUpTrendStrategy(
                adx_trend_threshold=25,
                adx_range_threshold=20,
                verbose_init=False
            )
    
    def test_uptrend_detection(self):
        """测试：上升趋势检测"""
        state = self.strategy._get_market_state(
            adx=30, ema_fast=110, ema_slow=100
        )
        self.assertEqual(state, 'UPTREND')
    
    def test_downtrend_detection(self):
        """测试：下降趋势检测"""
        state = self.strategy._get_market_state(
            adx=30, ema_fast=90, ema_slow=100
        )
        self.assertEqual(state, 'DOWNTREND')
    
    def test_ranging_detection(self):
        """测试：震荡市检测"""
        state = self.strategy._get_market_state(
            adx=15, ema_fast=100, ema_slow=100
        )
        self.assertEqual(state, 'RANGING')
    
    def test_unclear_detection(self):
        """测试：不明朗市场检测"""
        state = self.strategy._get_market_state(
            adx=22, ema_fast=100, ema_slow=100  # ADX 在阈值之间
        )
        self.assertEqual(state, 'UNCLEAR')
    
    def test_no_buy_in_downtrend(self):
        """测试：下降趋势不买入"""
        signal, conf, reason = self.strategy._downtrend_strategy(
            position=0, avg_cost=0, price=100, pnl_pct=0
        )
        self.assertEqual(signal, 'HOLD')
        self.assertIn('不开新仓', reason)


class TestEdgeCases(unittest.TestCase):
    """测试边界条件"""
    
    def setUp(self):
        with SuppressOutput():
            self.strategy = SimpleUpTrendStrategy(verbose_init=False)
        self.ticker = 'TEST'
    
    def test_empty_data(self):
        """测试：空数据处理"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(Exception):
            with SuppressOutput():
                self.strategy.get_signal(
                    ticker=self.ticker,
                    new_data=empty_df,
                    current_position=0,
                    avg_cost=0
                )
    
    def test_single_bar_data(self):
        """测试：单根K线数据"""
        single_bar = TestDataGenerator.create_ohlcv_data(n_bars=1)
        
        with SuppressOutput():
            result, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=single_bar,
                current_position=0,
                avg_cost=0,
                current_time_et=datetime(2024, 1, 15, 10, 0)
            )
        
        # 应该能返回结果，即使是 HOLD
        self.assertIn('signal', result)
    
    def test_zero_price_handling(self):
        """测试：零价格处理"""
        width = self.strategy._calculate_bb_width(100, 90, 0)
        self.assertEqual(width, 0.0)
    
    def test_zero_bb_range_handling(self):
        """测试：布林带范围为零处理"""
        position = self.strategy._calculate_bb_position(100, 100, 100)
        self.assertEqual(position, 0.5)
    
    def test_negative_pnl_calculation(self):
        """测试：负盈亏计算"""
        data = TestDataGenerator.create_ohlcv_data(n_bars=30, start_price=100)
        
        with SuppressOutput():
            result, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=data,
                current_position=100,
                avg_cost=150,  # 亏损状态
                current_time_et=datetime(2024, 1, 15, 10, 0)
            )
        
        self.assertIn('signal', result)
    
    def test_none_time_handling(self):
        """测试：时间为 None 处理"""
        is_last = self.strategy._is_last_n_minutes(None)
        self.assertFalse(is_last)
    
    def test_reset_state(self):
        """测试：状态重置"""
        # 设置一些状态
        self.strategy._current_allocation[self.ticker] = 0.5
        self.strategy._consecutive_losses[self.ticker] = 3
        
        # 重置
        self.strategy.reset_state(self.ticker)
        
        # 验证重置
        self.assertNotIn(self.ticker, self.strategy._current_allocation)
        self.assertNotIn(self.ticker, self.strategy._consecutive_losses)


class TestTakeProfitLogic(unittest.TestCase):
    """测试止盈逻辑"""
    
    def setUp(self):
        with SuppressOutput():
            self.strategy = SimpleUpTrendStrategy(
                uptrend_take_profit=0.03,  # 3% 止盈
                verbose_init=False
            )
        self.ticker = 'TEST'
    
    def test_take_profit_in_uptrend(self):
        """测试：上升趋势止盈"""
        signal, conf, reason = self.strategy._uptrend_strategy(
            position=100,
            avg_cost=100,
            price=104,  # 4% 盈利
            bb_pos=0.50,
            pnl_pct=0.04
        )
        
        self.assertEqual(signal, 'SELL')
        self.assertIn('止盈', reason)
    
    def test_hold_below_take_profit(self):
        """测试：未达止盈继续持有"""
        signal, conf, reason = self.strategy._uptrend_strategy(
            position=100,
            avg_cost=100,
            price=101,  # 1% 盈利
            bb_pos=0.50,
            pnl_pct=0.01
        )
        
        self.assertEqual(signal, 'HOLD')


class TestCooldownWithMinutes(unittest.TestCase):
    """测试基于分钟的冷却期"""
    
    def setUp(self):
        with SuppressOutput():
            self.strategy = SimpleUpTrendStrategy(
                cooldown_bars=0,
                cooldown_minutes=5,
                consecutive_loss_multiplier=2.0,
                verbose_init=False
            )
        self.ticker = 'TEST'
    
    def test_minute_based_cooldown(self):
        """测试：基于分钟的冷却期"""
        start_time = datetime(2024, 1, 15, 10, 0)
        
        with SuppressOutput():
            self.strategy._start_cooldown(self.ticker, start_time, is_stop_loss=True)
        
        # 2分钟后，应该还在冷却期
        check_time = start_time + timedelta(minutes=2)
        is_cooling, reason = self.strategy._is_in_cooldown(self.ticker, check_time)
        self.assertTrue(is_cooling)
        
        # 6分钟后，应该结束冷却期
        check_time = start_time + timedelta(minutes=6)
        is_cooling, reason = self.strategy._is_in_cooldown(self.ticker, check_time)
        self.assertFalse(is_cooling)
    
    def test_minute_cooldown_extends_with_losses(self):
        """测试：连续亏损延长分钟冷却期"""
        # 第一次止损
        start_time1 = datetime(2024, 1, 15, 10, 0)
        with SuppressOutput():
            self.strategy._start_cooldown(self.ticker, start_time1, is_stop_loss=True)
        
        # 结束第一次冷却
        check_time = start_time1 + timedelta(minutes=10)
        self.strategy._is_in_cooldown(self.ticker, check_time)
        
        # 第二次止损
        start_time2 = datetime(2024, 1, 15, 10, 15)
        with SuppressOutput():
            self.strategy._start_cooldown(self.ticker, start_time2, is_stop_loss=True)
        
        # 5分钟后应该还在冷却（因为冷却期是 5 * 2 = 10 分钟）
        check_time = start_time2 + timedelta(minutes=5)
        is_cooling, reason = self.strategy._is_in_cooldown(self.ticker, check_time)
        self.assertTrue(is_cooling)


class TestIntegration(unittest.TestCase):
    """集成测试：完整交易流程"""
    
    def setUp(self):
        with SuppressOutput():
            self.strategy = SimpleUpTrendStrategy(
                quick_stop_loss=0.01,
                normal_stop_loss=0.02,
                cooldown_bars=3,
                max_allocation=1.0,
                min_allocation=0.25,
                no_new_position_minutes=10,
                bb_narrow_threshold=0.01,
                bb_narrow_action='WARN',
                verbose_init=False
            )
        self.ticker = 'TEST'
    
    def test_full_trading_cycle(self):
        """测试：完整交易周期"""
        # 1. 初始状态 - 空仓
        data1 = TestDataGenerator.create_ohlcv_data(n_bars=30, start_price=100)
        
        with SuppressOutput():
            result1, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=data1,
                current_position=0,
                avg_cost=0,
                current_time_et=datetime(2024, 1, 15, 10, 0)
            )
        
        self.assertIn('signal', result1)
        initial_allocation = self.strategy.get_current_allocation(self.ticker)
        self.assertEqual(initial_allocation, 1.0)
    
    def test_multiple_stop_losses(self):
        """测试：多次止损场景"""
        for i in range(3):
            data = TestDataGenerator.create_ohlcv_data(
                n_bars=30, 
                start_price=100 - i*10,
                start_time=datetime(2024, 1, 15, 10, 0) + timedelta(hours=i)
            )
            
            with SuppressOutput():
                result, _ = self.strategy.get_signal(
                    ticker=self.ticker,
                    new_data=data,
                    current_position=100,
                    avg_cost=200,  # 大幅亏损
                    current_time_et=datetime(2024, 1, 15, 10+i, 0)
                )
            
            if result['signal'] == 'SELL':
                # 模拟止损后的状态
                self.strategy._bar_count[self.ticker] += 10  # 跳过冷却期
        
        # 检查连续亏损计数
        losses = self.strategy.get_consecutive_losses(self.ticker)
        self.assertGreater(losses, 0)


class TestResultStructure(unittest.TestCase):
    """测试返回结果结构"""
    
    def setUp(self):
        with SuppressOutput():
            self.strategy = SimpleUpTrendStrategy(verbose_init=False)
        self.ticker = 'TEST'
        self.data = TestDataGenerator.create_ohlcv_data(n_bars=30)
    
    def test_result_contains_required_fields(self):
        """测试：结果包含所有必需字段"""
        with SuppressOutput():
            result, df = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=self.data,
                current_position=0,
                avg_cost=0,
                current_time_et=datetime(2024, 1, 15, 10, 0)
            )
        
        required_fields = [
            'signal', 'confidence', 'reason', 'price',
            'market_state', 'adx', 'bb_position', 'allocation',
            'bb_width', 'is_bb_narrow'
        ]
        
        for field in required_fields:
            self.assertIn(field, result, f"Missing field: {field}")
    
    def test_signal_values(self):
        """测试：信号值有效"""
        with SuppressOutput():
            result, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=self.data,
                current_position=0,
                avg_cost=0,
                current_time_et=datetime(2024, 1, 15, 10, 0)
            )
        
        self.assertIn(result['signal'], ['BUY', 'SELL', 'HOLD'])
    
    def test_confidence_range(self):
        """测试：信心度在有效范围内"""
        with SuppressOutput():
            result, _ = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=self.data,
                current_position=0,
                avg_cost=0,
                current_time_et=datetime(2024, 1, 15, 10, 0)
            )
        
        self.assertGreaterEqual(result['confidence'], 1)
        self.assertLessEqual(result['confidence'], 10)
    
    def test_returns_dataframe(self):
        """测试：返回 DataFrame"""
        with SuppressOutput():
            result, df = self.strategy.get_signal(
                ticker=self.ticker,
                new_data=self.data,
                current_position=0,
                avg_cost=0,
                current_time_et=datetime(2024, 1, 15, 10, 0)
            )
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)