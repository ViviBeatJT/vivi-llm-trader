# src/visualization/simple_chart_visualizer.py

"""
简单图表可视化工具 - 无需后台服务器

特点：
- 生成静态 HTML 文件
- 每次更新时保存
- 浏览器手动刷新查看（或自动刷新插件）
- 无需 Dash 服务器，无线程问题
- 更稳定，更简单

使用：
1. 创建 visualizer
2. 每次策略运行后调用 update()
3. 自动生成 HTML 文件
4. 在浏览器中打开查看

v2 更新：
- 新增 BB Width（布林带宽度）子图
- BB Width = (BB_UPPER - BB_LOWER) / Price * 100%
- 显示窄幅阈值参考线
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Optional, Dict, List
import os


class SimpleChartVisualizer:
    """
    简单图表可视化工具

    每次更新时生成新的 HTML 文件
    浏览器手动刷新即可查看最新状态
    """

    def __init__(self,
                 ticker: str,
                 output_file: str = "trading_chart.html",
                 auto_open: bool = False,
                 initial_capital: float = 100000.0,
                 bb_narrow_threshold: float = 0.01):  # 新增：BB窄幅阈值
        """
        初始化可视化工具

        Args:
            ticker: 股票代码
            output_file: 输出 HTML 文件路径
            auto_open: 首次创建时是否自动打开浏览器
            initial_capital: 初始资金
            bb_narrow_threshold: BB窄幅阈值（默认2%）
        """
        self.ticker = ticker
        self.output_file = output_file
        self.auto_open = auto_open
        self.initial_capital = initial_capital
        self.bb_narrow_threshold = bb_narrow_threshold

        # 数据存储
        self.market_data: Optional[pd.DataFrame] = None
        self.trade_log: List[Dict] = []
        self.equity_history: List[Dict] = []
        self.position_history: List[Dict] = []

        # 统计
        self.stats = {
            'total_trades': 0,
            'current_position': 0,
            'current_equity': 0,
            'net_pnl': 0,
            'last_update': None,
            'current_bb_width': 0  # 新增：当前BB宽度
        }

        # 首次打开标志
        self._first_save = True

        print(f"📊 Simple Chart Visualizer 初始化")
        print(f"   输出文件: {output_file}")
        print(f"   BB窄幅阈值: {bb_narrow_threshold*100:.1f}%")
        print(f"   刷新方式: 浏览器手动刷新（或使用自动刷新插件）")

    def update_data(self,
                    market_data: pd.DataFrame,
                    trade_log: pd.DataFrame,
                    current_equity: float,
                    current_position: float,
                    timestamp: datetime):
        """
        更新所有数据并重新生成图表

        Args:
            market_data: 策略的完整 DataFrame（包含所有技术指标）
            trade_log: 交易记录 DataFrame
            current_equity: 当前权益
            current_position: 当前持仓
            timestamp: 当前时间
        """
        # 更新市场数据
        self.market_data = market_data.copy()

        # 计算 BB Width（如果有布林带数据）
        if 'BB_UPPER' in self.market_data.columns and 'BB_LOWER' in self.market_data.columns:
            self.market_data['BB_WIDTH'] = (
                (self.market_data['BB_UPPER'] - self.market_data['BB_LOWER'])
                / self.market_data['close'] * 100
            )
            self.stats['current_bb_width'] = self.market_data['BB_WIDTH'].iloc[-1]

        # 更新交易记录
        if not trade_log.empty:
            self.trade_log = trade_log.to_dict('records')
            self.stats['total_trades'] = len(trade_log)

        # 更新权益
        self.equity_history.append({
            'time': timestamp,
            'equity': current_equity
        })
        self.stats['current_equity'] = current_equity
        self.stats['net_pnl'] = current_equity - self.initial_capital

        # 更新持仓
        self.position_history.append({
            'time': timestamp,
            'position': current_position
        })
        self.stats['current_position'] = current_position

        # 更新时间
        self.stats['last_update'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')

        # 生成图表
        self.save_chart()

    def save_chart(self):
        """生成并保存 HTML 图表"""
        fig = self._create_figure()

        # 保存为 HTML
        fig.write_html(
            self.output_file,
            config={'displayModeBar': True, 'scrollZoom': True}
        )

        bb_width_str = f"BB宽: {self.stats['current_bb_width']:.2f}%" if self.stats['current_bb_width'] else ""
        print(f"✅ 图表已更新: {self.output_file}")
        print(f"   权益: ${self.stats['current_equity']:,.0f} | "
              f"持仓: {self.stats['current_position']:.0f} | "
              f"交易: {self.stats['total_trades']} | "
              f"{bb_width_str}")

        # 首次保存时自动打开
        if self._first_save and self.auto_open:
            import webbrowser
            abs_path = os.path.abspath(self.output_file)
            webbrowser.open(f'file://{abs_path}')
            print(f"📂 已在浏览器中打开")
            self._first_save = False

    def _create_figure(self) -> go.Figure:
        """创建完整图表"""
        # 创建子图 - 新增 BB Width 行
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.40, 0.12, 0.12, 0.18, 0.12],  # 调整高度比例
            subplot_titles=(
                f'{self.ticker} - Candlestick & Bollinger Bands (Upper/Middle/Lower)',
                'Volume',
                'BB Width (%)',  # 新增
                'Equity Curve',
                'Position Status (Long/Short/Flat)'
            )
        )

        # === 1. 布林带 + 蜡烛图 ===
        if self.market_data is not None and not self.market_data.empty:
            df = self.market_data

            # 检查是否有布林带数据
            has_bb = all(col in df.columns for col in [
                         'BB_UPPER', 'SMA', 'BB_LOWER'])

            if has_bb:
                print(f"✅ 布林带数据存在，准备绘制三条线")
            else:
                print(f"⚠️ 警告: 缺少布林带数据列")
                print(f"   存在的列: {df.columns.tolist()}")

            # 📊 第一步：绘制布林带（作为背景）

            # 1. 布林带填充区域（最下层，浅色背景）
            if 'BB_UPPER' in df.columns and 'BB_LOWER' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_UPPER'],
                        line=dict(width=0),
                        mode='lines',
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_LOWER'],
                        line=dict(width=0),
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(173,216,230,0.15)',  # 淡蓝色填充
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )

            # 2. 上轨（红色虚线 - 明显）
            if 'BB_UPPER' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_UPPER'],
                        name='BB Upper',
                        line=dict(
                            color='red',
                            width=2,
                            dash='dash'
                        ),
                        mode='lines',
                        showlegend=True
                    ),
                    row=1, col=1
                )

            # 3. 中线 / SMA（蓝色实线 - 明显）
            if 'SMA' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA'],
                        name='BB Middle (SMA)',
                        line=dict(
                            color='blue',
                            width=2.5
                        ),
                        mode='lines',
                        showlegend=True
                    ),
                    row=1, col=1
                )

            # 4. 下轨（绿色虚线 - 明显）
            if 'BB_LOWER' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_LOWER'],
                        name='BB Lower',
                        line=dict(
                            color='green',
                            width=2,
                            dash='dash'
                        ),
                        mode='lines',
                        showlegend=True
                    ),
                    row=1, col=1
                )

            # 🕯️ 第二步：绘制蜡烛图（在最上层）
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price',
                    increasing_line_color='#26A69A',
                    decreasing_line_color='#EF5350',
                    increasing_fillcolor='#26A69A',
                    decreasing_fillcolor='#EF5350',
                    showlegend=True
                ),
                row=1, col=1
            )

            # === 2. 成交量 ===
            if 'volume' in df.columns:
                colors = ['#26A69A' if df['close'].iloc[i] >= df['open'].iloc[i]
                          else '#EF5350' for i in range(len(df))]
                fig.add_trace(
                    go.Bar(x=df.index, y=df['volume'],
                           name='Volume', marker_color=colors, opacity=0.5,
                           showlegend=False),
                    row=2, col=1
                )

            # === 3. BB Width（新增）===
            if 'BB_WIDTH' in df.columns:
                # BB宽度线
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_WIDTH'],
                        name='BB Width',
                        line=dict(color='purple', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(128,0,128,0.1)',
                        showlegend=True
                    ),
                    row=3, col=1
                )

                # 窄幅阈值线（红色虚线）
                threshold_pct = self.bb_narrow_threshold * 100
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[0], df.index[-1]],
                        y=[threshold_pct, threshold_pct],
                        name=f'Narrow Threshold ({threshold_pct:.1f}%)',
                        line=dict(color='red', width=1.5, dash='dash'),
                        showlegend=True
                    ),
                    row=3, col=1
                )

                # 高亮显示窄幅区域（BB Width < 阈值时）
                narrow_mask = df['BB_WIDTH'] < threshold_pct
                if narrow_mask.any():
                    narrow_df = df[narrow_mask]
                    fig.add_trace(
                        go.Scatter(
                            x=narrow_df.index,
                            y=narrow_df['BB_WIDTH'],
                            name='Narrow BB',
                            mode='markers',
                            marker=dict(
                                color='red',
                                size=8,
                                symbol='circle',
                                opacity=0.7
                            ),
                            showlegend=True
                        ),
                        row=3, col=1
                    )

        # === 4. 交易信号 ===
        if self.trade_log:
            # 开仓
            buy_trades = [t for t in self.trade_log if t['type']
                          in ['BUY', 'SHORT']]
            if buy_trades:
                times = [t['time'] for t in buy_trades]
                prices = [t['price'] for t in buy_trades]
                types = [t['type'] for t in buy_trades]
                colors = ['green' if t == 'BUY' else 'red' for t in types]
                symbols = ['triangle-up' if t ==
                           'BUY' else 'triangle-down' for t in types]

                fig.add_trace(
                    go.Scatter(
                        x=times, y=prices,
                        name='Buy/Short',
                        mode='markers',
                        marker=dict(size=15, color=colors, symbol=symbols,
                                    line=dict(color='black', width=2))
                    ),
                    row=1, col=1
                )

            # 平仓
            sell_trades = [
                t for t in self.trade_log if t['type'] in ['SELL', 'COVER']]
            if sell_trades:
                times = [t['time'] for t in sell_trades]
                prices = [t['price'] for t in sell_trades]
                types = [t['type'] for t in sell_trades]
                colors = ['orange' if t == 'SELL' else 'purple' for t in types]
                symbols = ['triangle-down' if t ==
                           'SELL' else 'triangle-up' for t in types]

                fig.add_trace(
                    go.Scatter(
                        x=times, y=prices,
                        name='Sell/Cover',
                        mode='markers',
                        marker=dict(size=15, color=colors, symbol=symbols,
                                    line=dict(color='black', width=2))
                    ),
                    row=1, col=1
                )

        # === 5. 权益曲线 ===
        if self.equity_history:
            times = [e['time'] for e in self.equity_history]
            values = [e['equity'] for e in self.equity_history]

            fig.add_trace(
                go.Scatter(
                    x=times, y=values,
                    name='Equity',
                    line=dict(color='blue', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0,100,255,0.1)',
                    showlegend=False
                ),
                row=4, col=1
            )

            # 初始资金线
            fig.add_trace(
                go.Scatter(
                    x=[times[0], times[-1]],
                    y=[self.initial_capital, self.initial_capital],
                    name='Initial Capital',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False
                ),
                row=4, col=1
            )

        # === 6. 持仓 ===
        if self.position_history:
            times = [p['time'] for p in self.position_history]
            values = [p['position'] for p in self.position_history]
            colors = ['#26A69A' if v > 0 else '#EF5350' if v <
                      0 else 'gray' for v in values]

            fig.add_trace(
                go.Bar(x=times, y=values, name='Position',
                       marker_color=colors, opacity=0.7,
                       showlegend=False),
                row=5, col=1
            )

            # 零线
            fig.add_trace(
                go.Scatter(
                    x=[times[0], times[-1]], y=[0, 0],
                    line=dict(color='black', width=1),
                    showlegend=False
                ),
                row=5, col=1
            )

        # 更新坐标轴标签
        fig.update_xaxes(title_text="Time", row=5, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="BB Width (%)", row=3, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=4, col=1)
        fig.update_yaxes(title_text="Shares", row=5, col=1)

        # 添加标题 - 包含BB Width信息
        bb_width_str = f"BB Width: {self.stats['current_bb_width']:.2f}%" if self.stats['current_bb_width'] else ""
        narrow_warning = " ⚠️NARROW" if self.stats['current_bb_width'] and self.stats[
            'current_bb_width'] < self.bb_narrow_threshold * 100 else ""

        title_text = (
            f"<b>{self.ticker} Trading Dashboard</b><br>"
            f"<sub>Trades: {self.stats['total_trades']} | "
            f"Position: {self.stats['current_position']:.0f} | "
            f"Equity: ${self.stats['current_equity']:,.0f} | "
            f"P&L: ${self.stats['net_pnl']:+,.0f} | "
            f"{bb_width_str}{narrow_warning} | "
            f"Updated: {self.stats['last_update']}</sub>"
        )

        fig.update_layout(
            title=title_text,
            height=1000,  # 增加高度以容纳新子图
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )

        # 优化 Y 轴显示
        fig.update_yaxes(automargin=True, row=1, col=1)
        fig.update_yaxes(automargin=True, row=3, col=1)

        return fig

    def set_initial_capital(self, capital: float):
        """设置初始资金"""
        self.initial_capital = capital

    def set_bb_narrow_threshold(self, threshold: float):
        """设置BB窄幅阈值"""
        self.bb_narrow_threshold = threshold


# ==================== 使用示例 ====================

if __name__ == '__main__':
    import numpy as np
    from datetime import timedelta

    # 创建可视化工具
    visualizer = SimpleChartVisualizer(
        ticker='TSLA',
        output_file='test_chart.html',
        auto_open=True,
        initial_capital=100000.0,
        bb_narrow_threshold=0.02  # 2%
    )

    # 模拟数据
    start_time = datetime(2024, 12, 5, 9, 30)

    for i in range(10):
        # 创建市场数据
        time_index = pd.date_range(
            start=start_time, periods=50+i*10, freq='5min')
        prices = 100 + np.cumsum(np.random.randn(50+i*10) * 0.5)

        # 计算布林带
        sma = pd.Series(prices).rolling(20).mean()
        std = pd.Series(prices).rolling(20).std()

        df = pd.DataFrame({
            'close': prices,
            'open': prices - 0.1,
            'high': prices + 0.3,
            'low': prices - 0.3,
            'volume': np.random.randint(1000, 5000, 50+i*10),
            'SMA': sma,
            'BB_UPPER': sma + 2 * std,
            'BB_LOWER': sma - 2 * std,
        }, index=time_index)

        # 模拟交易记录
        trades = []
        if i > 0:
            for j in range(i):
                trades.append({
                    'time': time_index[j*10],
                    'type': ['BUY', 'SELL', 'SHORT', 'COVER'][j % 4],
                    'qty': 10,
                    'price': prices[j*10],
                    'fee': 5.0,
                    'net_pnl': np.random.randn() * 100
                })

        trade_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        # 更新图表
        current_equity = 100000 + i * 500 + np.random.randn() * 200
        current_position = [0, 50, 0, -40, 0, 30, 0, -20, 0, 10][i]

        visualizer.update_data(
            market_data=df,
            trade_log=trade_df,
            current_equity=current_equity,
            current_position=current_position,
            timestamp=time_index[-1]
        )

        print(f"更新 {i+1}/10 完成，按回车继续...")
        input()

    print("\n✅ 测试完成！查看 test_chart.html")
