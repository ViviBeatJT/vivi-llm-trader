# src/visualization/live_trading_dashboard.py

"""
实时交易仪表板 - Web 界面

使用 Plotly Dash 创建实时更新的交易图表
适用于回测和实盘交易

运行方式：
1. 在单独的线程中启动 dashboard
2. 主程序更新数据
3. 浏览器自动刷新显示

访问: http://localhost:8050
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import threading
import time
from datetime import datetime
from typing import Optional, Dict, List
import json


class TradingDashboard:
    """
    实时交易仪表板
    
    特点：
    - Web 界面，浏览器访问
    - 自动刷新（1秒间隔）
    - 显示价格、布林带、交易信号、权益曲线、持仓状态
    - 支持回测和实盘
    """
    
    def __init__(self, 
                 port: int = 8050,
                 update_interval: int = 1000,  # 毫秒
                 ticker: str = "UNKNOWN"):
        """
        初始化仪表板
        
        Args:
            port: Web 服务器端口
            update_interval: 图表更新间隔（毫秒）
            ticker: 股票代码
        """
        self.port = port
        self.update_interval = update_interval
        self.ticker = ticker
        
        # 数据存储（线程安全）
        self.market_data: Optional[pd.DataFrame] = None
        self.trade_log: List[Dict] = []
        self.equity_history: List[Dict] = []
        self.position_history: List[Dict] = []
        self.initial_capital: float = 100000.0
        
        # 统计信息
        self.stats = {
            'total_trades': 0,
            'current_position': 0,
            'current_equity': 0,
            'net_pnl': 0,
            'last_update': None
        }
        
        # Dash 应用
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
        # 服务器线程
        self.server_thread: Optional[threading.Thread] = None
        self.running = False
    
    def _setup_layout(self):
        """设置仪表板布局"""
        self.app.layout = html.Div([
            html.H1(
                f'🚀 Real-Time Trading Dashboard - {self.ticker}',
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'marginBottom': '20px',
                    'fontFamily': 'Arial, sans-serif'
                }
            ),
            
            # 统计信息栏
            html.Div(id='stats-bar', style={
                'backgroundColor': '#ecf0f1',
                'padding': '15px',
                'marginBottom': '20px',
                'borderRadius': '5px',
                'display': 'flex',
                'justifyContent': 'space-around',
                'fontFamily': 'monospace'
            }),
            
            # 主图表
            dcc.Graph(id='main-chart', style={'height': '70vh'}),
            
            # 自动更新
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self):
        """设置回调函数"""
        @self.app.callback(
            [Output('main-chart', 'figure'),
             Output('stats-bar', 'children')],
            Input('interval-component', 'n_intervals')
        )
        def update_dashboard(n):
            """更新仪表板"""
            fig = self._create_figure()
            stats_display = self._create_stats_display()
            return fig, stats_display
    
    def _create_figure(self) -> go.Figure:
        """创建主图表"""
        # 创建子图（4行1列）
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.15, 0.2, 0.15],
            subplot_titles=(
                'Price & Bollinger Bands',
                'Volume',
                'Equity Curve',
                'Position Status'
            )
        )
        
        # === 1. 价格 + 布林带 ===
        if self.market_data is not None and not self.market_data.empty:
            df = self.market_data
            
            # 价格线
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['close'],
                    name='Close Price',
                    line=dict(color='black', width=2),
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # 布林带
            if 'SMA' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA'],
                        name='SMA (Middle)',
                        line=dict(color='blue', width=1, dash='dash'),
                        mode='lines'
                    ),
                    row=1, col=1
                )
            
            if 'BB_UPPER' in df.columns and 'BB_LOWER' in df.columns:
                # 上轨
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_UPPER'],
                        name='BB Upper',
                        line=dict(color='red', width=1, dash='dash'),
                        mode='lines',
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # 下轨
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_LOWER'],
                        name='BB Lower',
                        line=dict(color='green', width=1, dash='dash'),
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)'
                    ),
                    row=1, col=1
                )
            
            # 成交量
            if 'volume' in df.columns:
                colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] 
                         else 'red' for i in range(len(df))]
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.5
                    ),
                    row=2, col=1
                )
        
        # === 2. 交易信号标记 ===
        if self.trade_log:
            buy_trades = [t for t in self.trade_log if t['type'] in ['BUY', 'SHORT']]
            sell_trades = [t for t in self.trade_log if t['type'] in ['SELL', 'COVER']]
            
            # 买入/做空
            if buy_trades:
                buy_times = [t['time'] for t in buy_trades]
                buy_prices = [t['price'] for t in buy_trades]
                buy_types = [t['type'] for t in buy_trades]
                buy_colors = ['green' if t == 'BUY' else 'red' for t in buy_types]
                buy_symbols = ['triangle-up' if t == 'BUY' else 'triangle-down' for t in buy_types]
                
                fig.add_trace(
                    go.Scatter(
                        x=buy_times,
                        y=buy_prices,
                        name='Buy/Short',
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=buy_colors,
                            symbol=buy_symbols,
                            line=dict(color='black', width=1)
                        )
                    ),
                    row=1, col=1
                )
            
            # 卖出/平空
            if sell_trades:
                sell_times = [t['time'] for t in sell_trades]
                sell_prices = [t['price'] for t in sell_trades]
                sell_types = [t['type'] for t in sell_trades]
                sell_colors = ['orange' if t == 'SELL' else 'purple' for t in sell_types]
                sell_symbols = ['triangle-down' if t == 'SELL' else 'triangle-up' for t in sell_types]
                
                fig.add_trace(
                    go.Scatter(
                        x=sell_times,
                        y=sell_prices,
                        name='Sell/Cover',
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=sell_colors,
                            symbol=sell_symbols,
                            line=dict(color='black', width=1)
                        )
                    ),
                    row=1, col=1
                )
        
        # === 3. 权益曲线 ===
        if self.equity_history:
            equity_times = [e['time'] for e in self.equity_history]
            equity_values = [e['equity'] for e in self.equity_history]
            
            fig.add_trace(
                go.Scatter(
                    x=equity_times,
                    y=equity_values,
                    name='Equity',
                    line=dict(color='blue', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0,100,255,0.1)'
                ),
                row=3, col=1
            )
            
            # 初始资金线
            if equity_times:
                fig.add_trace(
                    go.Scatter(
                        x=[equity_times[0], equity_times[-1]],
                        y=[self.initial_capital, self.initial_capital],
                        name='Initial Capital',
                        line=dict(color='gray', width=1, dash='dash'),
                        mode='lines'
                    ),
                    row=3, col=1
                )
        
        # === 4. 持仓状态 ===
        if self.position_history:
            pos_times = [p['time'] for p in self.position_history]
            pos_values = [p['position'] for p in self.position_history]
            pos_colors = ['green' if p > 0 else 'red' if p < 0 else 'gray' for p in pos_values]
            
            fig.add_trace(
                go.Bar(
                    x=pos_times,
                    y=pos_values,
                    name='Position',
                    marker_color=pos_colors,
                    opacity=0.7
                ),
                row=4, col=1
            )
            
            # 零线
            if pos_times:
                fig.add_trace(
                    go.Scatter(
                        x=[pos_times[0], pos_times[-1]],
                        y=[0, 0],
                        line=dict(color='black', width=1),
                        mode='lines',
                        showlegend=False
                    ),
                    row=4, col=1
                )
        
        # 更新布局
        fig.update_xaxes(title_text="Time", row=4, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=3, col=1)
        fig.update_yaxes(title_text="Shares", row=4, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _create_stats_display(self) -> List:
        """创建统计信息显示"""
        def stat_box(label, value, color='black'):
            return html.Div([
                html.Div(label, style={'fontSize': '12px', 'color': '#7f8c8d'}),
                html.Div(value, style={'fontSize': '18px', 'fontWeight': 'bold', 'color': color})
            ], style={'textAlign': 'center'})
        
        # 净盈亏颜色
        pnl_color = 'green' if self.stats['net_pnl'] >= 0 else 'red'
        pnl_sign = '+' if self.stats['net_pnl'] >= 0 else ''
        
        # 持仓状态
        pos = self.stats['current_position']
        pos_str = f"{pos:.0f} (Long)" if pos > 0 else f"{abs(pos):.0f} (Short)" if pos < 0 else "0 (Flat)"
        pos_color = 'green' if pos > 0 else 'red' if pos < 0 else 'gray'
        
        return [
            stat_box('Total Trades', str(self.stats['total_trades'])),
            stat_box('Current Position', pos_str, pos_color),
            stat_box('Current Equity', f"${self.stats['current_equity']:,.0f}"),
            stat_box('Net P&L', f"{pnl_sign}${self.stats['net_pnl']:,.0f}", pnl_color),
            stat_box('Last Update', self.stats['last_update'] or 'N/A', '#95a5a6')
        ]
    
    # ==================== 数据更新方法 ====================
    
    def update_market_data(self, df: pd.DataFrame):
        """更新市场数据"""
        self.market_data = df.copy()
    
    def add_trade(self, trade: Dict):
        """添加交易记录"""
        self.trade_log.append(trade)
        self.stats['total_trades'] = len(self.trade_log)
    
    def update_equity(self, timestamp: datetime, equity: float):
        """更新权益"""
        self.equity_history.append({
            'time': timestamp,
            'equity': equity
        })
        self.stats['current_equity'] = equity
        self.stats['net_pnl'] = equity - self.initial_capital
    
    def update_position(self, timestamp: datetime, position: float):
        """更新持仓"""
        self.position_history.append({
            'time': timestamp,
            'position': position
        })
        self.stats['current_position'] = position
    
    def set_initial_capital(self, capital: float):
        """设置初始资金"""
        self.initial_capital = capital
        self.stats['current_equity'] = capital
    
    def update_stats(self, **kwargs):
        """更新统计信息"""
        self.stats.update(kwargs)
        self.stats['last_update'] = datetime.now().strftime('%H:%M:%S')
    
    # ==================== 服务器控制 ====================
    
    def start(self, debug: bool = False):
        """启动仪表板服务器（在单独线程中）"""
        if self.running:
            print("⚠️ Dashboard already running!")
            return
        
        def run_server():
            print(f"🚀 Starting dashboard at http://localhost:{self.port}")
            print(f"   Press Ctrl+C in main program to stop")
            self.app.run_server(
                debug=debug,
                host='0.0.0.0',
                port=self.port,
                use_reloader=False  # 必须关闭，否则线程会冲突
            )
        
        self.running = True
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # 等待服务器启动
        time.sleep(2)
        print(f"✅ Dashboard is live at: http://localhost:{self.port}")
    
    def stop(self):
        """停止仪表板（注意：Dash 服务器不易优雅关闭）"""
        self.running = False
        print("⏹️ Dashboard stopped")
    
    def is_running(self) -> bool:
        """检查服务器是否运行中"""
        return self.running


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 创建仪表板
    dashboard = TradingDashboard(ticker='TSLA', port=8050)
    
    # 设置初始资金
    dashboard.set_initial_capital(100000.0)
    
    # 启动服务器
    dashboard.start()
    
    # 模拟数据更新
    import numpy as np
    
    print("\n模拟交易数据（按 Ctrl+C 停止）...")
    
    try:
        for i in range(100):
            # 模拟市场数据
            time_index = pd.date_range(start='2024-12-05 09:30', periods=50+i, freq='1min')
            prices = 100 + np.cumsum(np.random.randn(50+i) * 0.5)
            
            df = pd.DataFrame({
                'close': prices,
                'open': prices - 0.1,
                'high': prices + 0.3,
                'low': prices - 0.3,
                'volume': np.random.randint(1000, 5000, 50+i),
                'SMA': pd.Series(prices).rolling(20).mean(),
                'BB_UPPER': pd.Series(prices).rolling(20).mean() + 2 * pd.Series(prices).rolling(20).std(),
                'BB_LOWER': pd.Series(prices).rolling(20).mean() - 2 * pd.Series(prices).rolling(20).std(),
            }, index=time_index)
            
            dashboard.update_market_data(df)
            
            # 模拟交易
            if i % 10 == 0 and i > 0:
                trade_type = np.random.choice(['BUY', 'SELL', 'SHORT', 'COVER'])
                dashboard.add_trade({
                    'time': time_index[-1],
                    'type': trade_type,
                    'price': prices[-1],
                    'qty': 10
                })
            
            # 更新权益
            equity = 100000 + np.random.randn() * 1000 + i * 50
            dashboard.update_equity(time_index[-1], equity)
            
            # 更新持仓
            position = np.random.choice([-100, -50, 0, 50, 100])
            dashboard.update_position(time_index[-1], position)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
        dashboard.stop()