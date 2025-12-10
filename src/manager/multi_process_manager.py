#!/usr/bin/env python3
# multi_process_manager.py

"""
多进程交易管理器

功能：
1. 批量启动多个 ticker 的交易进程
2. 查看运行状态
3. 停止指定进程
4. 查看实时日志

用法：
    # 启动多个进程
    python multi_process_manager.py start --tickers TSLA AAPL NVDA --strategy moderate
    
    # 查看状态
    python multi_process_manager.py status
    
    # 停止特定进程
    python multi_process_manager.py stop --ticker TSLA
    
    # 停止所有进程
    python multi_process_manager.py stop --all
    
    # 查看日志
    python multi_process_manager.py logs --ticker TSLA --follow
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import signal
import os


class MultiProcessManager:
    """多进程交易管理器"""
    
    def __init__(self):
        self.base_dir = Path("live_trading")
        self.pids_dir = self.base_dir / "pids"
        self.logs_dir = self.base_dir / "logs"
        self.charts_dir = self.base_dir / "charts"
        
        # 确保目录存在
        for dir_path in [self.pids_dir, self.logs_dir, self.charts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def start_process(self, ticker: str, strategy: str = 'moderate', 
                     mode: str = 'paper', interval: int = 60,
                     no_chart: bool = False):
        """启动单个交易进程"""
        
        cmd = [
            sys.executable, 'live_runner.py',
            '--ticker', ticker,
            '--strategy', strategy,
            '--mode', mode,
            '--interval', str(interval)
        ]
        
        if no_chart:
            cmd.append('--no-chart')
        
        print(f"🚀 启动 {ticker} ({strategy}, {mode})...")
        
        try:
            # 后台运行
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # 独立会话，不受父进程影响
            )
            
            # 等待一下确保进程启动
            time.sleep(2)
            
            # 检查是否成功启动
            process_id = f"{ticker}_{strategy}_{mode}"
            pid_file = self.pids_dir / f"{process_id}.pid"
            
            if pid_file.exists():
                print(f"   ✅ {ticker} 启动成功 (PID: {pid_file.read_text().strip()})")
                return True
            else:
                print(f"   ❌ {ticker} 启动失败")
                return False
                
        except Exception as e:
            print(f"   ❌ {ticker} 启动失败: {e}")
            return False
    
    def get_running_processes(self):
        """获取所有运行中的进程"""
        processes = []
        
        if not self.pids_dir.exists():
            return processes
        
        for pid_file in self.pids_dir.glob("*.pid"):
            try:
                pid = int(pid_file.read_text().strip())
                # 检查进程是否存在
                os.kill(pid, 0)
                
                # 解析进程信息
                process_id = pid_file.stem
                parts = process_id.split('_')
                if len(parts) >= 3:
                    ticker = parts[0]
                    strategy = parts[1]
                    mode = parts[2]
                    
                    processes.append({
                        'ticker': ticker,
                        'strategy': strategy,
                        'mode': mode,
                        'pid': pid,
                        'process_id': process_id,
                        'pid_file': pid_file
                    })
                    
            except (ProcessLookupError, ValueError, OSError):
                # 进程不存在，清理 PID 文件
                pid_file.unlink()
        
        return processes
    
    def show_status(self):
        """显示所有进程状态"""
        processes = self.get_running_processes()
        
        if not processes:
            print("\n📊 当前没有运行中的交易进程")
            return
        
        print(f"\n📊 运行中的交易进程 (共 {len(processes)} 个):")
        print("="*80)
        print(f"{'Ticker':<8} {'Strategy':<20} {'Mode':<12} {'PID':<8} {'日志文件'}")
        print("-"*80)
        
        for proc in processes:
            log_file = self.logs_dir / f"{proc['process_id']}.log"
            log_status = "✅" if log_file.exists() else "❌"
            
            print(f"{proc['ticker']:<8} {proc['strategy']:<20} {proc['mode']:<12} "
                  f"{proc['pid']:<8} {log_status}")
        
        print("="*80)
        print(f"\n💡 提示:")
        print(f"   查看日志: python {sys.argv[0]} logs --ticker <TICKER>")
        print(f"   停止进程: python {sys.argv[0]} stop --ticker <TICKER>")
        print(f"   查看图表: open live_trading/charts/<TICKER>_<strategy>_<mode>.html")
    
    def stop_process(self, ticker: str = None, strategy: str = None, 
                    mode: str = None, stop_all: bool = False):
        """停止进程"""
        processes = self.get_running_processes()
        
        if not processes:
            print("\n⚠️ 没有运行中的进程")
            return
        
        stopped_count = 0
        
        for proc in processes:
            should_stop = stop_all
            
            if not stop_all:
                if ticker and proc['ticker'] != ticker:
                    continue
                if strategy and proc['strategy'] != strategy:
                    continue
                if mode and proc['mode'] != mode:
                    continue
                should_stop = True
            
            if should_stop:
                print(f"⏹️  停止 {proc['ticker']} (PID: {proc['pid']})...")
                try:
                    os.kill(proc['pid'], signal.SIGTERM)
                    time.sleep(1)
                    
                    # 检查是否已停止
                    try:
                        os.kill(proc['pid'], 0)
                        # 还在运行，强制杀死
                        os.kill(proc['pid'], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    
                    # 清理 PID 文件
                    proc['pid_file'].unlink()
                    print(f"   ✅ {proc['ticker']} 已停止")
                    stopped_count += 1
                    
                except Exception as e:
                    print(f"   ❌ 停止失败: {e}")
        
        if stopped_count == 0:
            print("\n⚠️ 没有匹配的进程")
        else:
            print(f"\n✅ 已停止 {stopped_count} 个进程")
    
    def show_logs(self, ticker: str, follow: bool = False, lines: int = 50):
        """显示日志"""
        processes = self.get_running_processes()
        
        # 查找匹配的进程
        matched = [p for p in processes if p['ticker'] == ticker]
        
        if not matched:
            print(f"\n⚠️ 没有找到 {ticker} 的运行进程")
            
            # 显示历史日志文件
            log_files = list(self.logs_dir.glob(f"{ticker}_*.log"))
            if log_files:
                print(f"\n💡 找到历史日志文件:")
                for log_file in log_files:
                    print(f"   {log_file}")
                    if not follow:
                        print(f"\n最后 {lines} 行:")
                        subprocess.run(['tail', '-n', str(lines), str(log_file)])
            return
        
        proc = matched[0]
        log_file = self.logs_dir / f"{proc['process_id']}.log"
        
        if not log_file.exists():
            print(f"\n⚠️ 日志文件不存在: {log_file}")
            return
        
        print(f"\n📋 {ticker} 日志 ({proc['strategy']}, {proc['mode']}):")
        print(f"   文件: {log_file}")
        print("="*80)
        
        if follow:
            # 实时跟踪日志
            try:
                subprocess.run(['tail', '-f', str(log_file)])
            except KeyboardInterrupt:
                print("\n已停止跟踪日志")
        else:
            # 显示最后 N 行
            subprocess.run(['tail', '-n', str(lines), str(log_file)])
    
    def open_chart(self, ticker: str):
        """在浏览器中打开图表"""
        processes = self.get_running_processes()
        matched = [p for p in processes if p['ticker'] == ticker]
        
        if not matched:
            print(f"\n⚠️ 没有找到 {ticker} 的运行进程")
            
            # 查找历史图表
            chart_files = list(self.charts_dir.glob(f"{ticker}_*.html"))
            if chart_files:
                print(f"\n💡 找到历史图表文件:")
                for chart_file in chart_files:
                    print(f"   {chart_file}")
            return
        
        proc = matched[0]
        chart_file = self.charts_dir / f"{proc['process_id']}.html"
        
        if not chart_file.exists():
            print(f"\n⚠️ 图表文件不存在: {chart_file}")
            return
        
        print(f"📊 打开 {ticker} 图表...")
        
        import webbrowser
        webbrowser.open(f'file://{chart_file.absolute()}')


def main():
    manager = MultiProcessManager()
    
    parser = argparse.ArgumentParser(
        description='多进程交易管理器',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # start 命令
    start_parser = subparsers.add_parser('start', help='启动交易进程')
    start_parser.add_argument('--tickers', nargs='+', required=True,
                             help='股票代码列表 (例: TSLA AAPL NVDA)')
    start_parser.add_argument('--strategy', default='moderate',
                             help='策略名称 (默认: moderate)')
    start_parser.add_argument('--mode', default='paper',
                             choices=['paper', 'live', 'simulation'],
                             help='交易模式 (默认: paper)')
    start_parser.add_argument('--interval', type=int, default=60,
                             help='运行间隔（秒，默认: 60）')
    start_parser.add_argument('--no-chart', action='store_true',
                             help='禁用图表')
    
    # status 命令
    subparsers.add_parser('status', help='查看运行状态')
    
    # stop 命令
    stop_parser = subparsers.add_parser('stop', help='停止进程')
    stop_parser.add_argument('--ticker', help='股票代码')
    stop_parser.add_argument('--strategy', help='策略名称')
    stop_parser.add_argument('--mode', help='交易模式')
    stop_parser.add_argument('--all', action='store_true', help='停止所有进程')
    
    # logs 命令
    logs_parser = subparsers.add_parser('logs', help='查看日志')
    logs_parser.add_argument('--ticker', required=True, help='股票代码')
    logs_parser.add_argument('--follow', '-f', action='store_true',
                            help='实时跟踪日志')
    logs_parser.add_argument('--lines', '-n', type=int, default=50,
                            help='显示行数 (默认: 50)')
    
    # chart 命令
    chart_parser = subparsers.add_parser('chart', help='打开图表')
    chart_parser.add_argument('--ticker', required=True, help='股票代码')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 执行命令
    if args.command == 'start':
        print(f"\n🚀 启动 {len(args.tickers)} 个交易进程...")
        print(f"   策略: {args.strategy}")
        print(f"   模式: {args.mode}")
        print(f"   间隔: {args.interval} 秒")
        print()
        
        success_count = 0
        for ticker in args.tickers:
            if manager.start_process(
                ticker=ticker.upper(),
                strategy=args.strategy,
                mode=args.mode,
                interval=args.interval,
                no_chart=args.no_chart
            ):
                success_count += 1
            time.sleep(1)  # 错开启动时间
        
        print(f"\n✅ 成功启动 {success_count}/{len(args.tickers)} 个进程")
        
        if success_count > 0:
            print(f"\n💡 提示:")
            print(f"   查看状态: python {sys.argv[0]} status")
            print(f"   查看日志: python {sys.argv[0]} logs --ticker <TICKER> --follow")
    
    elif args.command == 'status':
        manager.show_status()
    
    elif args.command == 'stop':
        manager.stop_process(
            ticker=args.ticker.upper() if args.ticker else None,
            strategy=args.strategy,
            mode=args.mode,
            stop_all=args.all
        )
    
    elif args.command == 'logs':
        manager.show_logs(
            ticker=args.ticker.upper(),
            follow=args.follow,
            lines=args.lines
        )
    
    elif args.command == 'chart':
        manager.open_chart(ticker=args.ticker.upper())


if __name__ == '__main__':
    main()