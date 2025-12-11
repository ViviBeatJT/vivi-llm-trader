# src/utils/log_viewer.py

"""
批量回测日志查看器

功能：
1. 列出所有日志文件
2. 查看特定日期的日志
3. 搜索包含特定内容的日志
4. 查看盈利/亏损最多的日志
5. 批量查看日志汇总

用法：
    # 列出所有日志
    python log_viewer.py --log-dir bulk_backtest_results/logs --list
    
    # 查看特定日期
    python log_viewer.py --log-dir bulk_backtest_results/logs --date 2024-12-05
    
    # 查看最赚钱的5天
    python log_viewer.py --log-dir bulk_backtest_results/logs --top-profit 5
    
    # 查看最亏损的5天
    python log_viewer.py --log-dir bulk_backtest_results/logs --top-loss 5
    
    # 搜索包含"止损"的日志
    python log_viewer.py --log-dir bulk_backtest_results/logs --search "止损"
"""

import argparse
from pathlib import Path
import re
from typing import List, Dict
import pandas as pd


class LogViewer:
    """批量回测日志查看器"""
    
    def __init__(self, log_dir: str):
        """
        初始化日志查看器
        
        Args:
            log_dir: 日志目录路径
        """
        self.log_dir = Path(log_dir)
        
        if not self.log_dir.exists():
            raise ValueError(f"日志目录不存在: {log_dir}")
        
        self.log_files = list(self.log_dir.glob("*.log"))
        
        if not self.log_files:
            print(f"⚠️ 目录中没有日志文件: {log_dir}")
    
    def list_logs(self):
        """列出所有日志文件"""
        print(f"\n{'='*80}")
        print(f"📋 日志文件列表 (共 {len(self.log_files)} 个)")
        print(f"{'='*80}\n")
        
        if not self.log_files:
            return
        
        # 按日期排序
        sorted_files = sorted(self.log_files, key=lambda x: x.stem)
        
        for i, log_file in enumerate(sorted_files, 1):
            file_size = log_file.stat().st_size / 1024  # KB
            print(f"{i:3d}. {log_file.name:40s} ({file_size:6.1f} KB)")
    
    def view_log(self, date: str = None, strategy: str = None, filename: str = None):
        """
        查看特定日志
        
        Args:
            date: 日期 (YYYY-MM-DD)
            strategy: 策略名称
            filename: 完整文件名
        """
        if filename:
            log_file = self.log_dir / filename
        elif date and strategy:
            log_file = self.log_dir / f"{date}_{strategy}.log"
        elif date:
            # 查找该日期的所有日志
            matching_files = [f for f in self.log_files if f.stem.startswith(date)]
            
            if not matching_files:
                print(f"❌ 未找到日期 {date} 的日志")
                return
            
            if len(matching_files) > 1:
                print(f"\n找到 {len(matching_files)} 个匹配的日志文件:")
                for i, f in enumerate(matching_files, 1):
                    print(f"{i}. {f.name}")
                
                choice = input("\n请选择要查看的文件 (输入数字): ")
                try:
                    log_file = matching_files[int(choice) - 1]
                except (ValueError, IndexError):
                    print("❌ 无效选择")
                    return
            else:
                log_file = matching_files[0]
        else:
            print("❌ 请指定 date, filename 或 date+strategy")
            return
        
        if not log_file.exists():
            print(f"❌ 日志文件不存在: {log_file}")
            return
        
        print(f"\n{'='*80}")
        print(f"📄 查看日志: {log_file.name}")
        print(f"{'='*80}\n")
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
    
    def extract_pnl_from_log(self, log_file: Path) -> float:
        """从日志文件中提取盈亏"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 查找 "盈亏: $XXX.XX (±X.XX%)" 格式
                match = re.search(r'盈亏: \$([+-]?\d+\.\d+)', content)
                if match:
                    return float(match.group(1))
        except Exception as e:
            pass
        
        return 0.0
    
    def get_top_profit_logs(self, n: int = 5):
        """获取盈利最多的日志"""
        print(f"\n{'='*80}")
        print(f"🏆 盈利最多的 {n} 天")
        print(f"{'='*80}\n")
        
        log_pnl = []
        for log_file in self.log_files:
            pnl = self.extract_pnl_from_log(log_file)
            log_pnl.append((log_file, pnl))
        
        # 排序
        sorted_logs = sorted(log_pnl, key=lambda x: x[1], reverse=True)[:n]
        
        for i, (log_file, pnl) in enumerate(sorted_logs, 1):
            print(f"{i}. {log_file.name:40s} | PnL: ${pnl:+8.2f}")
    
    def get_top_loss_logs(self, n: int = 5):
        """获取亏损最多的日志"""
        print(f"\n{'='*80}")
        print(f"📉 亏损最多的 {n} 天")
        print(f"{'='*80}\n")
        
        log_pnl = []
        for log_file in self.log_files:
            pnl = self.extract_pnl_from_log(log_file)
            log_pnl.append((log_file, pnl))
        
        # 排序
        sorted_logs = sorted(log_pnl, key=lambda x: x[1])[:n]
        
        for i, (log_file, pnl) in enumerate(sorted_logs, 1):
            print(f"{i}. {log_file.name:40s} | PnL: ${pnl:+8.2f}")
    
    def search_logs(self, keyword: str):
        """搜索包含特定关键词的日志"""
        print(f"\n{'='*80}")
        print(f"🔍 搜索关键词: '{keyword}'")
        print(f"{'='*80}\n")
        
        found_count = 0
        
        for log_file in sorted(self.log_files, key=lambda x: x.stem):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    if keyword.lower() in content.lower():
                        found_count += 1
                        
                        # 找到包含关键词的行
                        lines = content.split('\n')
                        matching_lines = [line for line in lines if keyword.lower() in line.lower()]
                        
                        print(f"\n📄 {log_file.name}")
                        print(f"-"*80)
                        for line in matching_lines[:5]:  # 只显示前5个匹配
                            print(f"   {line.strip()}")
                        
                        if len(matching_lines) > 5:
                            print(f"   ... 还有 {len(matching_lines) - 5} 个匹配")
            
            except Exception as e:
                continue
        
        if found_count == 0:
            print(f"❌ 未找到包含 '{keyword}' 的日志")
        else:
            print(f"\n✅ 找到 {found_count} 个包含 '{keyword}' 的日志文件")
    
    def generate_summary(self):
        """生成日志汇总统计"""
        print(f"\n{'='*80}")
        print(f"📊 日志汇总统计")
        print(f"{'='*80}\n")
        
        if not self.log_files:
            print("❌ 没有日志文件")
            return
        
        total_files = len(self.log_files)
        
        # 提取所有盈亏
        pnl_list = []
        dates = set()
        strategies = set()
        
        for log_file in self.log_files:
            pnl = self.extract_pnl_from_log(log_file)
            pnl_list.append(pnl)
            
            # 解析文件名: YYYY-MM-DD_strategy.log
            parts = log_file.stem.split('_')
            if len(parts) >= 2:
                dates.add(parts[0])
                strategies.add('_'.join(parts[1:]))
        
        pnl_array = pd.Series(pnl_list)
        
        print(f"日志文件总数: {total_files}")
        print(f"交易日数: {len(dates)}")
        print(f"策略数: {len(strategies)}")
        print(f"策略列表: {', '.join(sorted(strategies))}")
        
        print(f"\n盈亏统计:")
        print(f"  总盈亏: ${pnl_array.sum():,.2f}")
        print(f"  平均盈亏: ${pnl_array.mean():.2f}")
        print(f"  中位数: ${pnl_array.median():.2f}")
        print(f"  标准差: ${pnl_array.std():.2f}")
        print(f"  最大盈利: ${pnl_array.max():.2f}")
        print(f"  最大亏损: ${pnl_array.min():.2f}")
        
        print(f"\n盈亏分布:")
        profitable = (pnl_array > 0).sum()
        breakeven = (pnl_array == 0).sum()
        losing = (pnl_array < 0).sum()
        
        print(f"  盈利天数: {profitable} ({profitable/total_files*100:.1f}%)")
        print(f"  持平天数: {breakeven} ({breakeven/total_files*100:.1f}%)")
        print(f"  亏损天数: {losing} ({losing/total_files*100:.1f}%)")
    
    def tail_logs(self, n: int = 10):
        """查看最新的n个日志文件"""
        print(f"\n{'='*80}")
        print(f"📋 最新 {n} 个日志文件")
        print(f"{'='*80}\n")
        
        # 按修改时间排序
        sorted_files = sorted(self.log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:n]
        
        for i, log_file in enumerate(sorted_files, 1):
            pnl = self.extract_pnl_from_log(log_file)
            print(f"{i:2d}. {log_file.name:40s} | PnL: ${pnl:+8.2f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量回测日志查看器')
    
    parser.add_argument('--log-dir', type=str, required=True,
                       help='日志目录路径')
    
    parser.add_argument('--list', action='store_true',
                       help='列出所有日志文件')
    
    parser.add_argument('--date', type=str,
                       help='查看特定日期的日志 (YYYY-MM-DD)')
    
    parser.add_argument('--strategy', type=str,
                       help='策略名称 (与 --date 一起使用)')
    
    parser.add_argument('--filename', type=str,
                       help='查看特定日志文件')
    
    parser.add_argument('--top-profit', type=int,
                       help='显示盈利最多的N天')
    
    parser.add_argument('--top-loss', type=int,
                       help='显示亏损最多的N天')
    
    parser.add_argument('--search', type=str,
                       help='搜索包含关键词的日志')
    
    parser.add_argument('--summary', action='store_true',
                       help='生成日志汇总统计')
    
    parser.add_argument('--tail', type=int,
                       help='查看最新的N个日志')
    
    args = parser.parse_args()
    
    # 创建查看器
    try:
        viewer = LogViewer(args.log_dir)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # 执行命令
    if args.list:
        viewer.list_logs()
    
    elif args.date or args.filename:
        viewer.view_log(date=args.date, strategy=args.strategy, filename=args.filename)
    
    elif args.top_profit:
        viewer.get_top_profit_logs(args.top_profit)
    
    elif args.top_loss:
        viewer.get_top_loss_logs(args.top_loss)
    
    elif args.search:
        viewer.search_logs(args.search)
    
    elif args.summary:
        viewer.generate_summary()
    
    elif args.tail:
        viewer.tail_logs(args.tail)
    
    else:
        # 默认显示汇总
        viewer.generate_summary()
        print()
        viewer.list_logs()


if __name__ == '__main__':
    main()