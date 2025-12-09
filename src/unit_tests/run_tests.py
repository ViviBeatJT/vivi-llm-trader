#!/usr/bin/env python3
# src/unit_tests/run_tests.py
# 
# 运行所有单元测试的脚本
# Usage: python -m src.unit_tests.run_tests [options]
#
# Options:
#   -v, --verbose     详细输出
#   -q, --quiet       静默模式
#   -f, --failfast    遇到第一个失败立即停止
#   --pattern PATTERN 测试文件匹配模式 (默认: test_*.py)
#   --module MODULE   只运行指定模块的测试

import unittest
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path


# ==================== 配置 ====================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 测试目录
TEST_DIR = Path(__file__).parent

# 测试文件模式
DEFAULT_PATTERN = "test_*.py"

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(PROJECT_ROOT))


# ==================== 颜色输出 ====================

class Colors:
    """终端颜色代码"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @classmethod
    def disable(cls):
        """禁用颜色（Windows 或非 TTY）"""
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.RED = ''
        cls.BOLD = ''
        cls.UNDERLINE = ''
        cls.END = ''


# Windows 或非 TTY 环境禁用颜色
if not sys.stdout.isatty() or os.name == 'nt':
    Colors.disable()


# ==================== 自定义测试结果 ====================

class ColoredTestResult(unittest.TextTestResult):
    """带颜色的测试结果输出"""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.successes = []
        self._verbosity = verbosity  # 保存 verbosity
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)
        if self._verbosity > 1:
            self.stream.write(f"{Colors.GREEN}✓{Colors.END} ")
            self.stream.flush()
    
    def addError(self, test, err):
        super().addError(test, err)
        if self._verbosity > 1:
            self.stream.write(f"{Colors.RED}E{Colors.END} ")
            self.stream.flush()
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self._verbosity > 1:
            self.stream.write(f"{Colors.RED}F{Colors.END} ")
            self.stream.flush()
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self._verbosity > 1:
            self.stream.write(f"{Colors.YELLOW}S{Colors.END} ")
            self.stream.flush()


class ColoredTestRunner(unittest.TextTestRunner):
    """带颜色的测试运行器"""
    resultclass = ColoredTestResult


# ==================== 测试发现和运行 ====================

def discover_tests(test_dir: Path, pattern: str = DEFAULT_PATTERN) -> unittest.TestSuite:
    """
    发现测试目录下的所有测试
    
    Args:
        test_dir: 测试目录
        pattern: 测试文件匹配模式
    
    Returns:
        TestSuite 对象
    """
    loader = unittest.TestLoader()
    
    # 发现测试
    suite = loader.discover(
        start_dir=str(test_dir),
        pattern=pattern,
        top_level_dir=str(PROJECT_ROOT)
    )
    
    return suite


def load_specific_modules(test_dir: Path, modules: list) -> unittest.TestSuite:
    """
    加载指定的测试模块
    
    Args:
        test_dir: 测试目录
        modules: 模块名列表
    
    Returns:
        TestSuite 对象
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for module_name in modules:
        try:
            # 构建完整模块路径
            full_module_name = f"src.unit_tests.{module_name}"
            module = __import__(full_module_name, fromlist=[module_name])
            tests = loader.loadTestsFromModule(module)
            suite.addTests(tests)
            print(f"  {Colors.GREEN}✓{Colors.END} Loaded: {module_name}")
        except ImportError as e:
            print(f"  {Colors.YELLOW}⚠{Colors.END} Skipped: {module_name} ({e})")
        except Exception as e:
            print(f"  {Colors.RED}✗{Colors.END} Error loading {module_name}: {e}")
    
    return suite


def count_tests(suite: unittest.TestSuite) -> int:
    """递归计算测试数量"""
    count = 0
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            count += count_tests(test)
        else:
            count += 1
    return count


def run_tests(
    test_dir: Path = TEST_DIR,
    pattern: str = DEFAULT_PATTERN,
    verbosity: int = 2,
    failfast: bool = False,
    modules: list = None
) -> unittest.TestResult:
    """
    运行测试
    
    Args:
        test_dir: 测试目录
        pattern: 测试文件匹配模式
        verbosity: 详细程度 (0=quiet, 1=normal, 2=verbose)
        failfast: 遇到失败立即停止
        modules: 指定的模块列表（None=运行所有）
    
    Returns:
        TestResult 对象
    """
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}  Trading System Unit Tests{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Test Directory: {test_dir}")
    print(f"  Pattern: {pattern}")
    print(f"{'='*60}\n")
    
    # 加载测试
    print(f"{Colors.CYAN}Loading tests...{Colors.END}")
    if modules:
        suite = load_specific_modules(test_dir, modules)
    else:
        suite = discover_tests(test_dir, pattern)
    
    test_count = count_tests(suite)
    print(f"\n{Colors.CYAN}Found {test_count} tests{Colors.END}\n")
    print(f"{'='*60}\n")
    
    if test_count == 0:
        print(f"{Colors.YELLOW}No tests found!{Colors.END}")
        print(f"Make sure test files match pattern: {pattern}")
        return None
    
    # 运行测试
    runner = ColoredTestRunner(
        verbosity=verbosity,
        failfast=failfast,
        resultclass=ColoredTestResult
    )
    
    result = runner.run(suite)
    
    # 打印摘要
    print_summary(result, test_count)
    
    return result


def print_summary(result: unittest.TestResult, total: int):
    """打印测试摘要"""
    print(f"\n{'='*60}")
    print(f"{Colors.BOLD}  Test Summary{Colors.END}")
    print(f"{'='*60}")
    
    passed = len(result.successes) if hasattr(result, 'successes') else total - len(result.failures) - len(result.errors) - len(result.skipped)
    failed = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    
    # 状态行
    if failed == 0 and errors == 0:
        status = f"{Colors.GREEN}✓ ALL TESTS PASSED{Colors.END}"
    else:
        status = f"{Colors.RED}✗ SOME TESTS FAILED{Colors.END}"
    
    print(f"\n  {status}\n")
    
    # 详细统计
    print(f"  {Colors.GREEN}Passed:  {passed:4d}{Colors.END}")
    print(f"  {Colors.RED}Failed:  {failed:4d}{Colors.END}")
    print(f"  {Colors.RED}Errors:  {errors:4d}{Colors.END}")
    print(f"  {Colors.YELLOW}Skipped: {skipped:4d}{Colors.END}")
    print(f"  {Colors.BOLD}Total:   {total:4d}{Colors.END}")
    
    # 成功率
    if total > 0:
        success_rate = (passed / total) * 100
        color = Colors.GREEN if success_rate == 100 else (Colors.YELLOW if success_rate >= 80 else Colors.RED)
        print(f"\n  {color}Success Rate: {success_rate:.1f}%{Colors.END}")
    
    print(f"\n{'='*60}")
    
    # 显示失败详情
    if result.failures:
        print(f"\n{Colors.RED}{Colors.BOLD}Failed Tests:{Colors.END}")
        for test, traceback in result.failures:
            print(f"\n  {Colors.RED}✗ {test}{Colors.END}")
            # 只显示最后几行
            lines = traceback.strip().split('\n')
            for line in lines[-5:]:
                print(f"    {line}")
    
    # 显示错误详情
    if result.errors:
        print(f"\n{Colors.RED}{Colors.BOLD}Errors:{Colors.END}")
        for test, traceback in result.errors:
            print(f"\n  {Colors.RED}✗ {test}{Colors.END}")
            lines = traceback.strip().split('\n')
            for line in lines[-5:]:
                print(f"    {line}")


# ==================== 命令行接口 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="运行 Trading System 单元测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.unit_tests.run_tests              # 运行所有测试
  python -m src.unit_tests.run_tests -v           # 详细模式
  python -m src.unit_tests.run_tests -q           # 静默模式
  python -m src.unit_tests.run_tests -f           # 遇到失败立即停止
  python -m src.unit_tests.run_tests -m test_position_manager  # 只运行指定模块
  python -m src.unit_tests.run_tests --pattern "test_*strategy*.py"  # 自定义模式
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='静默模式'
    )
    
    parser.add_argument(
        '-f', '--failfast',
        action='store_true',
        help='遇到第一个失败立即停止'
    )
    
    parser.add_argument(
        '--pattern',
        default=DEFAULT_PATTERN,
        help=f'测试文件匹配模式 (默认: {DEFAULT_PATTERN})'
    )
    
    parser.add_argument(
        '--module', '-m',
        action='append',
        dest='modules',
        help='只运行指定模块的测试 (可多次使用)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用的测试模块'
    )
    
    return parser.parse_args()


def list_modules(test_dir: Path, pattern: str):
    """列出所有测试模块"""
    print(f"\n{Colors.BOLD}Available Test Modules:{Colors.END}\n")
    
    test_files = sorted(test_dir.glob(pattern))
    
    if not test_files:
        print(f"  {Colors.YELLOW}No test files found matching '{pattern}'{Colors.END}")
        return
    
    for f in test_files:
        module_name = f.stem
        print(f"  • {module_name}")
    
    print(f"\n{Colors.CYAN}Total: {len(test_files)} modules{Colors.END}")
    print(f"\nUsage: python -m src.unit_tests.run_tests --module <module_name>")


# ==================== 主函数 ====================

def main():
    """主函数"""
    args = parse_args()
    
    # 列出模块
    if args.list:
        list_modules(TEST_DIR, args.pattern)
        return 0
    
    # 确定详细程度
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    # 运行测试
    result = run_tests(
        test_dir=TEST_DIR,
        pattern=args.pattern,
        verbosity=verbosity,
        failfast=args.failfast,
        modules=args.modules
    )
    
    # 返回退出码
    if result is None:
        return 1
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())