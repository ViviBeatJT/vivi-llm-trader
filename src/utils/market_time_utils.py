# src/utils/market_time_utils.py

"""
市场时间工具模块

提供市场交易时间检查、收盘管理等功能
"""

from datetime import datetime, time as dt_time
import pytz


# 美股交易时间 (Eastern Time)
US_EASTERN = pytz.timezone('America/New_York')
MARKET_OPEN_TIME = dt_time(9, 30)   # 9:30 AM ET
MARKET_CLOSE_TIME = dt_time(16, 0)  # 4:00 PM ET

# 收盘时间控制
DEFAULT_LAST_ENTRY_TIME = dt_time(15, 50)   # 最后开仓时间
DEFAULT_FORCE_CLOSE_TIME = dt_time(15, 55)  # 强制平仓时间


def get_current_et_time() -> datetime:
    """获取当前东部时间"""
    return datetime.now(US_EASTERN)


def is_market_open(check_time: datetime = None) -> bool:
    """
    检查市场是否开盘
    
    Args:
        check_time: 要检查的时间（东部时间），默认为当前时间
        
    Returns:
        bool: 市场是否开盘
    """
    if check_time is None:
        check_time = get_current_et_time()
    
    current_time = check_time.time()
    weekday = check_time.weekday()
    
    # 周末休市
    if weekday >= 5:
        return False
    
    # 检查交易时间
    return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME


def is_in_no_entry_window(check_time: datetime = None, 
                          last_entry_time: dt_time = DEFAULT_LAST_ENTRY_TIME) -> bool:
    """
    检查是否在禁止开仓窗口（默认15:50-16:00）
    
    Args:
        check_time: 要检查的时间（东部时间），默认为当前时间
        last_entry_time: 最后开仓时间，默认15:50
        
    Returns:
        bool: 是否在禁止开仓窗口
    """
    if check_time is None:
        check_time = get_current_et_time()
    
    return check_time.time() >= last_entry_time


def is_force_close_time(check_time: datetime = None,
                       force_close_time: dt_time = DEFAULT_FORCE_CLOSE_TIME) -> bool:
    """
    检查是否到达强制平仓时间（默认15:55+）
    
    Args:
        check_time: 要检查的时间（东部时间），默认为当前时间
        force_close_time: 强制平仓时间，默认15:55
        
    Returns:
        bool: 是否到达强制平仓时间
    """
    if check_time is None:
        check_time = get_current_et_time()
    
    return check_time.time() >= force_close_time


def should_force_close_position(current_position: float,
                                check_time: datetime = None,
                                force_close_time: dt_time = DEFAULT_FORCE_CLOSE_TIME) -> bool:
    """
    判断是否应该强制平仓
    
    Args:
        current_position: 当前持仓（非0表示有持仓）
        check_time: 要检查的时间（东部时间），默认为当前时间
        force_close_time: 强制平仓时间，默认15:55
        
    Returns:
        bool: 是否应该强制平仓
    """
    if current_position == 0:
        return False
    
    return is_force_close_time(check_time, force_close_time)


def get_close_signal_for_position(current_position: float) -> str:
    """
    根据持仓类型返回对应的平仓信号
    
    Args:
        current_position: 当前持仓（正数=多仓，负数=空仓）
        
    Returns:
        str: 'SELL' (平多) 或 'COVER' (平空) 或 'HOLD' (无持仓)
    """
    if current_position > 0:
        return 'SELL'
    elif current_position < 0:
        return 'COVER'
    else:
        return 'HOLD'


def format_time_et(dt: datetime) -> str:
    """
    格式化东部时间显示
    
    Args:
        dt: datetime对象
        
    Returns:
        str: 格式化的时间字符串
    """
    if dt.tzinfo is None:
        dt = US_EASTERN.localize(dt)
    else:
        dt = dt.astimezone(US_EASTERN)
    
    return dt.strftime('%H:%M:%S ET')