# src/notification/email_notifier.py

"""
邮件通知模块 - Email Notification Module

功能：
1. 在买入/卖出时发送邮件警报
2. 支持 Gmail SMTP
3. 支持自定义邮件模板

配置方式：
1. 在 .env 文件中添加：
   EMAIL_SENDER=your_email@gmail.com
   EMAIL_PASSWORD=your_app_password  # Gmail 需要使用应用专用密码
   EMAIL_RECIPIENT=hww9130@gmail.com

2. Gmail 设置应用专用密码：
   - 登录 Google 账户 → 安全性 → 两步验证（需先开启）
   - 应用专用密码 → 生成新密码
   - 使用生成的16位密码作为 EMAIL_PASSWORD

使用方式：
    from src.notification.email_notifier import EmailNotifier
    
    notifier = EmailNotifier()
    notifier.send_trade_alert(
        signal='BUY',
        ticker='TSLA',
        price=350.25,
        quantity=10,
        reason='上升趋势回调 + 锤子线'
    )
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import pytz

from dotenv import load_dotenv

load_dotenv()


class AlertType(Enum):
    """警报类型"""
    BUY = "买入"
    SELL = "卖出"
    STOP_LOSS = "止损"
    TAKE_PROFIT = "止盈"
    MARKET_CLOSE = "收盘平仓"
    ERROR = "错误"
    INFO = "信息"


@dataclass
class TradeAlert:
    """交易警报数据"""
    alert_type: AlertType
    ticker: str
    price: float
    quantity: int = 0
    reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    market_state: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class EmailNotifier:
    """
    邮件通知器
    
    支持 Gmail SMTP 发送交易警报邮件
    """
    
    def __init__(self,
                 sender_email: str = None,
                 sender_password: str = None,
                 recipient_email: str = None,
                 smtp_server: str = "smtp.gmail.com",
                 smtp_port: int = 587,
                 enabled: bool = True):
        """
        初始化邮件通知器
        
        Args:
            sender_email: 发送方邮箱（默认从环境变量读取）
            sender_password: 发送方密码/应用专用密码
            recipient_email: 接收方邮箱
            smtp_server: SMTP 服务器
            smtp_port: SMTP 端口
            enabled: 是否启用通知
        """
        self.sender_email = sender_email or os.getenv('EMAIL_SENDER')
        self.sender_password = sender_password or os.getenv('EMAIL_PASSWORD')
        self.recipient_email = recipient_email or os.getenv('EMAIL_RECIPIENT', 'hww9130@gmail.com')
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.enabled = enabled
        
        # 验证配置
        self._validate_config()
        
        # 时区
        self._et = pytz.timezone('America/New_York')
        
        # 发送历史（防止重复发送）
        self._sent_alerts: List[str] = []
        self._max_history = 100
    
    def _validate_config(self):
        """验证邮件配置"""
        if not self.enabled:
            print("📧 邮件通知: 已禁用")
            return
        
        missing = []
        if not self.sender_email:
            missing.append("EMAIL_SENDER")
        if not self.sender_password:
            missing.append("EMAIL_PASSWORD")
        if not self.recipient_email:
            missing.append("EMAIL_RECIPIENT")
        
        if missing:
            print(f"⚠️ 邮件通知配置不完整，缺少: {', '.join(missing)}")
            print("   请在 .env 文件中配置以下环境变量：")
            print("   EMAIL_SENDER=your_email@gmail.com")
            print("   EMAIL_PASSWORD=your_app_password")
            print("   EMAIL_RECIPIENT=recipient@example.com")
            self.enabled = False
        else:
            print(f"📧 邮件通知: 已启用")
            print(f"   发送方: {self.sender_email}")
            print(f"   接收方: {self.recipient_email}")
    
    def _get_alert_emoji(self, alert_type: AlertType) -> str:
        """获取警报类型对应的 emoji"""
        emoji_map = {
            AlertType.BUY: "🟢",
            AlertType.SELL: "🔴",
            AlertType.STOP_LOSS: "🛑",
            AlertType.TAKE_PROFIT: "🎯",
            AlertType.MARKET_CLOSE: "⏰",
            AlertType.ERROR: "❌",
            AlertType.INFO: "ℹ️",
        }
        return emoji_map.get(alert_type, "📢")
    
    def _format_timestamp(self, dt: datetime) -> str:
        """格式化时间戳为美东时间"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        et_time = dt.astimezone(self._et)
        return et_time.strftime('%Y-%m-%d %H:%M:%S ET')
    
    def _create_html_content(self, alert: TradeAlert) -> str:
        """创建 HTML 格式的邮件内容"""
        emoji = self._get_alert_emoji(alert.alert_type)
        timestamp = self._format_timestamp(alert.timestamp)
        
        # 根据交易类型设置颜色
        if alert.alert_type in [AlertType.BUY]:
            color = "#28a745"  # 绿色
            action = "买入"
        elif alert.alert_type in [AlertType.SELL, AlertType.STOP_LOSS, AlertType.TAKE_PROFIT, AlertType.MARKET_CLOSE]:
            color = "#dc3545"  # 红色
            action = "卖出"
        else:
            color = "#6c757d"  # 灰色
            action = alert.alert_type.value
        
        # 盈亏显示
        pnl_html = ""
        if alert.pnl != 0:
            pnl_color = "#28a745" if alert.pnl > 0 else "#dc3545"
            pnl_sign = "+" if alert.pnl > 0 else ""
            pnl_html = f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>盈亏</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd; color: {pnl_color};">
                    {pnl_sign}${alert.pnl:.2f} ({pnl_sign}{alert.pnl_pct:.2f}%)
                </td>
            </tr>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
                .content {{ background-color: #f9f9f9; padding: 20px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 8px 8px; }}
                .info-table {{ width: 100%; border-collapse: collapse; }}
                .info-table td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin: 0;">{emoji} {action}警报</h1>
                    <p style="margin: 10px 0 0 0; opacity: 0.9;">{alert.ticker}</p>
                </div>
                <div class="content">
                    <table class="info-table">
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd; width: 30%;"><strong>股票</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.ticker}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>操作</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd; color: {color}; font-weight: bold;">{action}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>价格</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">${alert.price:.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>数量</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.quantity} 股</td>
                        </tr>
                        {pnl_html}
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>市场状态</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.market_state or 'N/A'}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>原因</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.reason or 'N/A'}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px;"><strong>时间</strong></td>
                            <td style="padding: 8px;">{timestamp}</td>
                        </tr>
                    </table>
                </div>
                <div class="footer">
                    <p>此邮件由交易系统自动发送</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _create_text_content(self, alert: TradeAlert) -> str:
        """创建纯文本格式的邮件内容"""
        emoji = self._get_alert_emoji(alert.alert_type)
        timestamp = self._format_timestamp(alert.timestamp)
        
        pnl_str = ""
        if alert.pnl != 0:
            pnl_sign = "+" if alert.pnl > 0 else ""
            pnl_str = f"盈亏: {pnl_sign}${alert.pnl:.2f} ({pnl_sign}{alert.pnl_pct:.2f}%)\n"
        
        text = f"""
{emoji} {alert.alert_type.value}警报 - {alert.ticker}

股票: {alert.ticker}
操作: {alert.alert_type.value}
价格: ${alert.price:.2f}
数量: {alert.quantity} 股
{pnl_str}市场状态: {alert.market_state or 'N/A'}
原因: {alert.reason or 'N/A'}
时间: {timestamp}

---
此邮件由交易系统自动发送
        """
        return text.strip()
    
    def _generate_alert_id(self, alert: TradeAlert) -> str:
        """生成警报唯一ID（用于防重复）"""
        return f"{alert.ticker}_{alert.alert_type.value}_{alert.price}_{alert.timestamp.strftime('%Y%m%d%H%M')}"
    
    def send_alert(self, alert: TradeAlert) -> bool:
        """
        发送交易警报邮件
        
        Args:
            alert: TradeAlert 对象
            
        Returns:
            bool: 是否发送成功
        """
        if not self.enabled:
            print(f"📧 邮件通知已禁用，跳过发送: {alert.alert_type.value} {alert.ticker}")
            return False
        
        # 防止重复发送
        alert_id = self._generate_alert_id(alert)
        if alert_id in self._sent_alerts:
            print(f"📧 警报已发送过，跳过: {alert_id}")
            return False
        
        try:
            # 创建邮件
            msg = MIMEMultipart('alternative')
            emoji = self._get_alert_emoji(alert.alert_type)
            msg['Subject'] = f"{emoji} {alert.alert_type.value}警报 - {alert.ticker} @ ${alert.price:.2f}"
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            # 添加纯文本和HTML内容
            text_content = self._create_text_content(alert)
            html_content = self._create_html_content(alert)
            
            msg.attach(MIMEText(text_content, 'plain', 'utf-8'))
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient_email, msg.as_string())
            
            # 记录已发送
            self._sent_alerts.append(alert_id)
            if len(self._sent_alerts) > self._max_history:
                self._sent_alerts = self._sent_alerts[-self._max_history:]
            
            print(f"✅ 邮件发送成功: {alert.alert_type.value} {alert.ticker} @ ${alert.price:.2f}")
            return True
            
        except smtplib.SMTPAuthenticationError:
            print(f"❌ 邮件发送失败: 认证错误，请检查 EMAIL_SENDER 和 EMAIL_PASSWORD")
            print("   Gmail 用户需要使用应用专用密码，而非账户密码")
            return False
        except Exception as e:
            print(f"❌ 邮件发送失败: {e}")
            return False
    
    def send_trade_alert(self,
                         signal: str,
                         ticker: str,
                         price: float,
                         quantity: int = 0,
                         reason: str = "",
                         pnl: float = 0.0,
                         pnl_pct: float = 0.0,
                         market_state: str = "",
                         timestamp: datetime = None) -> bool:
        """
        发送交易警报的便捷方法
        
        Args:
            signal: 交易信号 ('BUY', 'SELL', 等)
            ticker: 股票代码
            price: 价格
            quantity: 数量
            reason: 交易原因
            pnl: 盈亏金额
            pnl_pct: 盈亏百分比
            market_state: 市场状态
            timestamp: 时间戳
            
        Returns:
            bool: 是否发送成功
        """
        # 映射信号到警报类型
        signal_map = {
            'BUY': AlertType.BUY,
            'SELL': AlertType.SELL,
            'STOP_LOSS': AlertType.STOP_LOSS,
            'TAKE_PROFIT': AlertType.TAKE_PROFIT,
            'MARKET_CLOSE': AlertType.MARKET_CLOSE,
        }
        
        alert_type = signal_map.get(signal.upper(), AlertType.INFO)
        
        # 根据原因判断是否是止损/止盈
        if '止损' in reason:
            alert_type = AlertType.STOP_LOSS
        elif '止盈' in reason:
            alert_type = AlertType.TAKE_PROFIT
        elif '收盘' in reason:
            alert_type = AlertType.MARKET_CLOSE
        
        alert = TradeAlert(
            alert_type=alert_type,
            ticker=ticker,
            price=price,
            quantity=quantity,
            reason=reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
            market_state=market_state,
            timestamp=timestamp or datetime.now(timezone.utc)
        )
        
        return self.send_alert(alert)
    
    def send_error_alert(self, ticker: str, error_message: str) -> bool:
        """发送错误警报"""
        alert = TradeAlert(
            alert_type=AlertType.ERROR,
            ticker=ticker,
            price=0.0,
            reason=error_message
        )
        return self.send_alert(alert)
    
    def test_connection(self) -> bool:
        """测试邮件连接"""
        if not self.enabled:
            print("📧 邮件通知已禁用")
            return False
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
            print("✅ 邮件服务器连接成功")
            return True
        except Exception as e:
            print(f"❌ 邮件服务器连接失败: {e}")
            return False


# ==================== 全局通知器实例 ====================

_global_notifier: Optional[EmailNotifier] = None


def get_notifier() -> EmailNotifier:
    """获取全局通知器实例"""
    global _global_notifier
    if _global_notifier is None:
        _global_notifier = EmailNotifier()
    return _global_notifier


def send_trade_alert(signal: str, ticker: str, price: float, **kwargs) -> bool:
    """全局发送交易警报的便捷函数"""
    return get_notifier().send_trade_alert(signal, ticker, price, **kwargs)


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("测试邮件通知模块")
    print("=" * 60)
    
    # 创建通知器
    notifier = EmailNotifier()
    
    # 测试连接
    print("\n--- 测试邮件服务器连接 ---")
    notifier.test_connection()
    
    # 发送测试邮件
    print("\n--- 发送测试警报 ---")
    notifier.send_trade_alert(
        signal='BUY',
        ticker='TSLA',
        price=350.25,
        quantity=10,
        reason='测试 - 上升趋势回调 + 锤子线形态',
        market_state='UPTREND'
    )