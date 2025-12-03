# run_trade.sh

#!/bin/bash

# --- 配置 ---
# 切换到项目根目录（重要：确保相对路径正确）
PROJECT_ROOT="/Users/vivi/vivi-llm-trader"
cd $PROJECT_ROOT

# 日志文件路径
LOG_FILE="./trade_loop_log.txt"

# --- 执行循环 ---
echo "--- 启动 LLM 交易循环 (每分钟执行一次) ---" | tee -a $LOG_FILE
echo "PID: $$" | tee -a $LOG_FILE

# 无限循环
while true; do
    # 记录执行时间
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "" | tee -a $LOG_FILE
    echo "[$TIMESTAMP] --- 正在执行交易逻辑 ---" | tee -a $LOG_FILE

    # 1. 激活虚拟环境 (根据您的实际路径调整)
    source llm-trade/bin/activate

    # 2. 使用 Python 模块执行交易执行器
    # 错误和输出都追加到日志文件中 (tee -a $LOG_FILE)
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m src.executor.alpaca_trade_executor 2>&1 | tee -a $LOG_FILE
    # 3. 退出环境
    deactivate
    
    # 4. 暂停 60 秒（1 分钟）
    sleep 60
done