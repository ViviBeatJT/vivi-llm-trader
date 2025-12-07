import os  # 用于文件操作
import json

CACHE_FILE = 'gemini_cache.json'


def load_cache():
    """从本地文件加载 Gemini 响应缓存。"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # 文件损坏时返回空字典
                return {}
    return {}


def save_cache(cache_data):
    """将 Gemini 响应缓存保存到本地文件。"""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=4, ensure_ascii=False)