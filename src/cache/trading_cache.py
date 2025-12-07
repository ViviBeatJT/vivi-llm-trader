import os  # Used for file operations
import json

CACHE_FILE = 'gemini_cache.json'
class TradingCache:
    """
    用于存储 Gemini 响应的交易缓存类。
    实现加载、保存、检查和添加缓存数据的功能。
    """
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self.data = self._load()

    def _load(self) -> dict:
        """从本地文件加载 Gemini 响应缓存。"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"⚠️ 缓存文件 '{self.cache_file}' 损坏，返回空缓存。")
                    return {}
        return {}

    def save(self):
        """将当前的缓存数据保存到本地文件。"""
        print(f"💾 正在保存缓存到 {self.cache_file}...")
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            # 写入缓存数据，使用 indent=4 格式化，确保非 ASCII 字符正确显示
            json.dump(self.data, f, indent=4, ensure_ascii=False)
        print("✅ 缓存保存成功。")

    def get(self, key: str):
        """
        检查并获取缓存中的键值。
        
        Args:
            key: 缓存键 (通常是哈希值)。
            
        Returns:
            如果命中，返回缓存值；否则返回 None。
        """
        return self.data.get(key)

    def add(self, key: str, value):
        """
        添加新的键值对到缓存。
        
        Args:
            key: 缓存键。
            value: 要缓存的 Gemini 响应结果 (字典)。
        """
        self.data[key] = value

    def __len__(self):
        """返回缓存中的条目数量。"""
        return len(self.data)

    def __str__(self):
        return f"TradingCache(file='{self.cache_file}', size={len(self)})"