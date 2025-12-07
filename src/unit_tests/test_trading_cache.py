import unittest
import os
import json
import tempfile
from src.cache.trading_cache import TradingCache

# run with python -m unittest src.unit_tests.test_trading_cache

class TestTradingCache(unittest.TestCase):
    """
    测试 TradingCache 类的核心功能。
    使用临时文件来模拟缓存文件，确保测试的隔离性和非破坏性。
    """

    def setUp(self):
        """
        在每个测试方法运行前设置一个唯一的临时文件路径。
        """
        # 使用 tempfile 创建一个临时文件路径
        self.temp_file_path = os.path.join(tempfile.gettempdir(), f'test_cache_{os.getpid()}.json')
        # 确保文件不存在，避免干扰
        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)

    def tearDown(self):
        """
        在每个测试方法运行后清理临时文件。
        """
        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)

    def test_initialization_no_file(self):
        """测试在缓存文件不存在时，初始化应返回一个空字典。"""
        cache = TradingCache(self.temp_file_path)
        self.assertEqual(cache.data, {})
        self.assertEqual(len(cache), 0)
        
    def test_initialization_with_valid_file(self):
        """测试加载有效的缓存文件。"""
        # 1. 预写入有效数据到临时文件
        initial_data = {
            "key1": {"signal": "BUY", "confidence": 9},
            "key2": {"signal": "HOLD", "confidence": 5}
        }
        with open(self.temp_file_path, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f)
            
        # 2. 初始化 TradingCache
        cache = TradingCache(self.temp_file_path)
        
        # 3. 验证数据是否正确加载
        self.assertEqual(cache.data, initial_data)
        self.assertEqual(len(cache), 2)
        
    def test_initialization_with_corrupted_file(self):
        """测试加载损坏的 (非 JSON 格式的) 缓存文件。"""
        # 1. 预写入损坏数据
        with open(self.temp_file_path, 'w', encoding='utf-8') as f:
            f.write("这不是有效的JSON")
            
        # 2. 初始化 TradingCache
        # 由于损坏，应该返回一个空字典
        cache = TradingCache(self.temp_file_path)
        
        # 3. 验证数据是否为空
        self.assertEqual(cache.data, {})
        self.assertEqual(len(cache), 0)
        
    def test_add_and_get_data(self):
        """测试 add 和 get 方法 (添加和获取数据)。"""
        cache = TradingCache(self.temp_file_path)
        test_key = "test_hash_abc"
        test_value = {"signal": "SELL", "price": 150.5}
        
        # 1. 添加数据
        cache.add(test_key, test_value)
        
        # 2. 验证数据是否已添加
        self.assertEqual(len(cache), 1)
        
        # 3. 获取已添加的数据 (命中)
        retrieved_value = cache.get(test_key)
        self.assertEqual(retrieved_value, test_value)
        
        # 4. 获取不存在的键 (未命中)
        self.assertIsNone(cache.get("non_existent_key"))
        
    def test_save_data(self):
        """测试 save 方法，验证数据是否正确写入文件。"""
        cache = TradingCache(self.temp_file_path)
        
        # 1. 添加一些数据
        cache.add("save_key_1", {"data": 123})
        cache.add("save_key_2", {"data": 456})
        
        # 2. 执行保存
        cache.save()
        
        # 3. 验证文件是否存在
        self.assertTrue(os.path.exists(self.temp_file_path))
        
        # 4. 验证文件内容是否正确
        with open(self.temp_file_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            
        expected_data = {
            "save_key_1": {"data": 123},
            "save_key_2": {"data": 456}
        }
        self.assertEqual(saved_data, expected_data)
        
        # 5. 测试重新加载保存的数据
        new_cache = TradingCache(self.temp_file_path)
        self.assertEqual(new_cache.data, expected_data)
        
    def test_str_and_len(self):
        """测试 __str__ 和 __len__ 方法。"""
        cache = TradingCache(self.temp_file_path)
        self.assertEqual(len(cache), 0)
        
        cache.add("k1", 1)
        self.assertEqual(len(cache), 1)
        
        self.assertIn(self.temp_file_path, str(cache))
        self.assertIn("size=1", str(cache))

if __name__ == '__main__':
    # 为了运行这个单元测试，你需要将 src/cache/trading_cache.py 所在目录添加到 Python 路径中
    # 假设你在项目根目录执行测试
    # python -m unittest src.tests.test_trading_cache
    unittest.main()