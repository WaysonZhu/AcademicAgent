import threading
from collections import defaultdict
from typing import Dict, List, Tuple

class GlobalPaperStats:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GlobalPaperStats, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        初始化全局数据结构
        Key: paper_id (str)
        Value: count (int)
        """
        self.stats: Dict[str, int] = defaultdict(int)

    def increment_count(self, paper_id: str):
        """如果存在则+1，不存在则初始化为1 (defaultdict自动处理初始化为0，这里直接+1即可)"""
        with self._lock:
            self.stats[paper_id] += 1

    def set_initial_count(self, paper_id: str):
        """用于种子搜索，如果不存在则置为1，如果已存在则+1"""
        with self._lock:
            # 种子论文的初始默认频次给2，以防止因为其它论文出现频次较高而把种子论文的排序给挤下去
            self.stats[paper_id] += 2

    def get_top_k(self, k: int = 10) -> List[Tuple[str, int]]:
        """获取频次最高的 Top-K 论文ID"""
        with self._lock:
            # 倒序排序
            sorted_items = sorted(self.stats.items(), key=lambda item: item[1], reverse=True)
            return sorted_items[:k]

    def clear(self):
        """清空状态（用于新的一轮对话）"""
        with self._lock:
            self.stats.clear()

# 导出单例实例
global_stats = GlobalPaperStats()