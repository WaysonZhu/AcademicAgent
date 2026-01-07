import redis
import json
from typing import List, Dict
from config.settings import settings
from utils.logger import setup_logger
from utils.global_state import global_stats

logger = setup_logger("storage_agent")


class StorageAgent:
    def __init__(self):
        # 初始化 Redis 连接
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            decode_responses=True
        )

    def store_paper_data(self, papers: List[Dict]):
        """
        将论文数据存入 Redis。
        """
        if not papers:
            return

        pipeline = self.r.pipeline()
        for paper in papers:
            if not paper: continue
            paper_id = paper.get("paperId")
            if paper_id:
                # 序列化为 JSON 字符串
                pipeline.set(paper_id, json.dumps(paper))

        try:
            pipeline.execute()
            logger.info(f"Stored {len(papers)} papers into Redis.")
        except Exception as e:
            logger.error(f"Redis pipeline error: {e}")

    def process_seed_papers(self, papers: List[Dict]):
        """
        处理 Step 3: Initial Storage & Counting
        """
        self.store_paper_data(papers)

        if not papers: return

        for paper in papers:
            if not paper: continue
            pid = paper.get("paperId")
            if pid:
                global_stats.set_initial_count(pid)
        logger.info(f"Processed seed papers stats. Global map size: {len(global_stats.stats)}")

    def process_graph_expansion(self, detailed_papers: List[Dict]):
        """
        处理 Step 5: Recursive Counting & Update
        """
        # 1. 顶层防御：防止传入 None
        if not detailed_papers:
            logger.warning("process_graph_expansion received empty data.")
            return

        count_updates = 0

        for paper in detailed_papers:
            if not paper: continue

            # 使用 'or []' 强制将 None 转换为空列表
            refs = paper.get("references") or []

            for ref in refs:
                # 防御列表内部可能存在的空对象
                if not ref: continue
                ref_id = ref.get("paperId")
                if ref_id:
                    global_stats.increment_count(ref_id)
                    count_updates += 1

            # 同理，处理 citations 为 null 的情况
            cites = paper.get("citations") or []

            for cite in cites:
                if not cite: continue
                cite_id = cite.get("paperId")
                if cite_id:
                    global_stats.increment_count(cite_id)
                    count_updates += 1

        logger.info(f"Graph expansion complete. Updated counts for {count_updates} related nodes.")