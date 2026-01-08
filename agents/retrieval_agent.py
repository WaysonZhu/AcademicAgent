from typing import List, Dict
from tools.semantic_tools import tool_search_by_keyword, tool_search_batch_details, tool_search_by_title
from utils.logger import setup_logger

logger = setup_logger("retrieval_agent")


class RetrievalAgent:
    """
    论文检索 Agent，负责调用原子工具。
    修正说明：LangChain Tool 必须使用 .invoke(input_dict) 进行调用
    """

    # 按标题搜索种子
    def search_seed_by_title(self, title: str) -> List[Dict]:
        logger.info(f"RetrievalAgent: Searching seed by title '{title}'")
        # tool_search_by_title返回的是单篇Dict，为兼容后续流程，把它包装成List
        paper = tool_search_by_title.invoke({"title": title})
        return [paper] if paper else []

    def initial_search(self, query: str, limit: int = 10) -> List[Dict]:
        """执行 Step 2: Seed Search"""
        logger.info(f"RetrievalAgent: Performing initial search for '{query}'")
        return tool_search_by_keyword.invoke({"query": query, "limit": limit})

    def batch_details_search(self, paper_ids: List[str]) -> List[Dict]:
        """执行 Step 4: Batch Graph Expansion"""
        logger.info(f"RetrievalAgent: Fetching batch details for {len(paper_ids)} papers")
        return tool_search_batch_details.invoke({"paper_ids": paper_ids})

    def fetch_missing_papers(self, paper_ids: List[str]) -> List[Dict]:
        """
        辅助功能：用于在 Step 6 阅读阶段，如果发现 Redis 缺数据，进行补全下载
        """
        return tool_search_batch_details.invoke({"paper_ids": paper_ids})