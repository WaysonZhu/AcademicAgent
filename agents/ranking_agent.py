import json
from typing import List, Dict
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config.settings import settings
from utils.logger import setup_logger
from utils.global_state import global_stats
from agents.storage_agent import StorageAgent
from agents.retrieval_agent import RetrievalAgent
from config import prompts

logger = setup_logger("ranking_agent")


class RankingAgent:
    def __init__(self):
        self.llm = ChatTongyi(
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
            model_name=settings.MODEL_NAME,
            temperature=0.3
        )
        self.storage_agent = StorageAgent()
        self.retrieval_agent = RetrievalAgent()

        # 定义输出解析器
        self.parser = JsonOutputParser()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.RANKING_AGENT_SYSTEM_PROMPT),
            ("user", "Candidate Papers JSON:\n{papers_json}")
        ])

        self.chain = self.prompt | self.llm | self.parser

    def _get_paper_details(self, paper_ids: List[str]) -> List[Dict]:
        """
        从 Redis 获取详情，如果缺失则调用 API 补全
        """
        papers = []
        missing_ids = []

        # 1. 尝试从 Redis 读取
        pipeline = self.storage_agent.r.pipeline()
        for pid in paper_ids:
            pipeline.get(pid)
        results = pipeline.execute()

        for pid, data_str in zip(paper_ids, results):
            if data_str:
                papers.append(json.loads(data_str))
            else:
                missing_ids.append(pid)

        # 2. 补全缺失数据 (Step 6 关键点)
        if missing_ids:
            logger.info(f"Missing details for {len(missing_ids)} papers. Fetching from API...")
            fetched_papers = self.retrieval_agent.fetch_missing_papers(missing_ids)
            # 存入 Redis 以便下次使用
            self.storage_agent.store_paper_data(fetched_papers)
            papers.extend(fetched_papers)

        return papers

    def rank_papers(self) -> List[Dict]:
        """
        主逻辑：获取 Top IDs -> 补全数据 -> LLM 排序
        """
        # 1. 获取 Top-10 IDs
        top_items = global_stats.get_top_k(10)  # List[Tuple[id, count]]
        if not top_items:
            logger.warning("No papers found in global stats.")
            return []

        top_ids = [item[0] for item in top_items]
        logger.info(f"Ranking Top-10 papers: {top_ids}")

        # 2. 获取完整信息
        papers_data = self._get_paper_details(top_ids)

        # 3. 构建 LLM 输入 (精简字段以节省 Token)
        llm_input = []
        for p in papers_data:
            # 使用 (p.get(...) or "Default") 确保结果一定是字符串
            abstract_text = (p.get("abstract") or "No abstract available.")

            llm_input.append({
                "paperId": p.get("paperId"),
                "title": p.get("title"),
                "abstract": abstract_text,
                "year": p.get("year"),
                "citationCount": p.get("citationCount", 0)
            })

        # 4. 调用 LLM 进行语义打分
        try:
            response = self.chain.invoke({"papers_json": json.dumps(llm_input)})
            ranking_list = response.get("ranking", [])

            # 5. 根据 LLM 结果重组数据
            # 创建一个 lookup 字典
            paper_map = {p["paperId"]: p for p in papers_data if p.get("paperId")}
            ordered_papers = []

            for rank_item in ranking_list:
                pid = rank_item.get("paperId")
                if pid and pid in paper_map:
                    paper = paper_map[pid]
                    # 注入评分理由供报告使用
                    paper["ai_score"] = rank_item.get("score")
                    paper["ai_reason"] = rank_item.get("reason")
                    ordered_papers.append(paper)

            # 如果 LLM 漏掉了某些文章，把剩下的补在后面
            processed_ids = set(p["paperId"] for p in ordered_papers if p.get("paperId"))
            for p in papers_data:
                pid = p.get("paperId")
                if pid and pid not in processed_ids:
                    ordered_papers.append(p)

            return ordered_papers

        except Exception as e:
            logger.error(f"Error during LLM ranking: {e}")
            # 降级：直接返回按引用数排序的结果
            papers_data.sort(key=lambda x: x.get("citationCount", 0), reverse=True)
            return papers_data