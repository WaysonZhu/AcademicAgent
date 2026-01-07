from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.storage_agent import StorageAgent
from agents.ranking_agent import RankingAgent
from agents.reporting_agent import ReportingAgent
from utils.global_state import global_stats
from utils.logger import setup_logger

logger = setup_logger("workflow")

class SearchWorkflow:
    def __init__(self):
        self.intent_agent = IntentAgent()
        self.retrieval_agent = RetrievalAgent()
        self.storage_agent = StorageAgent()
        self.ranking_agent = RankingAgent()
        self.reporting_agent = ReportingAgent()

    async def run(self, user_query: str, status_callback=None):
        """
        执行完整搜索流程。
        status_callback: 用于向 UI 发送进度更新的函数 (msg: str)
        """
        async def update_status(msg):
            logger.info(msg)
            if status_callback:
                await status_callback(msg)

        # 0. 重置全局状态
        global_stats.clear()

        # Step 1: User Intent Parsing
        await update_status("Step 1/7: 正在识别用户意图并优化查询...")
        optimized_query = self.intent_agent.optimize_query(user_query)
        await update_status(f"检索关键词优化为: {optimized_query}")

        # Step 2: Initial Retrieval (Seed Search)
        await update_status("Step 2/7: 执行种子论文检索 (Limit=10)...")
        seed_papers = self.retrieval_agent.initial_search(optimized_query, limit=10)
        if not seed_papers:
            return "未找到相关论文，请尝试更换关键词。"

        # Step 3: Initial Storage & Counting
        await update_status("Step 3/7: 正在存储种子论文并初始化图谱...")
        self.storage_agent.process_seed_papers(seed_papers)

        # [DEBUG START] 打印种子论文，用于确认哪些论文被选为了种子、
        log_buffer = ["\n" + "-" * 20 + " [DEBUG] Seed Papers List " + "-" * 20]
        for i, p in enumerate(seed_papers, 1):
            pid = p.get('paperId', 'Unknown')
            title = p.get('title', 'No Title')[:80] + "..."
            log_buffer.append(f"Seed #{i:02d} | ID: {pid} | Title: {title}")
        log_buffer.append("-" * 50 + "\n")
        logger.info("\n".join(log_buffer))
        # [DEBUG END] 打印种子论文

        # Step 4: Batch Graph Expansion
        seed_ids = [p['paperId'] for p in seed_papers if p.get('paperId')]
        await update_status(f"Step 4/7: 正在扩展图谱，批量获取 {len(seed_ids)} 篇论文的详细引文关系...")
        detailed_papers = self.retrieval_agent.batch_details_search(seed_ids)

        # Step 5: Recursive Counting & Update
        await update_status("Step 5/7: 递归计算论文引用频次，挖掘核心理论...")
        self.storage_agent.process_graph_expansion(detailed_papers)

        # [DEBUG START] 打印全局变量数据
        # 使用列表构建日志信息，最后一次性打印，避免刷屏
        log_buffer = ["\n" + "=" * 50, "[DEBUG] Global State 数据监控",
                      f"图谱节点总数 (Total Papers): {len(global_stats.stats)}",
                      "引用频次最高的 Top-20 论文 (Top-20 Frequent Papers):"]

        # 获取Top-20用于展示
        top_debug = global_stats.get_top_k(20)
        for rank, (pid, count) in enumerate(top_debug, 1):
            log_buffer.append(f"  Rank {rank:02d} | Count: {count} | PaperID: {pid}")

        log_buffer.append("=" * 50 + "\n")

        # 一次性输出所有 Debug 信息
        logger.info("\n".join(log_buffer))
        # [DEBUG END] 打印全局变量数据

        # Step 6: Ranking & Reading
        await update_status("Step 6/7: 获取 Top-10 核心论文，进行 AI 深度阅读与评分...")
        ranked_papers = self.ranking_agent.rank_papers()

        # Step 7: Reporting
        await update_status("Step 7/7: 正在生成深度调研报告...")
        report = await self.reporting_agent.generate_report(user_query, ranked_papers)

        return report