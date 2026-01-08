from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.storage_agent import StorageAgent
from agents.ranking_agent import RankingAgent
from agents.reporting_agent import ReportingAgent
from utils.global_state import global_stats
from utils.logger import setup_logger

# 初始化工作流日志记录器
logger = setup_logger("workflow")


class SearchWorkflow:
    """
    搜索工作流引擎
    负责编排和调度各个Agent去执行学术调研任务。
    """

    def __init__(self):
        # 初始化各个功能 Agent
        self.intent_agent = IntentAgent()  # 意图识别 Agent
        self.retrieval_agent = RetrievalAgent()  # 论文检索 Agent
        self.storage_agent = StorageAgent()  # 存储与统计 Agent
        self.ranking_agent = RankingAgent()  # 阅读与评分 Agent
        self.reporting_agent = ReportingAgent()  # 总结报告 Agent

    async def run(self, user_query: str, status_callback=None):
        """
        核心调度入口：执行完整的学术搜索工作流。

        Args:
            user_query (str): 用户输入的原始自然语言问题
            status_callback (func, optional): 用于向前端 UI 推送实时进度的异步回调函数

        Returns:
            str: 最终生成的 Markdown 格式调研报告
        """

        # 定义内部辅助函数：用于同时打印日志并推送到前端UI
        async def update_status(msg):
            logger.info(msg)
            if status_callback:
                await status_callback(msg)

        # ------------------------------------------------------------------
        # Step 0: 状态重置
        # ------------------------------------------------------------------
        # 清空上一轮会话的全局论文频次统计数据，确保本次搜索的数据纯净
        global_stats.clear()

        # ------------------------------------------------------------------
        # Step 1: 意图识别与查询优化 (已升级为路由模式)
        # ------------------------------------------------------------------
        await update_status("Step 1/7: 正在识别用户意图并优化查询...")

        # 获取意图识别结果 (字典格式)
        intent_data = self.intent_agent.optimize_query(user_query)

        # 解析意图数据 (兼容性处理：防止旧版返回字符串)
        if isinstance(intent_data, dict):
            search_type = intent_data.get("search_type", "keyword")
            query_content = intent_data.get("query", user_query)
        else:
            search_type = "keyword"
            query_content = intent_data

        await update_status(f"意图识别结果: 类型=[{search_type}], 内容=[{query_content}]")

        # ------------------------------------------------------------------
        # Step 2: 种子论文检索 (动态路由)
        # ------------------------------------------------------------------
        await update_status(f"Step 2/7: 执行核心论文检索 (类型: {search_type})...")

        seed_papers = []

        # 核心路由逻辑
        if search_type == "title":
            # 分支A: 用户给了标题，精确查单篇
            await update_status(f"检测到论文标题，正在进行精确匹配...")
            # 调用RetrievalAgent的标题精确搜索方法
            seed_papers = self.retrieval_agent.search_seed_by_title(query_content)

        else:
            # 分支B: 默认根据关键词进行相关性搜索
            await update_status(f"执行相关性检索...")
            seed_papers = self.retrieval_agent.initial_search(query_content, limit=10)
        # --------------------

        # 若种子检索为空，直接中断流程并反馈
        if not seed_papers:
            return f"未找到相关论文（类型：{search_type}），请检查输入内容是否准确。"

        # ------------------------------------------------------------------
        # Step 3: 种子论文存储与初始化
        # ------------------------------------------------------------------
        # 将种子论文的基础信息存入 Redis，并在全局变量中初始化其频次
        await update_status("Step 3/7: 正在存储核心论文信息...")
        self.storage_agent.process_seed_papers(seed_papers)

        # [DEBUG START] 调试日志：打印种子论文清单
        # 用于确认检索到的初始论文ID和标题是否符合预期
        log_buffer = ["\n" + "-" * 20 + " [DEBUG] Seed Papers List " + "-" * 20]
        for i, p in enumerate(seed_papers, 1):
            pid = p.get('paperId', 'Unknown')
            # 截取标题前80个字符以保持日志整洁
            title = p.get('title', 'No Title')[:80] + "..."
            log_buffer.append(f"Seed #{i:02d} | ID: {pid} | Title: {title}")
        log_buffer.append("-" * 50 + "\n")
        logger.info("\n".join(log_buffer))
        # [DEBUG END]

        # ------------------------------------------------------------------
        # Step 4: 引用扩展与批量详情检索
        # ------------------------------------------------------------------
        # 提取种子论文ID，批量请求 API 获取详细的引用关系 (References) 和被引关系 (Citations)
        seed_ids = [p['paperId'] for p in seed_papers if p.get('paperId')]
        await update_status(f"Step 4/7: 正在扩展引用信息，批量获取 {len(seed_ids)} 篇论文的详细引文关系...")
        detailed_papers = self.retrieval_agent.batch_details_search(seed_ids)

        # ------------------------------------------------------------------
        # Step 5: 递归引用统计与核心挖掘
        # ------------------------------------------------------------------
        # 遍历详细引文关系，计算所有相关节点的出现频次，挖掘潜在的核心论文
        await update_status("Step 5/7: 递归计算论文引用频次，挖掘潜在的核心论文...")
        self.storage_agent.process_graph_expansion(detailed_papers)

        # [DEBUG START] 调试日志：监控全局频次统计状态
        # 批量构建日志信息并一次性输出，避免频繁IO导致控制台刷屏
        log_buffer = ["\n" + "=" * 50, "[DEBUG] Global State 数据监控",
                      f"全局文献总数量 (Total Papers): {len(global_stats.stats)}",
                      "引用频次最高的 Top-20 论文 (Top-20 Frequent Papers):"]

        # 提取频次最高的 Top-20 论文用于分析
        top_debug = global_stats.get_top_k(20)
        for rank, (pid, count) in enumerate(top_debug, 1):
            log_buffer.append(f"  Rank {rank:02d} | Count: {count} | PaperID: {pid}")

        log_buffer.append("=" * 50 + "\n")

        # 执行一次性日志输出
        logger.info("\n".join(log_buffer))
        # [DEBUG END]

        # ------------------------------------------------------------------
        # Step 6: 深度阅读与智能评分
        # ------------------------------------------------------------------
        # 1. 从全局统计中截取 Top-10 高频论文
        # 2. 检查 Redis 缺失数据并自动补全
        # 3. 调用大模型阅读摘要并进行多维度打分
        await update_status("Step 6/7: 获取 Top-10 核心论文，进行 AI 深度阅读与评分...")
        ranked_papers = self.ranking_agent.rank_papers()

        # ------------------------------------------------------------------
        # Step 7: 调研报告生成 (Reporting)
        # ------------------------------------------------------------------
        # 将评分排序后的论文列表交给大模型，生成最终的 Markdown 深度综述报告
        await update_status("Step 7/7: 正在生成深度调研报告...")
        report = await self.reporting_agent.generate_report(user_query, ranked_papers)

        return report