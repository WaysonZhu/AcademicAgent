from typing import List, Dict
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings
from utils.logger import setup_logger
from config import prompts

logger = setup_logger("reporting_agent")


class ReportingAgent:
    def __init__(self):
        self.llm = ChatTongyi(
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
            model_name=settings.MODEL_NAME,  # Qwen-Max for high quality writing
            temperature=0.5
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.REPORTING_AGENT_SYSTEM_PROMPT),
            ("user", "Papers Data:\n{papers_text}\n\nSearch Topic: {topic}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    async def generate_report(self, topic: str, papers: List[Dict]) -> str:
        """
        生成 Markdown 报告
        """
        if not papers:
            return "## 未找到相关论文，无法生成报告。"

        logger.info("Generating final report...")

        # 预处理数据为文本格式，方便 LLM 理解
        papers_text_list = []
        for i, p in enumerate(papers, 1):
            # 使用 (p.get(...) or "") 确保结果一定是字符串
            abstract_text = (p.get('abstract') or "")

            papers_text_list.append(
                f"Paper {i}:\n"
                f"Title: {p.get('title')}\n"
                f"Year: {p.get('year')}\n"
                f"Citations: {p.get('citationCount')}\n"
                f"Reason for selection: {p.get('ai_reason', 'High relevance')}\n"
                f"Abstract: {abstract_text}\n" 
                "---"
            )

        papers_text = "\n".join(papers_text_list)

        try:
            report = await self.chain.ainvoke({
                "papers_text": papers_text,
                "topic": topic
            })
            return report
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"生成报告时发生错误: {str(e)}"