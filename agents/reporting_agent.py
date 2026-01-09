from typing import List, Dict, Any
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings
from config import prompts
from utils.logger import setup_logger

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

    def _format_authors(self, authors: List[Any]) -> str:
        """辅助函数：格式化作者列表"""
        if not authors:
            return "Unknown Author"

        author_names = []
        for auth in authors:
            if isinstance(auth, dict):
                author_names.append(auth.get('name', ''))
            else:
                author_names.append(str(auth))

        # 过滤空名
        author_names = [n for n in author_names if n]

        if not author_names:
            return "Unknown Author"

        # 如果作者超过3个，只显示前3个 + et al.
        if len(author_names) > 3:
            return ", ".join(author_names[:3]) + " et al"
        return ", ".join(author_names)

    async def generate_report(self, topic: str, papers: List[Dict]) -> str:
        """
        生成 Markdown 报告
        """
        if not papers:
            return "## 未找到相关论文，无法生成报告。"

        logger.info("Generating final report...")

        # 1. 构建 LLM 输入上下文
        papers_text_list = []
        for i, p in enumerate(papers, 1):
            abstract_text = (p.get('abstract') or "")

            # 提取第一作者用于 [Response_Start] 标记
            authors = p.get('authors', [])
            first_author = "Unknown"
            if authors:
                first_obj = authors[0]
                if isinstance(first_obj, dict):
                    first_author = first_obj.get('name', 'Unknown')
                else:
                    first_author = str(first_obj)

            papers_text_list.append(
                f"Paper {i}:\n"
                f"PaperID: {p.get('paperId')}\n"
                f"URL: {p.get('url')}\n"
                f"FirstAuthor: {first_author}\n"
                f"Title: {p.get('title')}\n"
                f"Year: {p.get('year')}\n"
                f"Citations: {p.get('citationCount')}\n"
                f"Reason for selection: {p.get('ai_reason', 'High relevance')}\n"
                f"Abstract: {abstract_text[:800]}\n"
                "---"
            )

        papers_text = "\n".join(papers_text_list)

        try:
            # 2. 调用 LLM 生成报告主体 (Section 1-4)
            report_body = await self.chain.ainvoke({
                "papers_text": papers_text,
                "topic": topic
            })

            # 3. 拼接参考文献列表
            references_section = ["\n\n## 5. 参考文献 (References)"]

            for p in papers:
                title = p.get('title', 'Unknown Title')
                url = p.get('url', '#')
                year = p.get('year', 'N.A.')
                venue = p.get('venue', 'Unknown Venue')
                authors_str = self._format_authors(p.get('authors', []))

                # Markdown 格式: - [Title](URL). Authors. Year. Venue.
                # 点击标题可跳转
                ref_line = f"- [**{title}**]({url}). {authors_str}. {year}. {venue}."
                references_section.append(ref_line)

            # 参考文献拼接
            final_report = report_body + "\n".join(references_section)

            return final_report

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"生成报告时发生错误: {str(e)}"