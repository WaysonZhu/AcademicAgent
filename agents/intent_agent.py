from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings
from utils.logger import setup_logger
from config import prompts

logger = setup_logger("intent_agent")


class IntentAgent:
    def __init__(self):
        # 初始化 Qwen-Max 模型
        self.llm = ChatTongyi(
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
            model_name=settings.MODEL_NAME,
            temperature=0.1  # 低温度以保证输出的确定性
        )

        # 定义 Prompt 模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.INTENT_AGENT_SYSTEM_PROMPT),
            ("user", "{query}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def optimize_query(self, user_query: str) -> str:
        """
        执行意图识别
        """
        try:
            logger.info(f"Optimizing query: {user_query}")
            optimized_query = self.chain.invoke({"query": user_query})
            logger.info(f"Optimized query result: {optimized_query}")
            # 清理可能存在的引号（视模型输出情况而定，这里做简单的防御性清理）
            return optimized_query.strip().strip('"')
        except Exception as e:
            logger.error(f"Error in IntentAgent: {e}")
            # 如果LLM失败，降级为返回原始查询
            return user_query