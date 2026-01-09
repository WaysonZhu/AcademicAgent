import chainlit as cl
import re
from main import SearchWorkflow
from config.settings import settings

# 初始化工作流实例
workflow_engine = SearchWorkflow()


def process_citations(text: str) -> str:
    """
    后处理函数：将 LLM 生成的引用标记替换为 Chainlit 可渲染的 Markdown 超链接。

    原始格式: [Response_Start]PaperID|Year|URL|FirstAuthor[Response_End]
    目标格式: [(FirstAuthor et al. Year)](URL)
    """
    # 正则表达式匹配 [Response_Start]...[Response_End]
    # 捕获组: 1=ID, 2=Year, 3=URL, 4=Author
    pattern = r"\[Response_Start\](.*?)\|(.*?)\|(.*?)\|(.*?)\[Response_End\]"

    def replace_func(match):
        try:
            # 提取字段
            # paper_id = match.group(1)
            year = match.group(2).strip()
            url = match.group(3).strip()
            author = match.group(4).strip()

            # 构造学术引用格式 (Author et al. Year)
            citation_text = f"({author} et al. {year})"

            # 构造 Markdown 链接: [显示文本](链接地址)
            return f"[{citation_text}]({url})"
        except Exception:
            # 如果解析失败，返回原文本或空串，防止崩溃
            return ""

    # 执行替换
    return re.sub(pattern, replace_func, text)


@cl.on_chat_start
async def start():
    """会话开始时的欢迎语"""
    await cl.Message(
        content="欢迎使用智能论文检索 Agent！\n请输入您的研究方向（例如：'AI Agent最新研究' 或 '大模型推理能力'），我将为您生成深度调研报告。").send()


@cl.on_message
async def main(message: cl.Message):
    """主消息循环"""
    user_query = message.content

    # 创建一个空的 Step 用于显示进度
    msg = cl.Message(content="")
    await msg.send()

    async def status_callback(log_text):
        """回调函数，用于更新 UI 上的步骤显示"""
        async with cl.Step(name="Agent Thinking", type="run") as step:
            step.output = log_text

    try:
        # 运行工作流
        raw_report = await workflow_engine.run(user_query, status_callback)

        # 对报告进行正则替换，渲染超链接
        final_report = process_citations(raw_report)

        # 发送最终报告
        msg.content = final_report
        await msg.update()

    except Exception as e:
        await cl.Message(content=f"系统运行出错: {str(e)}").send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)