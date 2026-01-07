import chainlit as cl
from main import SearchWorkflow
from config.settings import settings

# 初始化工作流实例
workflow_engine = SearchWorkflow()


@cl.on_chat_start
async def start():
    """会话开始时的欢迎语"""
    await cl.Message(
        content="👋 欢迎使用智能论文检索 Agent！\n请输入您的研究方向（例如：'AI Agent最新研究' 或 '大模型推理能力'），我将为您生成深度调研报告。").send()


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
        # 也可以选择追加到主消息中
        # msg.content += f"\n> {log_text}"
        # await msg.update()

    try:
        # 运行工作流
        final_report = await workflow_engine.run(user_query, status_callback)

        # 发送最终报告
        msg.content = final_report
        await msg.update()

    except Exception as e:
        await cl.Message(content=f"❌ 系统运行出错: {str(e)}").send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)