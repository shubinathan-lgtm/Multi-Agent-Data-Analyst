import asyncio
from socket import timeout
import sys
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, CodeExecutorAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage, MultiModalMessage
from autogen_agentchat.ui import Console
from autogen_agentchat.base import TaskResult
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from typing import Any, Dict, List, Sequence
import re
from dotenv import load_dotenv

from prompt import planner_prompt, coder_prompt, summarizer_prompt
from model_client import general_model_client, coder_model_client
from tools import csv_head_tool, gen_analytics_report_tool

load_dotenv()

# Apply the Windows event loop policy fix
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

user_proxy = UserProxyAgent(
    name="user_proxy",
    description="Human proxy agent responsible for interacting with the user",
    input_func=input,
)

planner = AssistantAgent(
    name = "Planner",
    description="Planner agent responsible for previwing the data and generating a plan",
    model_client=general_model_client,
    tools=[csv_head_tool],
    system_message=planner_prompt,
)

coder = AssistantAgent(
    name = "Coder",
    description="Coder agent responsible for writing code according to the plan",
    model_client=coder_model_client,
    system_message=coder_prompt,
)

summarizer = AssistantAgent(
    name = "Summarizer",
    description="Summarizer agent responsible for summarizing all agents' messages and generating a summary and save it to a file",
    model_client=general_model_client,
    tools=[gen_analytics_report_tool],
    system_message=summarizer_prompt,
)

terminal = CodeExecutorAgent(
    name = "Terminal",
    description="Terminal agent responsible for executing code",
    code_executor=LocalCommandLineCodeExecutor(work_dir=".", timeout=600),
)

termination = TextMentionTermination(text="APPROVE")

def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    if not messages:
        return "Planner"  # 初始状态：Planner先开始
    
    last_message = messages[-1]
    last_source = last_message.source if hasattr(last_message, 'source') else None
    last_content = str(last_message.content) if hasattr(last_message, 'content') else ""

    messages_type = type(last_message).__name__

    is_planner_tool_call_request_event = False
    if last_source == "Planner" and "FunctionCall" in str(last_content):
        is_planner_tool_call_request_event = True

    is_planner_tool_call_execution_event = False
    if last_source == "Planner" and "FunctionExecutionResult" in str(last_content):
        is_planner_tool_call_execution_event = True

    is_summarizer_tool_call_request_event = False
    if last_source == "Summarizer" and "FunctionCall" in str(last_content):
        is_summarizer_tool_call_request_event = True

    is_summarizer_tool_execution_event = False
    if last_source == "Summarizer" and "FunctionExecutionResult" in str(last_content):
        is_summarizer_tool_execution_event = True

    if last_source == "user_proxy":
        return "Planner"

    elif last_source == "Planner":
        if "## Data Analysis Plan" in last_content:
            return "Coder"
        return "Planner"

    elif is_planner_tool_call_request_event:
        return "Planner"

    elif is_planner_tool_call_execution_event:
        return "Planner"

    elif last_source == "Coder":
        if "```python" in last_content:
            return "Terminal"
        return "Coder"

    elif last_source == "Terminal":
        content_str = str(last_content)
        has_error = False
        error_indicators = [
            "Error:", "error:", "Exception:", "Traceback", "ImportError",
            "ModuleNotFoundError", "SyntaxError", "NameError", "TypeError",
            "ValueError", "FileNotFoundError", "KeyError", "AttributeError", "failed", "Failed"
        ]
        for indicator in error_indicators:
            if indicator in content_str:
                has_error = True
                break
        
        if re.search(r"Error\s*:", content_str, re.IGNORECASE):
            has_error = True
        if re.search(r"Traceback.*most recent call", content_str, re.IGNORECASE):
            has_error = True
        if re.search(r"File.*line.*\n.*Error", content_str, re.IGNORECASE):
            has_error = True
        
        if has_error:
            return "Coder"
        else:
            return "Summarizer"

    elif last_source == "Summarizer":
        if is_summarizer_tool_call_request_event:
            return "Summarizer"
        elif is_summarizer_tool_execution_event:
            return "Summarizer"
        elif last_source == "Summarizer":
            return "user_proxy"
        print("Report generating completed.")
        return "user_proxy"

team = SelectorGroupChat(
    [planner, coder, terminal, summarizer, user_proxy],
    termination_condition=termination,
    model_client=general_model_client,
    selector_func=selector_func,
    selector_prompt="You are the coordinator of a multi-agent team. Decide which agent speaks next based on conversation history.",
    allow_repeated_speaker=True,
    max_turns=50
)

async def main():
    # stream = team.run_stream(task=r"""
    # 文件 'C:\Users\bsh97\Documents\Projects\DataAnalyst2\files\large_sales_data.csv' 已上传。
    # 请完成以下分析：
    # 1. 数据质量检查与清洗
    # 2. 过去两年月度销售额可视化
    # 3. 识别销售额的季节性模式
    # 4. 预测未来6个月的销售额""")
    # stream = team.run_stream(task=r"""
    # 文件 'C:/Users/bsh97/Documents/Projects/DataAnalyst2/files/real_estate_data.csv' 已上传。
    # 预测未来房价走势，请用中文回答"""
    # )
    
    async for message in team.run_stream(task="""文件 'C:/Users/bsh97/Documents/Projects/DataAnalyst2/files/real_estate_data.csv' 已上传。
预测未来房价走势，请用中文回答"""):  # type: ignore
        if isinstance(message, TaskResult):
            print("Stop Reason:", message.stop_reason)
        elif message.source == "Planner":
            if message.type in ["ToolCallRequestEvent", "ToolCallExecutionEvent"]:
                print(message.source, ":正在预览数据...")
            elif message.type == "ToolCallSummaryMessage":
                print(message.source, ":正在制定分析计划...")
            else:
                print(message.source, ":", message.content)
                print("分析计划指定完成，Coder正在编写代码...")
        elif message.source == "Coder":
            print(message.source, ":", message.content)
            print("Terminal正在运行代码...")
        elif message.source == "Terminal":
            has_error = False
            error_indicators = [
                "Error:", "error:", "Exception:", "Traceback", "ImportError",
                "ModuleNotFoundError", "SyntaxError", "NameError", "TypeError",
                "ValueError", "FileNotFoundError", "KeyError", "AttributeError", "failed", "Failed"
            ]
            for indicator in error_indicators:
                if indicator in message.content:
                    has_error = True
                    break
        
            if re.search(r"Error\s*:", message.content, re.IGNORECASE):
                has_error = True
            if re.search(r"Traceback.*most recent call", message.content, re.IGNORECASE):
                has_error = True
            if re.search(r"File.*line.*\n.*Error", message.content, re.IGNORECASE):
                has_error = True
            print(message.source, ":", message.content)
            if has_error:
                print(message.source, ":代码执行出错，Coder重新生成代码...")
            else:
                print(message.source, ":代码执行成功，Summarizer正在生成报告...")
        elif message.source == "Summarizer":
            if message.type in ["ToolCallRequestEvent", "ToolCallExecutionEvent"]:
                print(message.source, ":生成报告文件")
            elif message.type == "ToolCallSummaryMessage":
                content = eval(message.content)
                file_path= content['path']
                print(message.source, f":报告文件已生成，位置{file_path}")
                print(message.source, message.content)
            else:
                print(message.source, ":", message.content)
        else:
            print(message.source, ":", message.content)
    await general_model_client.close()
    await coder_model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
