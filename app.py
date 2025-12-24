import streamlit as st
import os
import asyncio
import pandas as pd
from datetime import datetime
import nest_asyncio
import re
from data_analyst_team import team

def process_markdown_with_images(markdown_content):
    """
    处理包含图片链接的Markdown内容，在Streamlit中显示图片
    """
    import re
    
    # 查找Markdown图片语法: ![alt text](image_path)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    # 分割文本和图片
    parts = re.split(image_pattern, markdown_content)
    
    # parts数组中，每3个元素为一组：[text, alt_text, image_path]
    i = 0
    while i < len(parts):
        # 普通文本部分
        if parts[i].strip():
            st.markdown(parts[i])
        i += 1
        
        # 如果还有元素，说明后面跟着alt_text和image_path
        if i < len(parts):
            alt_text = parts[i]
            i += 1
            if i < len(parts):
                image_path = parts[i]
                i += 1
                # 显示图片
                try:
                    if os.path.exists(image_path):
                        st.image(image_path, caption=alt_text if alt_text else None)
                    else:
                        # 如果图片不存在，显示原始Markdown
                        st.markdown(f"![{alt_text}]({image_path})")
                except Exception as e:
                    # 如果加载图片失败，显示原始Markdown
                    st.markdown(f"![{alt_text}]({image_path})")
# 应用 Nest Asyncio 解决事件循环冲突
nest_asyncio.apply()

# 设置页面配置
st.set_page_config(
    page_title="数据分析聊天机器人",
    page_icon="🤖",
    layout="wide"
)

# 初始化 session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
    
if 'file_saved_path' not in st.session_state:
    st.session_state.file_saved_path = None
    
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False
    
if 'waiting_for_user_input' not in st.session_state:
    st.session_state.waiting_for_user_input = False
    
if 'current_task' not in st.session_state:
    st.session_state.current_task = None

# 创建数据目录
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 页面标题
st.title("📊 数据分析聊天机器人")

# 侧边栏
with st.sidebar:
    st.header("📁 文件上传")
    uploaded_file = st.file_uploader(
        "请选择一个 CSV 文件",
        type=["csv"],
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        # 如果有新文件上传，重置状态
        if st.session_state.uploaded_file != uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.messages = []
            st.session_state.analysis_started = False
            
        # 显示文件信息
        st.success("文件上传成功！")
        st.info(f"文件名: {uploaded_file.name}")
        st.info(f"文件大小: {uploaded_file.size} bytes")
        
        # 保存文件到指定目录
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_dir = os.path.join(DATA_DIR, timestamp)
        os.makedirs(file_dir, exist_ok=True)
        
        file_path = os.path.join(file_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_path = file_path.replace("\\", "/")
        st.session_state.file_saved_path = file_path
        st.success(f"文件已保存至: {file_path}")

# 主界面
st.subheader("💬 与AI分析师对话")

# 显示消息历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "markdown":
            st.markdown(message["content"])
        elif message["type"] == "code":
            st.code(message["content"], language="python")
        elif message["type"] == "terminal":
            st.text_area("终端输出", value=message["content"], height=200, key=f"terminal_{message['id']}")
        else:
            st.write(message["content"])

# 用户输入
if st.session_state.file_saved_path:
    user_input_text = None
    
    # 检查是否在等待用户输入或者接收新问题
    if st.session_state.waiting_for_user_input:
        user_input_text = st.chat_input("请输入您的回复...")
        input_context = "follow_up"
    else:
        user_question = st.chat_input("请输入您的数据分析问题...")
        user_input_text = user_question
        input_context = "initial"
    
    if user_input_text:
        # 添加用户消息到历史记录
        st.session_state.messages.append({
            "role": "user",
            "content": user_input_text,
            "type": "text"
        })
        
        # 显示用户消息（在历史记录之后添加）
        with st.chat_message("user"):
            st.write(user_input_text)
            
        # 构造任务提示
        if st.session_state.waiting_for_user_input:
            user_prompt = user_input_text  # 直接传递用户回复
            # 重置等待状态
            st.session_state.waiting_for_user_input = False
        else:
            user_prompt = f"文件 '{st.session_state.file_saved_path}' 已上传。\n{user_input_text}"
            # 保存当前任务
            st.session_state.current_task = user_prompt
        
        # 添加助手消息容器
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # 运行分析 (修改为流式处理)
            async def run_analysis():
                response_container = st.container()
                response_messages = []
                
                try:
                    async for message in team.run_stream(task=user_prompt):
                        # 处理不同类型的消息
                        msg_content = ""
                        msg_source = "Unknown"
                        
                        # 安全地获取消息内容和来源
                        if hasattr(message, 'content'):
                            content_attr = getattr(message, 'content', "")
                            msg_content = str(content_attr) if content_attr else ""
                        
                        if hasattr(message, 'source'):
                            msg_source = getattr(message, 'source', "Unknown")
                        
                        # 打印到控制台（模拟原始main函数的行为）
                        print(f"{msg_source}: {msg_content}")
                        
                        # 特殊处理 TaskResult 类型的消息
                        if hasattr(message, 'stop_reason'):
                            stop_reason = getattr(message, 'stop_reason', 'Unknown')
                            with response_container:
                                st.markdown(f"**分析完成**。停止原因: {stop_reason}")
                            response_messages.append({
                                "role": "assistant",
                                "content": f"分析完成。停止原因: {stop_reason}",
                                "type": "text",
                                "source": "System"
                            })
                            continue
                        
                        # 根据源和类型处理消息
                        if msg_source == "Planner":
                            # 安全地检查消息类型
                            msg_type = getattr(message, 'type', None)
                            if msg_type in ["ToolCallRequestEvent", "ToolCallExecutionEvent"]:
                                with response_container:
                                    st.markdown("**Planner**: 正在预览数据...")
                            elif msg_type == "ToolCallSummaryMessage":
                                with response_container:
                                    st.markdown("**Planner**: 正在制定分析计划...")
                            else:
                                with response_container:
                                    st.markdown(f"**Planner**:\n\n")
                                    st.markdown(msg_content)
                                response_messages.append({
                                    "role": "assistant",
                                    "content": msg_content,
                                    "type": "text",
                                    "source": "Planner"
                                })
                                with response_container:
                                    st.success("**Coder**正在生成代码...")
                                
                        elif msg_source == "Coder":
                            with response_container:
                                st.markdown("**Coder**:")
                                # 分离代码和文本部分
                                parts = msg_content.split("```")
                                for i, part in enumerate(parts):
                                    if i % 2 == 1:  # 代码块
                                        if part.startswith("python"):
                                            code_content = part[6:].strip()
                                            st.code(code_content, language="python")
                                        else:
                                            st.code(part.strip())
                                    else:  # 文本部分
                                        if part.strip():
                                            st.markdown(part.strip())
                                st.markdown("---\nTerminal正在运行代码...")
                            response_messages.append({
                                "role": "assistant",
                                "content": msg_content,
                                "type": "text",
                                "source": "Coder"
                            })
                            
                        elif msg_source == "Terminal":
                            has_error = False
                            error_indicators = [
                                "Error:", "error:", "Exception:", "Traceback", "ImportError",
                                "ModuleNotFoundError", "SyntaxError", "NameError", "TypeError",
                                "ValueError", "FileNotFoundError", "KeyError", "AttributeError", "failed", "Failed"
                            ]
                            for indicator in error_indicators:
                                if indicator in msg_content:
                                    has_error = True
                                    break
                            
                            if re.search(r"Error\s*:", msg_content, re.IGNORECASE):
                                has_error = True
                            if re.search(r"Traceback.*most recent call", msg_content, re.IGNORECASE):
                                has_error = True
                            if re.search(r"File.*line.*\n.*Error", msg_content, re.IGNORECASE):
                                has_error = True
                            
                            with response_container:
                                st.markdown("**Terminal**:")
                                st.text_area("", value=msg_content, height=200, key=f"terminal_{len(response_messages)}")
                                if has_error:
                                    st.warning("代码执行出错，Coder重新生成代码...")
                                else:
                                    st.success("代码执行成功，Summarizer正在生成报告...")
                            
                            response_messages.append({
                                "role": "assistant",
                                "content": msg_content,
                                "type": "terminal",
                                "source": "Terminal",
                                "id": len(response_messages)
                            })
                            
                        elif msg_source == "Summarizer":
                            # 安全地检查消息类型
                            msg_type = getattr(message, 'type', None)
                            if msg_type in ["ToolCallRequestEvent", "ToolCallExecutionEvent"]:
                                with response_container:
                                    st.markdown("**Summarizer**: 生成报告文件")
                            elif msg_type == "ToolCallSummaryMessage":
                                try:
                                    content = eval(msg_content)
                                    file_path = content['path']
                                    with response_container:
                                        st.markdown(f"**Summarizer**: 报告文件已生成，位置 {file_path}")
                                except:
                                    with response_container:
                                        st.markdown(f"**Summarizer**: {msg_content}")
                            else:
                                with response_container:
                                    st.markdown(f"**Summarizer**:\n\n")
                                    # 处理可能包含图片的内容
                                    process_markdown_with_images(msg_content)
                            
                            response_messages.append({
                                "role": "assistant",
                                "content": msg_content,
                                "type": "text",
                                "source": "Summarizer"
                            })
                            
                        elif msg_source == "user_proxy":
                            # 当user_proxy请求输入时，允许用户继续输入
                            with response_container:
                                st.markdown("**user_proxy**: 等待用户输入...")
                                # 这里我们设置一个标志，让外部知道需要用户输入
                                st.session_state.waiting_for_user_input = True
                                
                            response_messages.append({
                                "role": "assistant",
                                "content": "等待用户输入...",
                                "type": "text",
                                "source": "user_proxy"
                            })
                            
                        else:
                            with response_container:
                                st.markdown(f"**{msg_source}**:\n\n{msg_content}")
                            response_messages.append({
                                "role": "assistant",
                                "content": msg_content,
                                "type": "text",
                                "source": msg_source
                            })
                    
                    return response_messages
                    
                except Exception as e:
                    error_msg = f"分析过程中发生错误: {str(e)}"
                    with response_container:
                        st.error(error_msg)
                    return [{
                        "role": "assistant",
                        "content": error_msg,
                        "type": "text",
                        "source": "System"
                    }]
            
            # 运行异步分析
            response_messages = asyncio.run(run_analysis())
            
            # 将响应消息添加到历史记录中
            for msg in response_messages:
                st.session_state.messages.append(msg)

else:
    st.info("请先在侧边栏上传一个 CSV 文件开始分析")
    
    # 显示示例数据格式
    st.subheader("📋 示例数据格式")
    sample_data = pd.DataFrame({
        '日期': ['2023-01-01', '2023-01-02', '2023-01-03'],
        '销售额': [1000, 1200, 900],
        '产品类别': ['电子产品', '服装', '食品']
    })
    st.dataframe(sample_data)
    st.caption("支持各种 CSV 格式的数据文件")



