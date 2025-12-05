import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, CodeExecutorAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_core.tools import FunctionTool
import os
from openai import api_key
import pandas as pd
from typing import Any, Dict, List, Sequence
import re
from dotenv import load_dotenv

load_dotenv()

planner_model_client = OpenAIChatCompletionClient(
    model = "qwen-plus-latest",
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = os.getenv("BASE_URL"),
    model_info = {
        "label": "qwen-plus-latest",
        "provider": "dashscope",
        "family": "qwen",
        "input_token_limit": 995904,
        "out_put_token_limit": 32768,
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "support_images": False,
        "is_reasoning_mopdel": True,
    },
    temperature=0.3,
)

coder_model_client = OpenAIChatCompletionClient(
    model = "qwen3-coder-plus",
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = os.getenv("BASE_URL"),
    model_info = {
        "label": "qwen3-coder-plus",
        "provider": "dashscope",
        "family": "qwen",
        "input_token_limit": 997952,
        "out_put_token_limit": 65536,
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "support_images": False,
        "is_reasoning_mopdel": True,
    },
    temperature=0.3,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    description="Human proxy agent responsible for interacting with the user",
    input_func=input,
)

def preview_csv_head(path: str, n:int=100) -> Dict[str, Any]:
    """Preview the head of a CSV file."""
    if not os.path.isfile(path):
        return {"ok": False, "error": f"File does not exist: {path}"}
    try:
        df = pd.read_csv(path)
        head = df.head(n).to_dict(orient="records")
        return {
            "ok": True,
            "path": path,
            "rows": len(df),
            "cols": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
            "head": head,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

csv_head_tool = FunctionTool(
    func=preview_csv_head,
    name="preview_csv_head",
    description="""
    Preview first N rows of a CSSV and return row number, column number, column names and column types.
    """,
)

planner = AssistantAgent(
    name = "Planner",
    description="Planner agent responsible for previwing the data and generating a plan",
    model_client=planner_model_client,
    tools=[csv_head_tool],
    system_message="""You are a data analysis planner. Create a structured analysis plan based on data preview and user request.

**Output ONLY the plan in this format:**

## Data Analysis Plan

### 1. Data Overview
- Data characteristics: [Brief description]
- File path: [User-provided path]
- Data scale: [Rows × Columns]

### 2. Analysis Objectives
- [Objective 1]
- [Objective 2]

### 3. Visualization Requirements
- Visualization needed: [Yes/No]
- Chart types: [List or "None"]

### 4. Modeling Requirements
- Modeling needed: [Yes/No]
- Model types: [List or "None"]

### 5. Detailed Steps
1. [Step 1]
2. [Step 2]
3. [Step 3]
4. [Step 4]
5. [Step 5]

### 6. Notes & Considerations
- [Consideration 1]
- [Consideration 2]

**Rules:**
- No additional text beyond this structure
- No questions to the user
- No conversational elements
- Base decisions on data preview results
    """,
)

coder = AssistantAgent(
    name = "Coder",
    description="Coder agent responsible for writing code according to the plan",
    model_client=coder_model_client,
    system_message="""You are a professional Python data analysis engineer (Coder). Your task is to generate executable Python code based on the analysis plan created by Planner.

**IMPORTANT LANGUAGE REQUIREMENT:**
- You MUST use **ENGLISH ONLY** for all print statements, comments, chart titles, labels, and error messages
- Do NOT use Chinese characters in the code
- Do NOT include Chinese in any output or comments
- Use simple, clear English for all text in the code

**Constraints:**
1. **Only use these third-party libraries:**
   - pandas, numpy, matplotlib, seaborn, sklearn, statsmodels, xgboost
   - Strictly prohibit using any other libraries

2. **Code Specifications:**
   - All code must be in a single code block
   - Must include complete import statements
   - Must handle file paths correctly
   - Must include exception handling mechanisms
   - Code must be self-contained and independently executable

3. **Image Processing Requirements:**
   - **Absolutely DO NOT use plt.show()**
   - All charts must be saved to files
   - Image save path: Same directory as the CSV file
   - Image filenames: Use meaningful English names like "correlation_heatmap.png"
   - Ensure images are saved before proceeding with subsequent operations

4. **Encoding Requirements:**
   - Add encoding='utf-8' when reading CSV files
   - Use English titles and labels for all charts
   - Ensure all string operations use UTF-8

**Code Structure Template:**
```python
# Import standard and allowed third-party libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.metrics import mean_squared_error
# ... other imports

# Set English font and chart style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_data(filepath):
    '''Load data'''
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        print(f"Data loaded successfully, shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")
        return df
    except UnicodeDecodeError:
        # Try alternative encodings for Windows
        try:
            df = pd.read_csv(filepath, encoding='gbk')
            print(f"Data loaded successfully (GBK encoding), shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Data loading failed: {e}")
            return None
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None

def save_plot(fig, filename, filepath):
    '''Save chart to file'''
    try:
        # Get directory of CSV file
        csv_dir = os.path.dirname(filepath)
        save_path = os.path.join(csv_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', encoding='utf-8')
        print(f"Chart saved: {save_path}")
    except Exception as e:
        print(f"Chart saving failed: {e}")

def main_analysis(csv_filepath):
    # Main analysis logic
    # Ensure all charts are saved, do not use plt.show()
    pass

if __name__ == "__main__":
    # Get file path from sys.argv or use fixed path
    import sys
    if len(sys.argv) > 1:
        csv_filepath = sys.argv[1]
    else:
        csv_filepath = "data.csv"
    
    main_analysis(csv_filepath)
```

5. **Important Reminders:**
   - All charts must be saved via fig.savefig(), do not use plt.show()
   - Ensure each chart has clear titles and labels
   - Use high resolution when saving charts (dpi=300)
   - Handle missing values and outliers
   - Output key statistical information

   **Simple Font Configuration:**
    Always include this code at the beginning of your script:
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
```
    """,
)

summerizer = AssistantAgent(
    name = "Summarizer",
    description="Summarizer agent responsible for summarizing all agents' messages and generating a summary",
    model_client=planner_model_client,
    system_message="""
## 3. Summarizer Agent Prompt (English)

```text
You are a data analysis summarization expert (Summarizer). Your task is to provide users with clear, comprehensive analysis summaries based on Planner's plan and Terminal's execution results.

**Input Information:**
1. Planner's original analysis plan
2. Terminal's code execution output
3. User's initial question/requirement

**Summary Requirements:**

**Output Format:**
Please generate a professional data analysis report using Markdown format:

# Data Analysis Report

## 📊 Project Overview
- **Analysis Objective**: [Brief description]
- **Data Source**: [File path]
- **Analysis Time**: [Current time]

## 📈 Basic Data Information
| Item | Details |
|------|---------|
| Data Scale | {Row count} rows × {Column count} columns |
| Key Fields | [List important columns] |
| Data Types | [Numerical/Categorical/Time series, etc.] |
| Missing Values | [Missing value statistics] |

## 🔍 Key Findings
### 1. Key Finding 1
[Describe finding 1, reference relevant statistics or charts]

### 2. Key Finding 2  
[Describe finding 2, reference relevant statistics or charts]

### 3. Key Finding 3
[Describe finding 3, reference relevant statistics or charts]

## 📊 Visualization Results
### Generated Charts:
1. **Chart 1 Name**: [Chart description, what information it shows]
   - File path: [Save path]
   - Key Insight: [Patterns observed from the chart]

2. **Chart 2 Name**: [Chart description, what information it shows]
   - File path: [Save path]
   - Key Insight: [Patterns observed from the chart]

## 🤖 Modeling Analysis (if applicable)
### Model Overview
- **Model Type**: [Regression/Classification/Clustering, etc.]
- **Algorithm Used**: [Specific algorithm]

### Model Performance
| Metric | Value | Explanation |
|--------|-------|-------------|
| Accuracy/MAE/R² | [Value] | [Performance evaluation] |
| Feature Importance | [Important features] | [Explanation] |

## ✅ User Question Answer
**[Directly answer the user's original question]**
- [Answer based on analysis results]
- [Supporting evidence]

## 💡 Conclusions & Recommendations
### Main Conclusions
1. [Conclusion 1]
2. [Conclusion 2]
3. [Conclusion 3]

### Actionable Recommendations
1. [Recommendation 1]
2. [Recommendation 2]

## ⚠️ Notes & Considerations
- **Analysis Limitations**: [Existing limitations]
- **Data Assumptions**: [Assumptions made]
- **Follow-up Suggestions**: [Directions for further analysis]

Ensure the summary is professional, objective, and based on actual analysis results.
    """,
)

terminal = CodeExecutorAgent(
    name = "Terminal",
    description="Terminal agent responsible for executing code",
    code_executor=LocalCommandLineCodeExecutor(),
)

termination = TextMentionTermination(text="APPROVE")

def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    """
    根据对话历史决定下一个发言的智能体。
    流程：Planner调用工具预览数据 -> Planner生成计划 -> Coder生成代码 -> Terminal执行代码
          -> 如果成功：Summarizer生成报告
          -> 如果失败：Coder修改代码 -> Terminal重新执行
    """
    if not messages:
        return "Planner"  # 初始状态：Planner先开始
    
    last_message = messages[-1]
    last_source = last_message.source if hasattr(last_message, 'source') else None
    last_content = last_message.content if hasattr(last_message, 'content') else ""
    
    # 获取消息类型
    message_type = type(last_message).__name__
    
    # 判断是否为工具调用结果
    is_tool_result = False
    if last_source == "Planner" and "preview_csv_head" in str(last_content):
        is_tool_result = True
    
    # 1. 用户提问后，Planner开始
    if last_source == "user_proxy":
        return "Planner"
    
    # 2. Planner相关逻辑
    elif last_source == "Planner":
        # 如果Planner的消息中包含完整的数据分析计划（有Data Analysis Plan标题）
        if "## Data Analysis Plan" in last_content:
            return "Coder"  # 计划完成，转到Coder
        else:
            # 检查是否需要继续让Planner调用工具
            # 如果消息中还没有显示完整的数据预览结果，继续让Planner工作
            if "Data Overview" not in last_content:
                return "Planner"
            # 如果已经有Data Overview但还没有完整计划，也继续
            if "Data Overview" in last_content and "## Data Analysis Plan" not in last_content:
                return "Planner"
            # 默认继续Planner
            return "Planner"
    
    # 3. 工具调用结果处理（工具结果可能来自Planner的调用）
    elif is_tool_result:
        return "Planner"  # 工具调用后，继续让Planner处理结果并生成计划
    
    # 4. Coder相关逻辑
    elif last_source == "Coder":
        # 检查Coder是否生成了代码（包含代码块）
        if "```python" in last_content:
            return "Terminal"  # 有代码，转到Terminal执行
        else:
            return "Coder"  # 没有代码，继续让Coder生成
    
    # 5. Terminal相关逻辑
    elif last_source == "Terminal":
        # 检查Terminal执行结果
        content_str = str(last_content)
        
        # 判断执行是否出错
        has_error = False
        error_indicators = [
            "Error:", "Error:", "error:", "Exception:", "Traceback", "ImportError",
            "ModuleNotFoundError", "SyntaxError", "NameError", "TypeError",
            "ValueError", "FileNotFoundError", "KeyError", "AttributeError"
        ]
        
        for indicator in error_indicators:
            if indicator in content_str:
                has_error = True
                break
        
        # 检查常见的Python错误模式
        if re.search(r"Error\s*:", content_str, re.IGNORECASE):
            has_error = True
        if re.search(r"Traceback.*most recent call", content_str, re.IGNORECASE):
            has_error = True
        if re.search(r"File.*line.*\n.*Error", content_str, re.IGNORECASE):
            has_error = True
        
        if has_error:
            print(f"DEBUG: Detected error in Terminal output, returning to Coder")
            return "Coder"  # 执行出错，返回Coder修改代码
        else:
            print(f"DEBUG: Terminal execution successful, moving to Summarizer")
            return "Summarizer"  # 执行成功，转到Summarizer
    
    # 6. Summarizer相关逻辑
    elif last_source == "Summarizer":
        # Summarizer完成后，返回user_proxy等待用户确认
        return "user_proxy"
    
    # 7. 默认情况：按顺序进行
    else:
        # 如果没有识别到来源，按顺序尝试
        agents_order = ["Planner", "Coder", "Terminal", "Summarizer", "user_proxy"]
        
        # 查找最后一个已知的agent
        for i, agent in enumerate(agents_order):
            if last_source == agent:
                # 返回下一个agent（循环）
                next_index = (i + 1) % len(agents_order)
                return agents_order[next_index]
        
        # 如果都没找到，默认返回Planner
        return "Planner"

team = SelectorGroupChat(
    [planner, coder, terminal, summerizer, user_proxy],
    termination_condition=termination,
    model_client=planner_model_client,
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
    stream = team.run_stream(task=r"""
    文件 'C:\Users\bsh97\Documents\Projects\DataAnalyst2\files\real_estate_data.csv' 已上传。
    分析不同类型房产的价格分布情况"""
    )
    
    await Console(stream)
    await planner_model_client.close()
    await coder_model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
