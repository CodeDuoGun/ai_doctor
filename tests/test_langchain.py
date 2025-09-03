"""
LangGraph 多 Agent 工作流（单文件示例）
-------------------------------------------------
功能：
- Planner 规划Agent：拆解用户目标 -> 生成任务计划与路由信号
- Researcher 检索Agent：基于内置“迷你知识库”做 RAG / 搜索（示例可离线运行）
- Coder 代码Agent：生成并执行 Python 代码（可控 REPL，禁止危险内置）
- Critic 评审Agent：对结果做自检与改进建议
- Finalizer 汇总Agent：输出最终答复（含引用与可复现步骤）

特性：
- 使用 LangGraph StateGraph 管理有状态对话与多Agent协作
- 支持条件分支与多轮迭代（Plan → Research → Code → Critic → …）
- 具备最小可运行依赖（无需联网）；如有 API 可无缝切换到真实 LLM / 检索

依赖（建议版本）：
  pip install -U "langchain>=0.2" "langgraph>=0.2" langchain-openai
  # 可选：向量检索演示
  pip install -U faiss-cpu sentence-transformers

运行：
  export OPENAI_API_KEY=sk-...   # 如无可忽略，将使用一个极简本地假模型
  python langgraph_multi_agent_demo.py
"""
from __future__ import annotations
import os
import ast
import math
import json
import operator
from typing import Annotated, List, Optional, TypedDict, Sequence, Dict, Any

# LangChain / LangGraph 基础
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ------------------------------
# 0) 一个可离线运行的“极简 LLM”兜底（无 OPENAI_API_KEY 时启用）
# ------------------------------
class TinyOfflineLLM:
    """非常简化的离线 LLM：
    - 识别简单关键词生成结构化JSON或模板化回复
    - 仅用于本地跑通 Graph 流程与演示，不代表真实模型能力
    """
    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        user = ""
        for m in messages[::-1]:
            if isinstance(m, HumanMessage):
                user = m.content
                break
        # 简单路由/模板：
        if "plan" in user.lower() or "规划" in user: # type: ignore
            plan = [
                "澄清需求与约束",
                "检索必要资料与示例",
                "（可选）生成并执行代码以验证",
                "整理最终答案与参考",
            ]
            return AIMessage(content=json.dumps({
                "plan": plan,
                "needs_research": True,
                "needs_coding": "代码" in user or "实现" in user or "demo" in user
            }, ensure_ascii=False))
        if "improve" in user.lower() or "审查" in user or "评审" in user: # type: ignore
            return AIMessage(content="结果整体合理。若需更高可信度，建议：增加数据源对比、提供可复现代码片段、补充边界与异常处理。")
        # 默认回声/模板
        return AIMessage(content=f"(离线LLM) 我已理解：{user[:120]} ...")

# ------------------------------
# 1) 安全的 Python REPL（用于 Coder Agent 执行）
# ------------------------------
ALLOWED_BUILTINS = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range, "print": print}
SAFE_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}

class SafePythonREPL:
    """极简安全 REPL：
    - 禁止 import / 文件IO / 网络
    - 仅开放 math、部分内置函数
    """
    def __init__(self):
        self.globals: Dict[str, Any] = {"__builtins__": ALLOWED_BUILTINS, **SAFE_MATH}
        self.locals: Dict[str, Any] = {}

    def run(self, code: str) -> str:
        # 简单静态检查：
        if "import" in code or "open(" in code or "__" in code:
            return "[安全沙箱] 禁止 import / 文件IO / 魔术属性"
        try:
            tree = ast.parse(code)
        except Exception as e:
            return f"[解析错误] {e}"
        # 执行
        try:
            compiled = compile(tree, filename="<repl>", mode="exec")
            exec(compiled, self.globals, self.locals)
            # 如果最后一条是表达式，尝试求值
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                value = eval(compile(ast.Expression(tree.body[-1].value), "<repl>", "eval"), self.globals, self.locals)
                return repr(value)
            return "[执行完成，无返回]"
        except Exception as e:
            return f"[运行错误] {e}"

# ------------------------------
# 2) 一个迷你“离线检索器”（可替换为真实向量库/搜索）
# ------------------------------
MINI_KB = [
    {"title": "LangChain 简介", "text": "LangChain 是用于构建基于大语言模型(LLM)的应用的框架，包含链、提示、内存、工具、RAG、Agent 等。"},
    {"title": "LangGraph 简介", "text": "LangGraph 是 LangChain 的有状态扩展，使用图(节点、边、状态)构建多Agent流程，支持检查点与并发。"},
    {"title": "RAG 核心", "text": "RAG 包括文本切分、嵌入、向量检索与重排，常用库有 FAISS、Milvus、Weaviate、Pinecone。"},
]

def mini_search(query: str, k: int = 3) -> List[Dict[str, str]]:
    q = query.lower()
    scored = []
    for doc in MINI_KB:
        score = sum(1 for w in q.split() if w in doc["text"].lower())
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:k] if s >= 0]

# ------------------------------
# 3) 定义 LangGraph 的 State（全局共享状态）
# ------------------------------
class GraphState(TypedDict):
    # 对话消息（自动累加）
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # 计划与路由信号
    plan: Optional[List[str]]
    needs_research: bool
    needs_coding: bool
    # 中间产物
    research_notes: Annotated[List[str], operator.add]
    code_snippet: Optional[str]
    result: Optional[str]
    citations: Annotated[List[str], operator.add]

# ------------------------------
# 4) 准备 LLM 与工具
# ------------------------------
OPENAI = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) if OPENAI else TinyOfflineLLM()
repl = SafePythonREPL()

# ------------------------------
# 5) 各 Agent 节点实现
# ------------------------------
# 5.1 Planner：生成计划 + 路由信号
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是资深项目规划助手。请输出 JSON，字段 plan(List[str])、needs_research(bool)、needs_coding(bool)。只输出 JSON。"),
    ("human", "为以下任务制定执行计划，并判断是否需要‘检索’或‘写代码’来完成：\n{task}\n如果涉及实现/验证/数据处理，needs_coding 设为 true。若需要查资料或知识库，needs_research 设为 true。")
])

def planner_node(state: GraphState) -> Dict:
    last_user = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m.content
            break
    msg = planner_prompt.format_messages(task=last_user)
    ai: AIMessage = llm.invoke(msg) # type: ignore
    try:
        data = json.loads(ai.content) # type: ignore
    except Exception:
        # 兜底：
        data = {"plan": ["检索资料", "编写代码(如需要)", "整理结果"], "needs_research": True, "needs_coding": False}
    return {
        "messages": [AIMessage(content=f"规划完成：{data.get('plan')}")],
        "plan": data.get("plan"),
        "needs_research": bool(data.get("needs_research", False)),
        "needs_coding": bool(data.get("needs_coding", False)),
    }

# 5.2 Researcher：做检索/RAG，写入 research_notes 与 citations
research_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是检索助手。给出与任务最相关的3条资料摘要，每条一句，并注明来源标题。"),
    ("human", "任务：\n{task}\n已有计划：{plan}\n请基于内部知识库检索并输出要点（无需虚构）。")
])

def researcher_node(state: GraphState) -> Dict:
    # 取用户任务
    last_user = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m.content
            break
    # 迷你检索
    hits = mini_search(last_user or "langchain langgraph rag") # type: ignore
    notes = [f"{i+1}. {h['text']}" for i, h in enumerate(hits)]
    cites = [h["title"] for h in hits]

    # LLM 提炼（可选）
    msg = research_prompt.format_messages(task=last_user, plan=state.get("plan"))
    _ = llm.invoke(msg)  # 这里仅用于演示，可将 notes 交给 llm 做进一步摘要

    return {
        "messages": [AIMessage(content="检索完成，已记录关键要点与引用。")],
        "research_notes": notes,
        "citations": cites,
    }

# 5.3 Coder：产出并执行代码，写入 code_snippet 与 result
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是 Python 工程助手。生成尽可能短小且可直接运行的 Python 代码以完成任务；只输出代码，不要解释。"),
    ("human", "任务：\n{task}\n已知要点(可选)：\n{notes}\n请给出可运行的 Python 代码。若不需要代码，也可返回 print('无需代码')。")
])

def coder_node(state: GraphState) -> Dict:
    last_user = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m.content
            break
    notes = "\n".join(state.get("research_notes", []))
    code_msg = code_prompt.format_messages(task=last_user, notes=notes)
    ai: AIMessage = llm.invoke(code_msg) # type: ignore
    code = ai.content.strip() # type: ignore

    # 运行（在沙箱中）
    output = repl.run(code)
    return {
        "messages": [AIMessage(content=f"代码已执行，输出：{output[:200]}")],
        "code_snippet": code,
        "result": output,
    }

# 5.4 Critic：自检并决定是否需要再迭代
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是严谨的技术评审。请对结果进行自检，给出是否需要再次检索/编码的建议与理由。"),
    ("human", "任务：\n{task}\n计划：{plan}\n当前结果：\n{result}\n研究要点：\n{notes}\n请简要给出改进建议（无需客套），若已足够则说明‘可以收敛’。")
])

def critic_node(state: GraphState) -> Dict:
    last_user = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m.content
            break
    notes = "\n".join(state.get("research_notes", []))
    result = state.get("result")
    msg = critic_prompt.format_messages(task=last_user, plan=state.get("plan"), result=result, notes=notes)
    ai: AIMessage = llm.invoke(msg) # type: ignore
    advice = ai.content

    # 简单规则：若包含“建议再次检索/编码”等词，就回到相应分支；否则收敛
    rerun_research = any(k in advice for k in ["再次检索", "补充资料", "更多来源"])
    rerun_code = any(k in advice for k in ["补充代码", "完善代码", "修复"])

    return {
        "messages": [AIMessage(content=f"评审意见：{advice}")],
        "needs_research": bool(rerun_research),
        "needs_coding": bool(rerun_code),
    }

# 5.5 Finalizer：汇总输出（带引用与复现实验步骤）
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是严谨的总结助手。请以结构化要点输出结论、关键证据(引用标题)、以及可复现步骤(若有代码)。"),
    ("human", "任务：\n{task}\n计划：{plan}\n研究要点：\n{notes}\n代码：\n{code}\n执行结果：\n{result}\n请输出：结论 / 关键证据(引用标题) / 复现步骤。")
])

def finalizer_node(state: GraphState) -> Dict:
    last_user = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user = m.content
            break
    notes = "\n".join(state.get("research_notes", []))
    code = state.get("code_snippet") or "(无)"
    result = state.get("result") or "(无)"

    msg = final_prompt.format_messages(task=last_user, plan=state.get("plan"), notes=notes, code=code, result=result)
    ai: AIMessage = llm.invoke(msg) # type: ignore

    return {
        "messages": [AIMessage(content=ai.content)],
    }

# ------------------------------
# 6) 路由函数（条件边）
# ------------------------------
def route_from_planner(state: GraphState) -> str:
    if state.get("needs_research"):
        return "researcher"
    if state.get("needs_coding"):
        return "coder"
    return "finalizer"

def route_after_research(state: GraphState) -> str:
    # 有代码需求则去 coder，否则直接 final
    if state.get("needs_coding"):
        return "coder"
    return "finalizer"

def route_after_critic(state: GraphState) -> str:
    if state.get("needs_research"):
        return "researcher"
    if state.get("needs_coding"):
        return "coder"
    return "finalizer"

# ------------------------------
# 7) 构建 Graph
# ------------------------------
workflow = StateGraph(GraphState)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)
workflow.add_node("critic", critic_node)
workflow.add_node("finalizer", finalizer_node)

workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", route_from_planner, {
    "researcher": "researcher",
    "coder": "coder",
    "finalizer": "finalizer",
})
workflow.add_conditional_edges("researcher", route_after_research, {
    "coder": "coder",
    "finalizer": "finalizer",
})
# coder → critic → 条件跳转
workflow.add_edge("coder", "critic")
workflow.add_conditional_edges("critic", route_after_critic, {
    "researcher": "researcher",
    "coder": "coder",
    "finalizer": "finalizer",
})
workflow.add_edge("finalizer", END)

# 检查点（可恢复状态/多轮）
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ------------------------------
# 8) 运行示例
# ------------------------------
if __name__ == "__main__":
    user_task = (
        "请给出一份‘如何用 LangGraph 构建多Agent工作流’的简要说明，并用代码计算 1 到 100 的平方和。"
    )

    # 第一次流式运行（打印每步更新）
    print("\n=== 流式执行 ===")
    config = {"configurable": {"thread_id": "demo-session-1"}}
    for event in app.stream({
        "messages": [HumanMessage(content=user_task)],
        "plan": None,
        "needs_research": False,
        "needs_coding": False,
        "research_notes": [],
        "citations": [],
        "code_snippet": None,
        "result": None,
    }, config=config): # type: ignore
        for node, update in event.items():
            print(f"[node={node}] -> {list(update.keys())}")
    print("=== 执行完成 ===\n")

    # 获取最终结果
    final_state = app.get_state(config) # type: ignore
    final_messages = final_state.values.get("messages", [])
    print("\n=== 最终输出 ===")
    if final_messages:
        print(final_messages[-1].content)

    # 二次提问：沿用同一线程，Graph 会保留上下文
    print("\n=== 二次提问（继续在同一会话）===")
    follow_up = "补充列出关键参考标题。"
    for event in app.stream({"messages": [HumanMessage(content=follow_up)]}, config=config): # type: ignore
        pass
    state2 = app.get_state(config) # type: ignore
    msgs2 = state2.values.get("messages", [])
    print(msgs2[-1].content)
