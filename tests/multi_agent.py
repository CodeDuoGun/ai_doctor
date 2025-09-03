"""
LangGraph 多 Agent 医疗工单流（单文件示例）
-------------------------------------------------
功能：
- Intake Agent（接诊）：结构化收集病人主诉、病史、既往史与约束（过敏、用药等）
- Retrieval Agent（检索）：对医院/指南/病例库做 RAG 检索，返回要点与证据标题
- Clinician Agent（临床推理）：基于检索结果与接诊信息给出诊断思路、建议检查与初步处理
- Compliance Agent（合规/风险控制）：检查是否涉及禁忌、药物处方或高风险建议，标注并给出替代方案
- Finalizer（汇总）：生成结构化病历摘要与后续建议（含引用与复现步骤）

特性：
- 使用 LangGraph StateGraph 管理会话状态与多 Agent 协作
- 可离线运行（内置 TinyOfflineLLM 与迷你知识库），也可接真实 LLM（OpenAI）
- 支持检查点与会话恢复

依赖（建议）：
  pip install -U "langchain>=0.2" "langgraph>=0.2" langchain-openai
  # 可选：更真实检索
  pip install -U faiss-cpu sentence-transformers

注意：本示例为演示用途，不构成医学诊断或医疗建议。任何临床决策请以专业医生为准。

运行：
  export OPENAI_API_KEY=sk-...   # 可选
  python langgraph_medical_workflow.py
"""
from __future__ import annotations
import os
import json
from typing import List, Optional, Dict, Any, TypedDict

# LangChain / LangGraph 基础（示例 API，按你本地版本调整）
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# ------------------------------
# 极简离线 LLM（无 OPENAI_KEY 时启用）
# ------------------------------
class TinyOfflineLLM:
    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        user = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user = m.content
                break
        # 简单规则化回复，结构化为 JSON 或模板文本
        if "主诉" in user or "症状" in user or "病史" in user:
            return AIMessage(content=json.dumps({
                "structured": {
                    "chief_complaint": "发热、咳嗽",
                    "history": "起病3天，伴乏力，无明显慢性病史",
                    "constraints": "无已知药物过敏"
                }
            }, ensure_ascii=False))
        if "检索" in user or "查找" in user or "指南" in user:
            hits = [
                {"title": "上呼吸道感染诊疗指南", "snippet": "首选对症处理，必要时行血常规与胸片。"},
                {"title": "社区获得性肺炎处理建议", "snippet": "有呼吸窘迫/发热高应考虑抗感染及住院评估。"}
            ]
            return AIMessage(content=json.dumps(hits, ensure_ascii=False))
        # 默认
        return AIMessage(content=f"(离线LLM) 已收到：{user[:120]}")

# ------------------------------
# 迷你知识库检索（演示）
# ------------------------------
MED_KB = [
    {"title": "上呼吸道感染指南", "text": "对症处理，退热，必要时化验"},
    {"title": "社区获得性肺炎指南", "text": "胸片、经验抗感染、评估住院指征"},
    {"title": "药物过敏警示", "text": "记录受试者既往过敏史，避免相关药物"},
]

def mini_med_search(query: str, k: int = 3) -> List[Dict[str, str]]:
    q = query.lower()
    scored = []
    for doc in MED_KB:
        score = sum(1 for w in q.split() if w in doc["text"].lower() or w in doc["title"].lower())
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:k] if s >= 0]

# ------------------------------
# Graph 状态定义
# ------------------------------
class MedState(TypedDict):
    messages: List[BaseMessage]
    intake_struct: Optional[Dict[str, Any]]
    research_notes: List[str]
    citations: List[str]
    clinician_note: Optional[str]
    compliance_flags: List[str]
    final_summary: Optional[str]

# ------------------------------
# LLM & Checkpoint
# ------------------------------
USE_DOUBAO = True
if USE_DOUBAO:
    from langchain_doubao import ChatDoubao
    llm = ChatDoubao(model="ep-20240811150229-kxgqk", temperature=0)
else:
    OPENAI = os.environ.get("OPENAI_API_KEY")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) if OPENAI else TinyOfflineLLM() # type: ignore
memory = MemorySaver()

# ------------------------------
# Agent 节点实现
# ------------------------------
# Intake Agent：结构化收集并做初步规范化
intake_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是临床接诊助手。把患者自然表述转换为结构化字段：chief_complaint, duration, past_medical_history, medication, allergies, vitals_if_provided。输出 JSON。"),
    ("human", "患者陈述：\n{raw}\n请只输出 JSON。")
])

def intake_node(state: MedState) -> Dict:
    raw = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            raw = m.content
            break
    if not raw:
        return {"messages": [AIMessage(content="未发现患者陈述")]} 
    msg = intake_prompt.format_messages(raw=raw)
    ai = llm.invoke(msg)
    try:
        structured = json.loads(ai.content) # type: ignore
    except Exception:
        # 兜底：尝试 TinyOfflineLLM 格式或手动填充
        try:
            # TinyOfflineLLM 返回嵌套 JSON 字符串 under 'structured'
            d = json.loads(ai.content) # type: ignore
            structured = d.get("structured") if isinstance(d, dict) else {"chief_complaint": raw}
        except Exception:
            structured = {"chief_complaint": raw}

    return {
        "messages": [AIMessage(content="接诊信息已结构化")],
        "intake_struct": structured,
    }

# Retrieval Agent：RAG 检索并摘录要点
research_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是检索助手。给出与患者问题最相关的3条指南/证据的简短要点与来源标题。输出 JSON 列表。"),
    ("human", "任务上下文：\n{context}\n请检索并返回要点。")
])

def research_node(state: MedState) -> Dict:
    # 使用 intake_struct 的摘要进行检索
    context = state.get("intake_struct") or {}
    q = " ".join(str(v) for v in context.values())
    hits = mini_med_search(q)
    notes = [f"{i+1}. {h['text']}" for i, h in enumerate(hits)]
    cites = [h['title'] for h in hits]
    # 可调用 LLM 进一步精炼（演示）
    _ = llm.invoke(research_prompt.format_messages(context=json.dumps(context, ensure_ascii=False)))
    return {
        "messages": [AIMessage(content="检索完成")],
        "research_notes": notes,
        "citations": cites,
    }

# Clinician Agent：临床推理与建议（诊断思路、必要检查、初步处理）
clinician_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是有资深经验的临床医生。基于接诊资料和检索到的证据，给出诊断思路(可能性排序)、建议的检查、初步处置(注意风险)，并标出不确定项。输出结构化 JSON：diagnoses(list), recommended_tests(list), initial_management(list), uncertainties(list)。"),
    ("human", "接诊：{intake}\n证据：{evidence}\n请输出 JSON。")
])

def clinician_node(state: MedState) -> Dict:
    intake = state.get("intake_struct") or {}
    evidence = "\n".join(state.get("research_notes", []))
    msg = clinician_prompt.format_messages(intake=json.dumps(intake, ensure_ascii=False), evidence=evidence)
    ai = llm.invoke(msg)
    try:
        clinical = json.loads(ai.content) # type: ignore
    except Exception:
        clinical = {"diagnoses": ["上呼吸道感染(待排除肺炎)"], "recommended_tests": ["血常规, 胸片(如有呼吸道症状加重)"], "initial_management": ["对症处理, 观察"], "uncertainties": ["是否并发肺炎需影像学证实"]}

    return {
        "messages": [AIMessage(content="临床建议已生成")],
        "clinician_note": json.dumps(clinical, ensure_ascii=False),
    }

# Compliance Agent：合规与风险检查（标注高风险或处方类建议）
compliance_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是医疗合规与质量控制审核员。检查临床建议是否存在处方药推荐、侵入性操作建议、或需要医生面诊/紧急处理的高风险信息。输出 JSON：flags(list), actions(list)。"),
    ("human", "临床建议：{clinical}\n接诊信息：{intake}\n请审核并输出 JSON。")
])

def compliance_node(state: MedState) -> Dict:
    clinical = state.get("clinician_note") or ""
    intake = state.get("intake_struct") or {}
    msg = compliance_prompt.format_messages(clinical=clinical, intake=json.dumps(intake, ensure_ascii=False))
    ai = llm.invoke(msg)
    try:
        review = json.loads(ai.content) # type: ignore
    except Exception:
        # 简单规则：若 initial_management 含 '抗感染' 或 '处方'，则标记需处方医生复核
        try:
            c = json.loads(clinical)
            flags = []
            actions = []
            if any('抗' in s or '处方' in s for s in c.get('initial_management', [])):
                flags.append('需处方医生复核')
                actions.append('安排医生面诊或远程处方审核')
            review = {"flags": flags, "actions": actions}
        except Exception:
            review = {"flags": [], "actions": []}

    return {
        "messages": [AIMessage(content="合规审核完成")],
        "compliance_flags": review.get("flags", []),
    }

# Finalizer：生成结构化病历摘要与建议，包含引用与下一步操作
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是病历与沟通撰写助手。请把接诊信息、临床建议、合规标记和证据整合成一份结构化病程记录，包含结论、推荐的下一步（含是否需线下诊疗/处方审核）和引用标题。输出文本，不要包含多余解释。"),
    ("human", "接诊：{intake}\n临床建议：{clinical}\n合规标记：{flags}\n证据：{evidence}\n请输出病程记录。")
])

def finalizer_node(state: MedState) -> Dict:
    intake = state.get("intake_struct") or {}
    clinical = state.get("clinician_note") or ""
    flags = state.get("compliance_flags", [])
    evidence = ", ".join(state.get("citations", []))
    msg = final_prompt.format_messages(intake=json.dumps(intake, ensure_ascii=False), clinical=clinical, flags=flags, evidence=evidence)
    ai = llm.invoke(msg)
    summary = ai.content

    return {
        "messages": [AIMessage(content=summary)],
        "final_summary": summary,
    }

# ------------------------------
# 路由逻辑
# ------------------------------
def after_intake(state: MedState) -> str:
    # 通常先检索证据，再临床推理
    return "researcher"

def after_research(state: MedState) -> str:
    return "clinician"

def after_clinician(state: MedState) -> str:
    return "compliance"

def after_compliance(state: MedState) -> str:
    return "finalizer"

# ------------------------------
# 构建 Graph
# ------------------------------
workflow = StateGraph(MedState)
workflow.add_node("intake", intake_node)
workflow.add_node("researcher", research_node)
workflow.add_node("clinician", clinician_node)
workflow.add_node("compliance", compliance_node)
workflow.add_node("finalizer", finalizer_node)

workflow.set_entry_point("intake")
workflow.add_edge("intake", "researcher")
workflow.add_edge("researcher", "clinician")
workflow.add_edge("clinician", "compliance")
workflow.add_edge("compliance", "finalizer")
workflow.add_edge("finalizer", END)

app = workflow.compile(checkpointer=memory)

# ------------------------------
# 运行示例（演示流）
# ------------------------------
if __name__ == "__main__":
    patient_text = (
        "患者：我发烧三天，最高38.7℃，伴干咳和乏力。没有慢性病史，目前在服用感冒药，不知道有没有药物过敏。"
    )

    config = {"configurable": {"thread_id": "med-demo-1"}}
    initial_state = {
        "messages": [HumanMessage(content=patient_text)],
        "intake_struct": None,
        "research_notes": [],
        "citations": [],
        "clinician_note": None,
        "compliance_flags": [],
        "final_summary": None,
    }

    print("=== 开始医疗工单流 ===")
    for event in app.stream(initial_state, config=config): # type: ignore
        for node, update in event.items():
            print(f"[node={node}] 更新字段: {list(update.keys())}")
    print("=== 完成 ===\n")

    final_state = app.get_state(config) # type: ignore
    summary = final_state.values.get("final_summary")
    print("=== 最终病程记录 ===")
    print(summary or "(无总结)")

    # 会话恢复示例：基于同一 thread_id 继续会话
    follow_up = "患者又问：我能吃退烧药吗？"
    print("\n=== 继续会话：患者追问 ===")
    for event in app.stream({"messages": [HumanMessage(content=follow_up)]}, config=config): # type: ignore
        pass
    s2 = app.get_state(config) # type: ignore
    print(s2.values.get("final_summary"))
