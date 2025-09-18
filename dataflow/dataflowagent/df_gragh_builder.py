from __future__ import annotations

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from dataflow.dataflowagent.agentroles.classifier import DataContentClassifier
from dataflow.dataflowagent.state import DFState
from dataflow import get_logger

log = get_logger()


def build_classification_graph(state: DFState) -> StateGraph:
    """
    根据前一步 classifier.execute(..., use_agent=True) 产生的 state，
    构建 LangGraph 图。
    """
    # 从 state.temp_data 里取出实例与前置工具结果
    classifier: DataContentClassifier = state.temp_data["classifier_instance"]
    pre_tool_results = state.temp_data["pre_tool_results"]

    # 生成助手节点函数（不再捕获外层 state，而用运行时 graph_state）
    def assistant_node(graph_state: DFState):
        """
        graph_state: LangGraph 运行时传入的 DFState，
        里面会逐步累积 messages / classification 等字段。
        """
        messages = getattr(graph_state, "messages", [])

        # 第一次进入还没有 messages，需要构建
        if not messages:
            messages = classifier.build_messages(graph_state, pre_tool_results)

        # 调 LLM，可能产生 tool_calls
        return classifier.process_with_llm_for_graph(messages, graph_state)

    builder = StateGraph(DFState)

    # 1) LLM 决策节点
    builder.add_node("assistant", assistant_node)

    post_tools = classifier.get_post_tools()
    if post_tools:
        # 2) 工具执行节点
        builder.add_node("tools", ToolNode(post_tools))

        # assistant → tools  (条件跳转)
        builder.add_conditional_edges("assistant", tools_condition)

        # tools → assistant  (执行完回到 LLM)
        builder.add_edge("tools", "assistant")

    builder.set_entry_point("assistant")
    return builder.compile()