from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any
from langgraph.graph import StateGraph, START, END

from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.agentroles.classifier import data_content_classification
from dataflow import get_logger

log = get_logger()


@dataclass
class NodeConfig:
    """节点配置"""
    model_name: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    # 可以添加更多参数
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}

def create_simple_graph(node_configs: Dict[str, NodeConfig] = None):
    """
    创建简单的分类图
    
    Args:
        node_configs: 节点配置字典，格式: {"节点名": NodeConfig(...)}
    """
    
    if node_configs is None:
        node_configs = {}
    
    # 包装分类节点
    async def classify_node(state: DFState) -> DFState:
        config = node_configs.get("classify", NodeConfig())
        return await data_content_classification(
            state, 
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    # 创建图
    builder = StateGraph(DFState)
    builder.add_node("classify", classify_node)
    
    # 添加边
    builder.add_edge(START, "classify")
    builder.add_edge("classify", END)
    
    log.info(f"图创建完成，节点配置: {list(node_configs.keys())}")
    return builder.compile()

def create_multi_node_graph(node_configs: Dict[str, NodeConfig] = None):
    """
    创建多节点图（预留，方便后续扩展）
    """
    if node_configs is None:
        node_configs = {}
    
    # 包装分类节点
    async def classify_node(state: DFState) -> DFState:
        config = node_configs.get("classify", NodeConfig())
        log.info(f"分类节点使用模型: {config.model_name or '默认'}")
        return await data_content_classification(
            state, 
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    # 包装预处理节点
    async def preprocess_node(state: DFState) -> DFState:
        config = node_configs.get("preprocess", NodeConfig())
        log.info(f"预处理节点配置: 温度={config.temperature}")
        # 这里可以添加预处理逻辑
        return state
    
    # 包装验证节点
    async def validate_node(state: DFState) -> DFState:
        config = node_configs.get("validate", NodeConfig())
        log.info("验证节点执行中...")
        
        if not state.classification or "error" in state.classification:
            state.classification = {"status": "failed", "error": "No valid classification"}
        else:
            state.classification["status"] = "validated"
        return state
    
    # 创建图
    builder = StateGraph(DFState)
    builder.add_node("preprocess", preprocess_node)
    builder.add_node("classify", classify_node)
    builder.add_node("validate", validate_node)
    
    # 添加边
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "classify")
    builder.add_edge("classify", "validate")
    builder.add_edge("validate", END)
    
    log.info(f"多节点图创建完成，配置: {list(node_configs.keys())}")
    return builder.compile()

# 测试用
if __name__ == "__main__":
    import json
    from dataclasses import dataclass, field
    from typing import Dict, Any
    
    @dataclass
    class DFRequest:
        language: str
        chat_api_url: str
        api_key: str
        model: str
    
    @dataclass 
    class DFState:
        request: DFRequest
        classification: Dict[str, Any] = field(default_factory=dict)
    
    async def test_simple():
        """测试简单图"""
        print("\n=== 测试简单图 ===")
        
        req = DFRequest(
            language="zh",
            chat_api_url="http://123.129.219.111:3000/v1",
            api_key="sk-h6nyfmxUx70YUMBQNwaayrmXda62L7rxvytMNshxACWVzJXe", 
            model="gpt-3.5-turbo"  # 默认模型
        )
        
        init_state = DFState(request=req)
        
        # 配置分类节点使用特定模型
        configs = {
            "classify": NodeConfig(
                model_name="gpt-4o",
                temperature=0.1,
                max_tokens=800
            )
        }
        
        graph = create_simple_graph(node_configs=configs)
        
        print("开始执行简单图...")
        result = await graph.ainvoke(init_state)
        
        if isinstance(result, dict):
            print("结果:", json.dumps(result.get("classification", {}), indent=2, ensure_ascii=False))
        else:
            print("结果:", json.dumps(result.classification, indent=2, ensure_ascii=False))
    
    async def test_multi():
        """测试多节点图"""
        print("\n=== 测试多节点图 ===")
        
        req = DFRequest(
            language="zh",
            chat_api_url="http://123.129.219.111:3000/v1",
            api_key="sk-h6nyfmxUx70YUMBQNwaayrmXda62L7rxvytMNshxACWVzJXe", 
            model="gpt-3.5-turbo"
        )
        
        init_state = DFState(request=req)
        
        # 为不同节点配置不同参数
        configs = {
            "preprocess": NodeConfig(
                temperature=0.0,
                extra_params={"preprocessing": True}
            ),
            "classify": NodeConfig(
                model_name="gpt-4o",
                temperature=0.2,
                max_tokens=1024
            ),
            "validate": NodeConfig(
                temperature=0.1,
                extra_params={"strict": True}
            )
        }
        
        graph = create_multi_node_graph(node_configs=configs)
        
        print("开始执行多节点图...")
        result = await graph.ainvoke(init_state)
        
        if isinstance(result, dict):
            print("结果:", json.dumps(result.get("classification", {}), indent=2, ensure_ascii=False))
        else:
            print("结果:", json.dumps(result.classification, indent=2, ensure_ascii=False))
    
    async def main():
        await test_simple()
        await test_multi()
    
    asyncio.run(main())