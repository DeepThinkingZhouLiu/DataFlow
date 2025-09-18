# dataflow/dataflowagent/agentroles/recommender.py
from __future__ import annotations

from typing import Any, Dict, Optional, List

from dataflow.dataflowagent.agentroles.base_agent import BaseAgent
from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager

from dataflow import get_logger
log = get_logger()


class DataPipelineRecommender(BaseAgent):
    """
    根据样本、目标意图和算子库，生成数据处理管线推荐。
    前置工具:
        - sample   : local_tool_for_sample
        - target   : local_tool_for_get_purpose
        - operator : get_operator_content_map_from_all_operators
    后置工具:
        - post_process_combine_pipeline_result  (LangChain Tool)
    """
    @property
    def role_name(self) -> str:
        return "recommender"

    @property
    def system_prompt_template_name(self) -> str:
        # 你可以在 promptstemplates/resources/template.json 中添加对应模板
        return "system_prompt_for_data_pipeline_recommendation"

    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_data_pipeline_recommendation"

    # --- 向任务提示词中注入变量 --------------------------------------------
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        将前置工具结果映射到 prompt 变量名。
        这些变量名要与模板文件里的占位符一致。
        """
        return {
            "sample": pre_tool_results.get("sample", ""),
            "target": pre_tool_results.get("target", ""),
            "operator": pre_tool_results.get("operator", "[]"),
        }

    # --- 默认前置工具结果（兜底）-------------------------------------------
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        return {
            "sample": "",
            "target": "",
            "operator": "[]",
        }

    # --- 将结果写回 DFState -------------------------------------------------
    def update_state_result(
        self,
        state: DFState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        # 对外暴露属性名为 recommendation
        state.recommendation = result
        super().update_state_result(state, result, pre_tool_results)

async def data_pipeline_recommendation(
    state: DFState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    use_agent: bool = False,
    **kwargs,
) -> DFState:
    recommender = DataPipelineRecommender(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await recommender.execute(state, use_agent=use_agent, **kwargs)


def create_recommender(
    tool_manager: Optional[ToolManager] = None,
    **kwargs,
) -> DataPipelineRecommender:
    return DataPipelineRecommender(tool_manager=tool_manager, **kwargs)