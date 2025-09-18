from __future__ import annotations

import asyncio
import os

from langchain_core.tools import Tool

from dataflow.dataflowagent.state import DFRequest, DFState
from dataflow.dataflowagent.agentroles.recommender import create_recommender
from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager

from dataflow.dataflowagent.toolkits.basetool.file_tools import (
    local_tool_for_sample
)
from dataflow.dataflowagent.toolkits.optool.op_tools import (
    get_operator_content,
    local_tool_for_get_purpose,
    post_process_combine_pipeline_result,
)

from dataflow.cli_funcs.paths import DataFlowPath

BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent


async def main() -> None:
    req = DFRequest(
        language="en",
        chat_api_url="http://123.129.219.111:3000/v1",
        api_key=os.getenv("DF_API_KEY", "test"),
        model="gpt-4o",
        json_file=f"{DATAFLOW_DIR}/dataflow/example/DataflowAgent/mq_test_data.jsonl",
        target = "我需要3个算子，其中不需要去重的算子！"
    )
    state = DFState(request=req)

    tm = get_tool_manager()

    # ---------- 前置工具：sample / target / operator -------------------------
    tm.register_pre_tool(
        name="sample",
        func=lambda: local_tool_for_sample(req, sample_size=3)["samples"],
        role="recommender",
    )
    tm.register_pre_tool(
        name="target",
        func=lambda: local_tool_for_get_purpose(req),
        role="recommender",
    )
    tm.register_pre_tool(
        name="operator",
        func= get_operator_content,
        role="recommender",
    )

    # ---------- 后置工具：post_process_combine_pipeline_result ---------------
    post_tool = Tool(
        name="post_process_combine_pipeline_result",
        description=(
            "根据推荐步骤，对中间产物进行组装，得到最终可执行的数据处理管线"
        ),
        func=post_process_combine_pipeline_result,
    )
    tm.register_post_tool(post_tool, role="recommender")

    # 3) 创建 Recommender 并执行 ----------------------------------------------
    # 若希望让 LLM 有机会调用后置工具，需要 use_agent=True
    recommender = create_recommender(tool_manager=tm, model_name="deepseek-v3")
    state = await recommender.execute(state, use_agent=True)

    # 4) 输出结果 -------------------------------------------------------------
    print("推荐结果：", state.recommendation)
    print("state:", state)

if __name__ == "__main__":
    asyncio.run(main())