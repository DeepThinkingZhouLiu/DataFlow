from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from dataflow.agent.toolkits import (
    local_tool_for_get_categories,
    local_tool_for_sample,
)
from dataflow.dataflowagent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow.dataflowagent.state import DFState
from dataflow.dataflowagent.utils import robust_parse_json  
from dataflow import get_logger

log = get_logger()

async def data_content_classification(
    state: DFState, 
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    **kwargs,
) -> DFState:

    log.info("开始执行数据内容分类...")    
    actual_model = model_name or state.request.model
    log.info(f"使用模型: {actual_model}")
    
    sample = ''
    categories = '文学, 小说, 诗歌'

    ptg = PromptsTemplateGenerator(state.request.language)
    sys_prompt: str = ptg.render("system_prompt_for_data_content_classification")
    task_prompt: str = ptg.render(
        "task_prompt_for_data_content_classification",
        local_tool_for_sample=sample,
        local_tool_for_get_categories=categories,
    )
    messages: List[BaseMessage] = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=task_prompt),
    ]
    
    llm = ChatOpenAI(
        openai_api_base=state.request.chat_api_url,
        openai_api_key=state.request.api_key,
        model_name=actual_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    try:
        answer_msg = await llm.ainvoke(messages) 
        answer_text = answer_msg.content
    except AttributeError:  
        loop = asyncio.get_running_loop()
        answer_text = await loop.run_in_executor(None, lambda: llm.invoke(messages).content)
    except Exception as e:
        log.exception("LLM 调用失败: %s", e)
        state.classification = {"error": str(e)}
        return state
    

    try:
        parsed: Dict[str, Any] = robust_parse_json(answer_text)
    except ValueError as e:
        log.warning(f"robust_parse_json 解析失败: {e}")
        parsed = {"raw": answer_text}
    except Exception as e:
        log.warning(f"解析过程出错: {e}，保存原文")
        parsed = {"raw": answer_text}

    state.classification = parsed
    log.info("数据内容分类完成")
    return state