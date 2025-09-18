from dataclasses import dataclass, field
import os
from typing import Any, Dict
from dataflow.cli_funcs.paths import DataFlowPath
BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent

@dataclass
class DFRequest:
    language: str = "en"
    chat_api_url: str = "http://123.129.219.111:3000/v1"
    api_key: str = os.getenv("DF_API_KEY", "test")
    model: str = "gpt-4o"
    json_file: str = f"{DATAFLOW_DIR}/dataflow/example/DataflowAgent/mq_test_data.jsonl" 

@dataclass
class DFState:
    request: DFRequest
    agent_results: Dict[str, Any] = field(default_factory=dict)
    classification: Dict[str, Any] = field(default_factory=dict)