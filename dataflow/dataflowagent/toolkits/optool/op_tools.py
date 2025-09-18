from __future__ import annotations
import asyncio
import inspect
import sys
import os
from pydantic import BaseModel
import httpx
import json
import uuid
from typing import List, Dict, Sequence, Any, Union, Optional, Iterable, Mapping, Set, Callable
from pathlib import Path
 
from functools import lru_cache
import yaml
# from clickhouse_connect import get_client
import subprocess
from collections import defaultdict, deque
from dataflow.utils.storage import FileStorage
from dataflow import get_logger
logger = get_logger()
from dataflow.cli_funcs.paths import DataFlowPath
from dataflow.dataflowagent.storage.storage_service import SampleFileStorage
from dataflow.dataflowagent.state import DFState,DFRequest

def local_tool_for_get_purpose(req: DFRequest) -> str:
    return req.target or ""


def get_operator_content() -> str:
    pass


def post_process_combine_pipeline_result() -> str:
    pass


if __name__ == "__main__":
    pass