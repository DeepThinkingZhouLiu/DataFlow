from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DFRequest:
    language: str = "en"
    chat_api_url: str = "http://123.129.219.111:3000/v1"  # ！！注意：不带 /chat/completions
    api_key: str = "sk-h6nyfmxUx70YUMBQNwaayrmXda62L7rxvytMNshxACWVzJXe"
    model: str = "gpt-3.5-turbo"


@dataclass
class DFState:
    request: DFRequest
    classification: Dict[str, Any] = field(default_factory=dict)