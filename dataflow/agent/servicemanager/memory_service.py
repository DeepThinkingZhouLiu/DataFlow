#!/usr/bin/env python3
"""
memory_service.py  ── Simple in-memory session & conversation store
Author  : Zhou Liu
License : MIT
Created : 2024-06-24

This module provides the `Memory` class, a lightweight container that keeps:

* full chat history (list of role/content dicts)
* caches of the latest user / assistant messages
* arbitrary per-session key-value data
* helper functions for pickling complex objects

It is completely thread-safe in CPython **only if** each request handler
works on its own event-loop thread; otherwise add locking by yourself.
"""

#!/usr/bin/env python3
"""
memory_service.py  ── Simple in-memory session & conversation store
Author  : Zhou Liu
License : MIT
Created : 2024-06-24
Updated : 2025-08-19  ← add token-counting support
"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import collections
import httpx
from typing import Any, Dict, List, Union

# ───────────────────────────── Memory ──────────────────────────────
class Memory:
    """
    In-memory container that keeps conversation history, arbitrary per-session
    data, **and cumulative token usage**.
    """

    def __init__(self) -> None:
        # Conversation history                     
        self.storage: Dict[str, List[Dict[str, Any]]] = {}
        self.prompt_token_stats: Dict[str, int] = collections.defaultdict(int)
        self.completion_token_stats: Dict[str, int] = collections.defaultdict(int)
        self.total_token_stats: Dict[str, int] = collections.defaultdict(int)

        # Cache only the latest user / assistant messages
        self._last_user: Dict[str, Dict[str, Any]] = {}
        self._last_assistant: Dict[str, Dict[str, Any]] = {}

        # Arbitrary key-value payload per session
        self.sessions: Dict[str, Dict[str, Any]] = {}

    # ------------------------  token helpers  -----------------------
    def add_tokens(self, session_id: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        """Accumulate *prompt_tokens* and *completion_tokens* to the running total of *session_id*."""
        self.prompt_token_stats[session_id] += int(prompt_tokens)
        self.completion_token_stats[session_id] += int(completion_tokens)
        self.total_token_stats[session_id] += int(total_tokens)

    def get_total_tokens(self, session_id: str) -> int:
        """Return cumulative token count for *session_id*; 0 if none."""
        return self.total_token_stats.get(session_id, 0)
    # ----------------------------------------------------------------

    # ----------------------  conversation I/O  ----------------------
    def get_session_id(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def add_messages(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> None:
        buf = self.storage.setdefault(session_id, [])
        for m in messages:
            buf.append({"role": m["role"], "content": m["content"]})

    def add_response(self, session_id: str, message: Dict[str, Any]) -> None:
        buf = self.storage.setdefault(session_id, [])
        buf.append({"role": message["role"], "content": message["content"]})

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.storage.get(session_id, [])

    def add_content(self, session_id: str, role: str, message: Any) -> None:
        buf = self.storage.setdefault(session_id, [])
        buf.append({"role": role, "content": str(message)})

    def get_last_messages(
        self, session_id: str, n: int = 2
    ) -> List[Dict[str, Any]]:
        history = self.get_history(session_id)
        return history[-n:] if len(history) >= n else history

    def add_last_user(self, session_id: str, message: Dict[str, Any]) -> None:
        self._last_user[session_id] = message

    def add_last_assistant(
        self, session_id: str, message: Dict[str, Any]
    ) -> None:
        self._last_assistant[session_id] = message

    def get_last_user(self, session_id: str) -> Dict[str, Any]:
        return self._last_user.get(session_id, {})

    def get_last_assistant(self, session_id: str) -> Dict[str, Any]:
        return self._last_assistant.get(session_id, {})

    # ----------------------  arbitrary payload  ---------------------
    def set_session_data(self, session_id: str, key: str, value: Any) -> None:
        self.sessions.setdefault(session_id, {})[key] = value

    def get_session_data(
        self, session_id: str, key: str, default: Any = None
    ) -> Any:
        return self.sessions.get(session_id, {}).get(key, default)

    def save_object(self, session_id: str, key: str, obj: Any) -> None:
        data = pickle.dumps(obj)
        self.set_session_data(session_id, key, data)

    def load_object(
        self, session_id: str, key: str, default: Any = None
    ) -> Any:
        data = self.get_session_data(session_id, key)
        if data is None:
            return default
        try:
            return pickle.loads(data)
        except Exception:
            return default

    def append_session_list(
        self, session_id: str, key: str, item: Any
    ) -> None:
        buf = self.get_session_data(session_id, key) or []
        buf.append(item)
        self.set_session_data(session_id, key, buf)

    # ---------------------------  wipe  ------------------------------
    def clear_history(self, session_id: str) -> None:
        self.storage.pop(session_id, None)
        self._last_user.pop(session_id, None)
        self._last_assistant.pop(session_id, None)

    def clear_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self.prompt_token_stats.pop(session_id, None)
        self.completion_token_stats.pop(session_id, None)
        self.total_token_stats.pop(session_id, None)

    def reset(self, session_id: str) -> None:
        self.clear_history(session_id)
        self.clear_session(session_id)
        
    # ────────────────── 1) 汇总（覆盖写）──────────────────
    def dump_token_summary(
        self,
        filepath: Union[str, Path] = "token_summary.json"
    ) -> None:
        """
        把 *累计* token 数写入 JSON（覆盖写）。
        结构：
        {
            "<session_id>": {
                "prompt_tokens": 1234,
                "completion_tokens": 1234,
                "total_tokens": 2468,
                "updated_at": "2025-08-19T13:45:12"
            },
            ...
        }
        """
        payload = {
            sid: {
                "prompt_tokens": self.prompt_token_stats.get(sid, 0),
                "completion_tokens": self.completion_token_stats.get(sid, 0),
                "total_tokens": self.total_token_stats.get(sid, 0),
                "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
            for sid in self.total_token_stats.keys()
        }
        Path(filepath).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )

    def load_token_summary(
        self,
        filepath: Union[str, Path] = "token_summary.json"
    ) -> None:
        """启动时读取汇总文件（若不存在就跳过）。"""
        try:
            data = json.loads(Path(filepath).read_text())
            for sid, info in data.items():
                self.prompt_token_stats[sid] = int(info.get("prompt_token_stats", 0))
                self.completion_token_stats[sid] = int(info.get("completion_tokens", 0))
                self.total_token_stats[sid] = int(info.get("total_tokens", 0))
        except FileNotFoundError:
            pass
        except Exception as err:
            print(f"[Memory] load_token_summary() failed: {err}")


    # ────────────────── 2) 明细（追加写）──────────────────
    def append_token_event(
        self,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        filepath: Union[str, Path] = "token_events.jsonl",
    ) -> None:
        """
        以 **JSON Lines** 格式记录每次 API 调用的 token 使用情况。
        每次调用都会在日志文件中追加一行新的 JSON 数据，不会覆盖已有内容。
        日志格式示例:
        {"session_id": "some_session_id", "tokens": {"prompt": 100, "completion": 50, "total": 150}, "cum_tokens": {"prompt": 300, "completion": 200, "total": 500}, "ts": "2025-09-02T03:06:59Z"}
        """
        record = {
            "session_id": session_id,
            "tokens": {  # 本次请求消耗的 token
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens,
            },
            "cum_tokens": {  # 该 session 累计消耗的 token
                "prompt": self.prompt_token_stats[session_id],
                "completion": self.completion_token_stats[session_id],
                "total": self.total_token_stats[session_id],
            },
            "ts": datetime.utcnow().isoformat(timespec="seconds"),
        }
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add_tokens_and_persist(
        self,
        session_id: str,
        prompt_tokens: int, 
        completion_tokens: int, 
        total_tokens: int,
        summary_path: Union[str, Path] = "token_summary.json",
        event_path: Union[str, Path] = "token_events.jsonl",
    ) -> None:
        """
        • 累加 n_tokens  
        • 立即把汇总覆盖写到 *summary_path*  
        • 立即把本次事件追加到 *event_path*
        """
        self.add_tokens(session_id, prompt_tokens, completion_tokens, total_tokens)
        self.dump_token_summary(summary_path)
        self.append_token_event(session_id, prompt_tokens, completion_tokens, total_tokens, event_path)

# ─────────────────────────── MemoryClient ───────────────────────────
class MemoryClient:
    def __init__(self, memory: Memory):
        self.memory = memory
    async def post(
        self,
        url: str,
        headers: dict,
        json_data: dict,
        session_key: str
    ) -> str:
        session_id = self.memory.get_session_id(session_key)

        if "messages" in json_data:
            self.memory.add_messages(session_id, json_data["messages"])

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                headers=headers,
                json=json_data,
                timeout=360.0,
            )
            resp.raise_for_status()
            result = resp.json()

        choice = result.get("choices", [{}])[0].get("message")
        if choice:
            self.memory.add_response(session_id, choice)

        prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
        completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
        total_tokens = result.get("usage", {}).get("total_tokens", 0)
        self.memory.add_tokens_and_persist(session_id, prompt_tokens, completion_tokens, total_tokens)
        return choice["content"] if choice and "content" in choice else ""