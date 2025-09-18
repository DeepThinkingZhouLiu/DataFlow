"""
prompts_template.py ── Prompt Template Manager
Author  : Zhou Liu
License : MIT
Updated : 2025-09-17

全部模板均从 Python 模块动态加载，不再读取任何 JSON 资源文件。
"""

from __future__ import annotations

import importlib, inspect, re
from string import Formatter
from typing import Any, Dict, Sequence


class PromptsTemplateGenerator:
    ANSWER_SUFFIX = ".(Answer in {lang}!!!)"

    # ---------- Singleton ----------
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    # ---------- Init ----------
    def __init__(
        self,
        output_language: str,
        *,
        python_modules: Sequence[str] | None = None,
    ) -> None:
        """
        output_language : 模型最终回复语言
        python_modules  : 需扫描的模块名列表（支持多个）
                          缺省为 ["prompts_repo"]，若不存在需显式传参
        """
        self.output_language = output_language
        self.templates: Dict[str, str] = {}
        self.json_form_templates: Dict[str, str] = {}
        self.code_debug_templates: Dict[str, str] = {}
        self.operator_templates: Dict[str, Dict] = {}

        self._load_python_templates(
            python_modules or ["prompts_repo"]
        )

    # ---------- Safe formatter ----------
    @staticmethod
    def _safe_format(tpl: str, **kwargs) -> str:
        class _Missing(dict):
            def __missing__(self, k):   # 保留占位符
                return "{" + k + "}"
        try:
            return Formatter().vformat(tpl, [], _Missing(**kwargs))
        except Exception:                 # 极端情况下回退
            for k in re.findall(r"{(.*?)}", tpl):
                tpl = tpl.replace("{" + k + "}", str(kwargs.get(k, "{"+k+"}")))
            return tpl

    # ---------- Loader ----------
    def _load_python_templates(self, modules: Sequence[str]) -> None:
        """
        扫描给定模块中的所有 class / 顶层变量，把字符串模板或 operator dict
        归档到内部字典。
        """
        for mod_name in modules:
            # mod = importlib.import_module(mod_name)
            mod = importlib.import_module('.prompts_repo', package=__package__)

            # 1. class 内部的属性
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod.__name__:
                    continue
                self._collect_from_mapping(vars(cls))

            # 2. 顶层变量
            self._collect_from_mapping(vars(mod))

    # ---------- Collect helper ----------
    def _collect_from_mapping(self, mapping: dict) -> None:
        for attr, value in mapping.items():
            if attr.startswith("_"):
                continue
            # ---- operator dict ----
            if attr == "operator_templates" and isinstance(value, dict):
                self.operator_templates.update(value)
                continue
            # ---- 字符串模板 ----
            if not isinstance(value, str):
                continue
            if attr.startswith("system_prompt_for_") or attr.startswith("task_prompt_for_"):
                self.templates[attr] = value
            elif attr.startswith("json_form_template_for_"):
                key = attr.replace("json_form_template_for_", "")
                self.json_form_templates[key] = value
            elif attr.startswith("code_debug_template_for_"):
                key = attr.replace("code_debug_template_for_", "")
                self.code_debug_templates[key] = value
            else:  
                self.templates[attr] = value

    # ---------- Renderers ----------
    def render(self, template_name: str, *, add_suffix: bool = False, **kwargs) -> str:
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        txt = self._safe_format(self.templates[template_name], **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=self.output_language) if add_suffix else "")

    def render_json_form(self, template_name: str, *, add_suffix=False, **kwargs) -> str:
        if template_name not in self.json_form_templates:
            raise ValueError(f"JSON-form template '{template_name}' not found")
        txt = self._safe_format(self.json_form_templates[template_name], **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=self.output_language) if add_suffix else "")

    def render_code_debug(self, template_name: str, *, add_suffix=False, **kwargs) -> str:
        if template_name not in self.code_debug_templates:
            raise ValueError(f"Code-debug template '{template_name}' not found")
        txt = self._safe_format(self.code_debug_templates[template_name], **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=self.output_language) if add_suffix else "")

    def render_operator_prompt(
        self,
        operator_name: str,
        prompt_type: str = "task",
        language: str | None = None,
        *,
        add_suffix: bool = False,
        **kwargs,
    ) -> str:
        lang = language or self.output_language
        op = self.operator_templates.get(operator_name)
        if not op:
            raise ValueError(f"Operator '{operator_name}' not found")
        try:
            tpl = op["prompts"][lang][prompt_type]
        except KeyError:
            raise KeyError(
                f"Missing prompt (operator={operator_name}, lang={lang}, type={prompt_type})"
            )
        txt = self._safe_format(tpl, **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=lang) if add_suffix else "")

    # ---------- Runtime add ----------
    def add_sys_template(self, name: str, template: str) -> None:
        self.templates[f"system_prompt_for_{name}"] = template

    def add_task_template(self, name: str, template: str) -> None:
        self.templates[f"task_prompt_for_{name}"] = template

    def add_json_form_template(self, task_name: str, template: str | dict) -> None:
        if isinstance(template, dict):
            import json
            template = json.dumps(template, ensure_ascii=False, indent=2)
        self.json_form_templates[task_name] = template



if __name__ == "__main__":
    ptg = PromptsTemplateGenerator(
        output_language="zh",
        python_modules=["prompts_repo"],  
    )

    print(ptg.render("system_prompt_for_data_content_classification"))
    print(
        ptg.render(
            "task_prompt_for_data_content_classification",
            local_tool_for_sample="《红楼梦》…",
            local_tool_for_get_categories="文学, 小说, 诗歌",
            add_suffix=True,
        )
    )