from .api_llm_serving_request import APILLMServing_request


__all__ = [
    "APILLMServing_request",
]


def _optional_import(module_name: str, export_names: list[str]) -> None:
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=export_names)
    except ImportError:
        return

    for export_name in export_names:
        globals()[export_name] = getattr(module, export_name)
        __all__.append(export_name)


_optional_import("local_model_llm_serving", ["LocalModelLLMServing_vllm", "LocalModelLLMServing_sglang"])
_optional_import("localhost_llm_api_serving", ["LocalHostLLMAPIServing_vllm"])
_optional_import("localmodel_lalm_serving", ["LocalModelLALMServing_vllm"])
_optional_import("api_vlm_serving_openai", ["APIVLMServing_openai"])
_optional_import("google_api_serving", ["PerspectiveAPIServing"])
_optional_import("lite_llm_serving", ["LiteLLMServing"])
_optional_import("LocalSentenceLLMServing", ["LocalEmbeddingServing"])
_optional_import("light_rag_serving", ["LightRAGServing"])
_optional_import("api_google_vertexai_serving", ["APIGoogleVertexAIServing"])
_optional_import("local_model_vlm_serving", ["LocalVLMServing_vllm"])
