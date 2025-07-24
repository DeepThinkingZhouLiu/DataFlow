from __future__ import annotations

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd


@OPERATOR_REGISTRY.register()
class DomainClassifier(OperatorABC):
    def __init__(
        self,
        llm_serving: LLMServingABC | None = None,
        domains: list[str] | None = None,
    ):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")
        self.llm_serving = llm_serving
        self.domains = domains or [
            "technology",
            "science",
            "health",
            "finance",
            "sports",
            "entertainment",
            "politics",
            "education",
            "travel",
            "others",
        ]
        self.score_name = "Domain"
        self.logger.info(f"{self.__class__.__name__} initialized.")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用LLM对文本进行领域分类，在给定领域列表内返回最符合的领域名称，并在数据中新增“domain”列。\n"
                "输入参数：\n"
                "- llm_serving：LLM 服务对象，需实现 LLMServingABC 接口\n"
                "- domains：领域名称列表，默认为常见十个领域\n"
                "- input_key：待分类文本字段名，默认为'question'\n"
                "输出：在原 DataFrame 中新增一列 'domain'"
            )
        return (
            "Classify text into one of the given domains using an LLM and add a 'domain' column to the DataFrame.\n"
            "Input Parameters:\n"
            "- llm_serving: LLM serving object implementing LLMServingABC\n"
            "- domains: List of domain names, defaults to ten common domains\n"
            "- input_key: Column name that contains the text to classify, default 'question'\n"
            "Output: Original DataFrame with an added 'domain' column"
        )

    def _build_prompts(self, texts: list[str]):
        system_prompt = (
            "You are a domain classification assistant. "
            f"Possible domains: {', '.join(self.domains)}. "
            "Given a text, respond with exactly one domain name from the list without extra words."
        )
        return [f"{system_prompt}\nText:\n{txt}\nDomain:" for txt in texts]

    def _parse_response(self, resp: str):
        first_line = resp.strip().split("\n")[0].strip().lower()
        for d in self.domains:
            if d.lower() in first_line:
                return d
        return "unknown"

    def eval(self, dataframe: pd.DataFrame, input_key: str):
        if input_key not in dataframe.columns:
            raise KeyError(
                f"Input key '{input_key}' not found in DataFrame columns: {dataframe.columns.tolist()}"
            )
        texts = dataframe[input_key].astype(str).tolist()
        prompts = self._build_prompts(texts)
        self.logger.info(f"Classifying {len(prompts)} samples for domain...")
        responses = self.llm_serving.generate_from_input(user_inputs=prompts)
        domains_pred = [self._parse_response(r) for r in responses]
        self.logger.info("Domain classification complete!")
        return domains_pred

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "question",
        output_key: str = "domain",
    ):
        dataframe = storage.read("dataframe")
        preds = self.eval(dataframe, input_key)
        dataframe[output_key] = preds
        storage.write(dataframe)


# ======== Auto-generated runner ========
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
from dataflow.core import LLMServingABC

if __name__ == "__main__":
    # 1. FileStorage
    storage = FileStorage(
        first_entry_file_name="/mnt/h_h_public/lh/lz/DataFlow/dataflow/example/DataflowAgent/mq_test_data.jsonl",
        cache_path="./cache_local",
        file_name_prefix="dataflow_cache_step",
        cache_type="jsonl",
    )

    # 2. LLM-Serving
    # -------- LLM Serving (Remote) --------
    llm_serving = APILLMServing_request(
        api_url="http://123.129.219.111:3000/v1/chat/completions",
        key_name_of_api_key = 'DF_API_KEY',
        model_name="gpt-4o",
        max_workers=100,
    )
    # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# 3. Instantiate operator
operator = DomainClassifier(llm_serving=llm_serving, domains="")

# 4. Run
operator.run(storage=storage.step())
