import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class ClinicalPromptEnricher(OperatorABC):
    """
    Enriches raw medical prompt questions with realistic clinical context.
    """
    def __init__(self, llm_serving: LLMServingABC, system_prompt: str = None):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt or (
            "You are a senior clinician writing complex, realistic clinical scenarios. "
            "Given a basic question, enrich it by adding plausible patient details (age, gender, chief complaint, relevant history, vitals, labs, imaging) so that the question becomes more context-rich while preserving the original intent. "
            "Output ONLY the enriched question."
        )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "将基础医学问题扩充为包含更多临床细节的复杂情景问题。\n"
                "输入参数：\n"
                "- llm_serving：实现LLMServingABC接口的LLM服务对象\n"
                "- system_prompt：系统提示词，可选，默认提供医生视角提示\n"
                "- input_key：原始问题字段名，默认'question'\n"
                "- output_key：扩充后问题字段名，默认'questionCONTEXT'\n\n"
                "输出：\n"
                "- 追加了扩充问题列的DataFrame\n"
                "- 返回输出列名，供后续算子引用"
            )
        elif lang == "en":
            return (
                "Enhances basic medical questions by adding realistic clinical context.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC\n"
                "- system_prompt: Optional system prompt (default provided)\n"
                "- input_key: Field name for raw question, default 'question'\n"
                "- output_key: Field name for enriched question, default 'questionCONTEXT'\n\n"
                "Output:\n"
                "- DataFrame with an additional enriched question column\n"
                "- Returns output column name for downstream operators"
            )
        else:
            return "ClinicalPromptEnricher adds clinical context to medical questions."

    def run(self, storage: DataFlowStorage, input_key: str = "question", output_key: str = "questionCONTEXT"):
        self.logger.info("Running ClinicalPromptEnricher ...")
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loaded {len(dataframe)} rows")

        llm_inputs = []
        for _, row in dataframe.iterrows():
            q = str(row.get(input_key, '')).strip()
            if q:
                prompt = f"{self.system_prompt}\n\nBasic Question: {q}\nEnriched Question:"
                llm_inputs.append(prompt)
            else:
                llm_inputs.append("")  # keep alignment for indexing

        try:
            self.logger.info("Generating enriched questions via LLM ...")
            enriched = self.llm_serving.generate_from_input(llm_inputs)
            self.logger.info("Generation completed")
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return

        dataframe[output_key] = enriched
        storage.write(dataframe)
        return output_key


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
        api_url='https://api.chatanywhere.com.cn/v1/chat/completions',
        key_name_of_api_key = 'DF_API_KEY',
        model_name="gpt-4o",
        max_workers=100,
    )
    # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# 3. Instantiate operator
operator = ClinicalPromptEnricher(llm_serving=llm_serving, system_prompt="")

# 4. Run
operator.run(storage=storage.step())
