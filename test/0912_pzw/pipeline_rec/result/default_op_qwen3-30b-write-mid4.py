import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import json

@OPERATOR_REGISTRY.register()
class NERExtractor(OperatorABC):
    
    def __init__(self, llm_serving: LLMServingABC, prompt_template=None):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = prompt_template
        self.json_failures = 0
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "NERExtractor 用于从文本中提取命名实体（如人名、组织、地点等），基于大语言模型实现结构化信息抽取。"
                "算子根据提示模板（prompt_template）构造输入，调用 LLM 生成实体识别结果，并将 JSON 格式输出写回指定列。"
                "\n输入参数："
                "- llm_serving：LLM 服务对象，需实现 LLMServingABC 接口"
                "- prompt_template：提示词模板对象，用于构造模型输入"
                "- content_key：包含待处理文本的列名（默认 'text'）"
                "- output_key：写回实体识别结果的列名（默认 'entities'）"
                "\n输出参数："
                "- DataFrame，其中 output_key 列为模型返回并经 JSON 解析后的实体列表"
                "- 返回 output_key，供后续算子引用"
                "\n备注："
                "- 模型输出会尝试解析为 JSON；若解析失败，将返回 [] 并记录失败次数。"
            )
        elif lang == "en":
            return (
                "NERExtractor extracts named entities (e.g., person, organization, location) from text using LLM-based inference for structured information extraction."
                "It constructs input using a prompt_template, calls the LLM to generate entity recognition results, and writes the JSON-formatted output back to the specified column."
                "\nInput Parameters:"
                "- llm_serving: LLM serving object implementing LLMServingABC interface"
                "- prompt_template: Prompt template object used to build model input"
                "- content_key: Column name containing text to process (default 'text')"
                "- output_key: Column name to store extracted entities (default 'entities')"
                "\nOutput Parameters:"
                "- DataFrame with output_key column containing parsed entity lists from model output"
                "- Returns output_key for downstream operators"
                "\nNotes:"
                "- Model output is attempted to be parsed as JSON; failures return [] and are counted."
            )
        else:
            return "NERExtractor extracts named entities from text using LLM-based inference."
    
    def _strip_code_fence(self, s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        return s.strip()
    
    def _safe_json_load(self, item):
        try:
            if isinstance(item, (list, dict)):
                return item
            if item is None:
                return []
            if not isinstance(item, str):
                item = str(item)
            s = item.strip()
            if not s:
                return []
            s = self._strip_code_fence(s)
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                s = s[1:-1].strip()
            s = re.sub(r"^\s*(?:json|JSON)\s*", "", s)
            m = re.search(r"[\[\{]", s)
            if m:
                s = s[m.start():].strip()
            last_bracket = max(s.rfind("]"), s.rfind("}"))
            if last_bracket != -1:
                s = s[:last_bracket + 1].strip()
            obj = json.loads(s)
            if isinstance(obj, str):
                try:
                    obj2 = json.loads(obj)
                    return obj2
                except json.JSONDecodeError:
                    return obj
            return obj
        except Exception as e:
            self.json_failures += 1
            preview = "" 
            try:
                preview = (s if len(s) <= 200 else s[:200] + "...").replace("\n", "\\n")
            except Exception:
                preview = "<unavailable>"
            self.logger.warning(f"[safe_json_load] 解析失败，第{self.json_failures}次；错误：{type(e).__name__}: {e}；预览: {preview}")
            return []
    
    def run(self, storage: DataFlowStorage, content_key: str = "text", output_key: str = "entities"):
        self.logger.info("Running NERExtractor...")
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")
        
        llm_inputs = []
        for index, row in dataframe.iterrows():
            text = row.get(content_key, '')
            llm_input = self.prompt_template.build_prompt(text)
            llm_inputs.append(llm_input)
        
        try:
            self.logger.info("Generating NER results using the model...")
            generated_outputs = self.llm_serving.generate_from_input(llm_inputs)
            self.logger.info("NER generation completed.")
        except Exception as e:
            self.logger.error(f"Error during NER generation: {e}")
            return
        
        parsed_outputs = [self._safe_json_load(item) for item in generated_outputs]
        dataframe[output_key] = parsed_outputs
        
        output_file = storage.write(dataframe)
        return output_key


# ======== Auto-generated runner ========
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
from dataflow.core import LLMServingABC

if __name__ == "__main__":
    # 1. FileStorage
    storage = FileStorage(
        first_entry_file_name="/mnt/public/data/lh/pzw/DataFlow/dataflow/example/ReasoningPipeline/pipeline_math_short.json",
        cache_path="./cache_local",
        file_name_prefix="dataflow_cache_step",
        cache_type="jsonl",
    )

    # 2. LLM-Serving
    # -------- LLM Serving (Local) --------
    llm_serving = LocalModelLLMServing_vllm(
        hf_model_name_or_path="/mnt/public/model/huggingface/Qwen3-30B-A3B-Instruct-2507",
        vllm_tensor_parallel_size=1,
        vllm_max_tokens=8192,
        hf_local_dir="local",
    )

# 3. Instantiate operator
operator = NERExtractor(llm_serving=llm_serving, prompt_template="")

# 4. Run
operator.run(storage=storage.step())
