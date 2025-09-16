from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import MetaPrompt  # Updated import to match available class
import ast

@OPERATOR_REGISTRY.register()
class DialogueQualityEvaluator(OperatorABC):
    def __init__(self, 
                 llm_serving: LLMServingABC = None,
                 criteria: list[dict] = None,
                 weight_config: dict = None
                ):


        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.criteria = criteria or []
        self.weight_config = weight_config or {}

        # Validate criteria
        for item in self.criteria:
            if 'criterion_name' not in item or 'description' not in item or 'example_list' not in item:
                raise ValueError('Invalid criterion format. Refer to the docstring for the correct format.')
            for example in item['example_list']:
                if 'dialogue' not in example or 'score' not in example:
                    raise ValueError('Invalid example format. Refer to the docstring for the correct format.')
            if 'weight' not in item:
                item['weight'] = 1.0

        # Apply weight override
        for criterion in self.criteria:
            name = criterion['criterion_name']
            if name in self.weight_config:
                criterion['weight'] = self.weight_config[name]

        self.output_columns = [item['criterion_name'] for item in self.criteria]
        self.score_name = 'DialogueQualityScore'
        self.prompt = MetaPrompt(dimensions=criteria)  # Use MetaPrompt instead of MultiTurnDialoguePrompt

        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "DialogueQualityEvaluator：基于多轮对话分析、角色识别、上下文理解、异常检测和可定制评估标准，对对话质量进行综合评估的算子。支持自定义评分维度与权重，输出多维度质量得分。\n"
                "功能：对每条对话进行多维度质量评估，支持角色识别、上下文连贯性、安全性、教育性、逻辑一致性等。\n"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- criteria：评估标准列表，每个标准包含criterion_name、description、example_list和可选weight\n"
                "- weight_config：可选的权重配置字典，用于覆盖默认权重\n"
                "- input_key：输入对话字段名（默认：'dialogue'）\n"
                "输出参数：\n"
                "- 包含各评估维度得分的DataFrame，列名为：{criterion_names}，每行对应一条对话的多维评分"
            )
        elif lang == "en":
            return (
                "DialogueQualityEvaluator: A comprehensive dialogue quality evaluation operator supporting multi-turn analysis, role identification, context understanding, anomaly detection, and customizable evaluation criteria.\n"
                "Features: Evaluates dialogue quality across multiple dimensions including coherence, relevance, safety, role consistency, educational value, and logical flow. Supports dynamic weight configuration.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- criteria: List of evaluation criteria, each with criterion_name, description, example_list, and optional weight\n"
                "- weight_config: Optional dictionary to override default weights\n"
                "- input_key: Column name containing input dialogue (default: 'dialogue')\n"
                "Output Parameters:\n"
                "- DataFrame with scores for each criterion, columns: {criterion_names}, one row per dialogue"
            )
        else:
            return "Comprehensive dialogue quality evaluation with customizable criteria and multi-turn analysis."

    def get_score(self, samples, input_key):
        system_prompt = self.prompt.build_system_prompt()
        user_prompts = []
        for sample in samples:
            dialogue_text = sample.get(input_key, '')
            user_prompt = self.prompt.build_user_prompt(dialogue_text)
            full_prompt = system_prompt + "\n" + user_prompt
            user_prompts.append(full_prompt)

        responses = self.llm_serving.generate_from_input(user_inputs=user_prompts)
        scores = []

        for i, response in enumerate(responses):
            try:
                lines = response.strip().split("\n")
                last_line = lines[-1].strip()
                parsed_scores = ast.literal_eval(last_line)
                if isinstance(parsed_scores, list) and len(parsed_scores) == len(self.criteria):
                    scores.append(parsed_scores)
                else:
                    raise ValueError("Score format invalid")
            except Exception as e:
                self.logger.warning(f"Failed to extract score from response {i}: {e}")
                scores.append([float('nan')] * len(self.criteria))

        return scores

    def eval(self, dataframe: pd.DataFrame, input_key: str):
        samples = dataframe.to_dict(orient='records')
        self.logger.info(f"Evaluating {self.score_name}...")
        scores = self.get_score(samples, input_key)
        self.logger.info("Evaluation complete!")
        return scores

    def run(self, storage: DataFlowStorage, input_key: str = "dialogue"):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, self.input_key)

        # Create score DataFrame with criterion names
        score_df = pd.DataFrame(scores, columns=self.output_columns)
        dataframe = pd.concat([dataframe, score_df], axis=1)
        storage.write(dataframe)


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
operator = DialogueQualityEvaluator(llm_serving=llm_serving, criteria="", weight_config="")

# 4. Run
operator.run(storage=storage.step())
