from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import PromptedEvaluatorPrompt
import ast

@OPERATOR_REGISTRY.register()
class PromptedEvaluator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None, prompt_template: str = None, score_range: tuple = (0, 1), output_format: str = 'json'):

        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.score_name = 'PromptedScore'
        self.prompt_template = prompt_template or "Evaluate the quality of the following text on a scale from {min} to {max}. Provide only the score in JSON format: {\"score\": <number>}"
        self.score_range = score_range
        self.output_format = output_format
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于自定义提示词模板对文本质量进行评分，支持灵活定义评估维度与输出格式。通过LLM生成0-1之间的分数，适用于多场景文本质量评估。"
                "输入参数："
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口"
                "- prompt_template：自定义提示词模板，支持{min}, {max}, {text}占位符"
                "- score_range：评分范围，默认为(0, 1)"
                "- output_format：输出格式，支持'json'或'plain'"
                "- input_key：输入文本字段名"
                "输出参数："
                "- DataFrame，包含单列评分结果，列名为'PromptedScore'"
            )
        elif lang == "en":
            return (
                "Evaluate text quality based on a custom prompt template, supporting flexible definition of evaluation dimensions and output format. Generates a score between 0 and 1 using LLM, suitable for various text quality assessment scenarios."
                "Input parameters:"
                "- llm_serving: LLM serving object implementing LLMServingABC interface"
                "- prompt_template: Custom prompt template with placeholders {min}, {max}, {text}"
                "- score_range: Score range, default (0, 1)"
                "- output_format: Output format, supports 'json' or 'plain'"
                "- input_key: Field name for input text"
                "Output parameters:"
                "- DataFrame with a single column of scores named 'PromptedScore'"
            )
        else:
            return "Evaluate text quality using a customizable prompt template with LLM."

    def get_score(self, samples, input_key):
        system_prompt = "You are a helpful assistant that evaluates text quality."
        user_prompts = []
        for sample in samples:
            input_text = sample.get(input_key, '')
            prompt = self.prompt_template.format(
                min=self.score_range[0],
                max=self.score_range[1],
                text=input_text
            )
            full_prompt = system_prompt + "\n" + prompt
            user_prompts.append(full_prompt)

        responses = self.llm_serving.generate_from_input(user_inputs=user_prompts)
        scores = []

        for i, response in enumerate(responses):
            try:
                if self.output_format == 'json':
                    # Extract JSON score
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    if start != -1 and end != 0:
                        json_str = response[start:end]
                        parsed = ast.literal_eval(json_str)
                        score = parsed.get('score', None)
                        if score is not None:
                            scores.append(float(score))
                        else:
                            raise ValueError("No 'score' key in JSON")
                    else:
                        raise ValueError("No valid JSON found")
                else:
                    # Plain text extraction
                    score_str = ''.join(filter(str.isdigit, response.replace('.', '')))
                    if score_str:
                        score = float(score_str) / 100.0  # Assume 0-100 scale
                        scores.append(score)
                    else:
                        raise ValueError("No valid number found")
            except Exception as e:
                self.logger.warning(f"Failed to extract score from response {i}: {e}")
                scores.append(float('nan'))

        return scores

    def eval(self, dataframe: pd.DataFrame, input_key: str):
        samples = dataframe.to_dict(orient='records')
        self.logger.info(f"Evaluating {self.score_name}...")
        scores = self.get_score(samples, input_key)
        self.logger.info("Evaluation complete!")
        return scores

    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, self.input_key)
        score_df = pd.DataFrame(scores, columns=[self.score_name])
        dataframe = pd.concat([dataframe, score_df], axis=1)
        storage.write(dataframe)