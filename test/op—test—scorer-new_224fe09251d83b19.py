from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class MedicalEnhancementWriter(OperatorABC):
    def __init__(self, 
                 llm_serving: LLMServingABC = None,
                 audience: str = "patient",
                 tone: str = "friendly",
                 length: str = "concise"):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.audience = audience
        self.tone = tone
        self.length = length
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "医学文本优化器：使用LLM根据受众、语气及长度要求对医学文本进行润色。
"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- audience：目标受众(patient/doctor)，默认patient\n"
                "- tone：语气(friendly/formal)，默认friendly\n"
                "- length：长度(concise/detailed)，默认concise\n"
                "- input_key：待优化文本字段名\n"
                "- output_key：输出字段名，默认'enhanced_text'\n"
                "输出：\n"
                "- 添加优化后文本列的DataFrame"
            )
        elif lang == "en":
            return (
                "Medical text enhancer that rewrites medical content for a specified audience with desired tone and length.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- audience: Target audience ('patient' or 'doctor'), default 'patient'\n"
                "- tone: Writing tone ('friendly' or 'formal'), default 'friendly'\n"
                "- length: Output length ('concise' or 'detailed'), default 'concise'\n"
                "- input_key: Column containing original text\n"
                "- output_key: Column to store enhanced text, default 'enhanced_text'\n"
                "Output:\n"
                "- DataFrame with an additional column of enhanced text"
            )
        else:
            return "Rewrite medical text for specified audience."

    def build_prompt(self, raw_text):
        system_prompt = (
            f"You are a professional medical writer. Rewrite the following text for a {self.audience}. "
            f"Use a {self.tone} tone and make it {self.length}."
        )
        user_prompt = f"Original text:\n{raw_text}\n\nRewritten text:"
        return system_prompt + "\n" + user_prompt

    def enhance(self, samples, input_key):
        prompts = [self.build_prompt(sample.get(input_key, "")) for sample in samples]
        responses = self.llm_serving.generate_from_input(user_inputs=prompts)
        return [resp.strip() for resp in responses]

    def run(self, 
            storage: DataFlowStorage, 
            input_key: str, 
            output_key: str = 'enhanced_text'):
        dataframe = storage.read("dataframe")
        samples = dataframe.to_dict(orient='records')
        enhanced_texts = self.enhance(samples, input_key)
        dataframe[output_key] = enhanced_texts
        storage.write(dataframe)