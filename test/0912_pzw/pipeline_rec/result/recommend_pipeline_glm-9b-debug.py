from dataflow.pipeline import PipelineABC
import pytest
from dataflow.operators.agentic_rag.filter.content_chooser import ContentChooser
from dataflow.operators.agentic_rag.eval.f1_scorer import F1Scorer
from dataflow.operators.agentic_rag.generate.atomic_task_generator import AtomicTaskGenerator
from dataflow.operators.agentic_rag.generate.auto_prompt_generator import AutoPromptGenerator
from dataflow.operators.agentic_rag.generate.depth_qa_generator import DepthQAGenerator
from dataflow.operators.agentic_rag.generate.qa_generator import QAGenerator
from dataflow.operators.agentic_rag.generate.qa_scorer import QAScorer
from dataflow.operators.agentic_rag.generate.width_qa_generator import WidthQAGenerator
from dataflow.operators.chemistry.generate.extract_smiles_from_text import ExtractSmilesFromText
from dataflow.operators.chemistry.eval.eval_smiles_equivalence import EvaluateSmilesEquivalence
from dataflow.operators.conversations.func_call_operators import ScenarioExtractor
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang


class RecommendPipeline(PipelineABC):
    def __init__(self):
        super().__init__()

        # -------- FileStorage (请根据需要修改参数) --------
        self.storage = FileStorage(
            first_entry_file_name="/mnt/public/data/lh/pzw/DataFlow/dataflow/example/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )


        # -------- LLM Serving (Local) --------
        llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="/mnt/public/model/huggingface/glm-4-9b-chat",
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=8192,
            hf_local_dir="local",
        )

        self.contentchooser = ContentChooser(num_samples="", method="random", embedding_serving="")
        self.f1scorer = F1Scorer()
        self.atomictaskgenerator = AtomicTaskGenerator(llm_serving=llm_serving, data_num=100, max_per_task=10, max_question=10)
        self.autopromptgenerator = AutoPromptGenerator(llm_serving=llm_serving)
        self.depthqagenerator = DepthQAGenerator(llm_serving=llm_serving, n_rounds=2)
        self.qagenerator = QAGenerator(llm_serving=llm_serving)
        self.qascorer = QAScorer(llm_serving=llm_serving)
        self.widthqagenerator = WidthQAGenerator(llm_serving=llm_serving)
        self.extractsmilesfromtext = ExtractSmilesFromText(llm_serving=llm_serving, prompt_template=None)
        self.evaluatesmilesequivalence = EvaluateSmilesEquivalence(llm_serving=llm_serving)
        self.scenarioextractor = ScenarioExtractor(llm_serving=llm_serving)

    def forward(self):
        self.contentchooser.run(
            storage=self.storage.step(), input_key="content"
        )
        self.f1scorer.run(
            storage=self.storage.step(), input_prediction_key="refined_answer", input_ground_truth_key="golden_doc_answer", output_key="F1Score"
        )
        self.atomictaskgenerator.run(
            storage=self.storage.step(), input_key="prompts", output_question_key="question", output_answer_key="answer", output_refined_answer_key="refined_answer", output_optional_answer_key="optional_answer", output_llm_answer_key="llm_answer", output_golden_doc_answer_key="golden_doc_answer"
        )
        self.autopromptgenerator.run(
            storage=self.storage.step(), input_key="text", output_key="generated_prompt"
        )
        self.depthqagenerator.run(
            storage=self.storage.step(), input_key="question", output_key="depth_question"
        )
        self.qagenerator.run(
            storage=self.storage.step(), input_key="text", output_prompt_key="generated_prompt", output_quesion_key="generated_question", output_answer_key="generated_answer"
        )
        self.qascorer.run(
            storage=self.storage.step(), input_question_key="generated_question", input_answer_key="generated_answer", output_question_quality_key="question_quality_grades", output_question_quality_feedback_key="question_quality_feedbacks", output_answer_alignment_key="answer_alignment_grades", output_answer_alignment_feedback_key="answer_alignment_feedbacks", output_answer_verifiability_key="answer_verifiability_grades", output_answer_verifiability_feedback_key="answer_verifiability_feedbacks", output_downstream_value_key="downstream_value_grades", output_downstream_value_feedback_key="downstream_value_feedbacks"
        )
        self.widthqagenerator.run(
            storage=self.storage.step(), input_question_key="question", input_identifier_key="identifier", input_answer_key="answer", output_question_key="generated_width_task"
        )
        self.extractsmilesfromtext.run(
            storage=self.storage.step(), content_key="text", abbreviation_key="abbreviations", output_key="synth_smiles"
        )
        self.evaluatesmilesequivalence.run(
            storage=self.storage.step(), golden_key="golden_label", synth_key="synth_smiles", output_key="final_result"
        )
        self.scenarioextractor.run(
            storage=self.storage.step(), input_chat_key="", output_key="scenario"
        )


if __name__ == "__main__":
    pipeline = RecommendPipeline()
    pipeline.compile()
    pipeline.forward()
