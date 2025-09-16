from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.general_text import NgramScorer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.nn.functional import normalize
from transformers import BertModel, BertTokenizer

@OPERATOR_REGISTRY.register()
class SemDeduplicator(OperatorABC):
    def __init__(self, eps: float = 0.05, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', model_cache_dir: str = './dataflow_cache', device: str = 'cuda', use_dynamic_threshold: bool = False, output_similarity_matrix: bool = False):
        self.logger = get_logger()
        self.eps = eps
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.device = device
        self.use_dynamic_threshold = use_dynamic_threshold
        self.output_similarity_matrix = output_similarity_matrix
        self.model = BertModel.from_pretrained(self.model_name, cache_dir=model_cache_dir).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, cache_dir=model_cache_dir)
        self.logger.info(f"Initializing {self.__class__.__name__} with eps = {self.eps}, model_name = {self.model_name}, model_cache_dir = {self.model_cache_dir}, device = {self.device}, use_dynamic_threshold = {self.use_dynamic_threshold}, output_similarity_matrix = {self.output_similarity_matrix}")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于BERT语义相似度实现高精度语义去重，支持动态阈值调节、相似度矩阵输出、大规模数据处理与增量更新。通过计算文本嵌入向量间的余弦相似度，识别语义相似的文本并保留唯一样本，支持多字段组合输入与可定制算法。"
                "输入参数："
                "- eps：基础相似度阈值，值越小表示允许的相似度越低，默认为0.05（即余弦相似度大于0.95视为重复）"
                "- model_name：预训练模型名称，默认为'sentence-transformers/all-MiniLM-L6-v2'"
                "- model_cache_dir：模型缓存目录，默认为'./dataflow_cache'"
                "- device：模型运行设备，默认为'cuda'"
                "- use_dynamic_threshold：是否启用动态阈值，基于数据分布自动调整阈值"
                "- output_similarity_matrix：是否输出相似度矩阵，用于后续分析"
                "- input_keys：多个输入字段名列表，与input_key二选一"
                "- input_key：单个输入字段名，与input_keys二选一"
                "- output_key：去重结果字段名，默认为'minhash_deduplicated_label'"
                "输出参数："
                "- 过滤后的DataFrame，仅保留语义不重复的样本（标记为1的样本）"
                "- 返回包含去重结果字段名和相似度矩阵字段名（若启用）的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Semantic deduplication based on BERT embeddings with support for dynamic threshold, similarity matrix output, large-scale processing, and incremental updates. Computes cosine similarity between text embeddings to identify semantically similar texts and retain unique samples. Supports multi-field input and customizable algorithms."
                "Input Parameters:"
                "- eps: Base similarity threshold, smaller values allow lower similarity, default is 0.05 (cosine similarity > 0.95 considered duplicate)"
                "- model_name: Pretrained model name, default is 'sentence-transformers/all-MiniLM-L6-v2'"
                "- model_cache_dir: Model cache directory, default is './dataflow_cache'"
                "- device: Model running device, default is 'cuda'"
                "- use_dynamic_threshold: Whether to enable dynamic threshold based on data distribution"
                "- output_similarity_matrix: Whether to output similarity matrix for downstream analysis"
                "- input_keys: List of multiple input field names, alternative to input_key"
                "- input_key: Single input field name, alternative to input_keys"
                "- output_key: Deduplication result field name, default is 'minhash_deduplicated_label'"
                "Output Parameters:"
                "- Filtered DataFrame containing only semantically unique samples (marked as 1)"
                "- List containing deduplication result field name and similarity matrix field name (if enabled) for downstream operator reference"
            )
        else:
            return "High-precision semantic deduplication using BERT embeddings with dynamic threshold and similarity matrix output."

    def run(self, storage: DataFlowStorage, input_keys: list = None, input_key: str = None, output_key: str = 'sem_deduplicated_label'):
        if input_keys is None and input_key is None:
            self.logger.error(f"Need to specify either input_keys or input_key!")
            raise ValueError(f"Need to specify either input_keys or input_key!")
        if input_keys is not None and input_key is not None:
            self.logger.error(f"{self.__class__.__name__} only need one input args!")
            raise ValueError(f"{self.__class__.__name__} only need one input args!")
        if input_keys is not None:
            self.logger.info(f"Running {self.__class__.__name__} with input_keys = {input_keys} and output_key = {output_key}")
        else:
            self.logger.info(f"Running {self.__class__.__name__} with input_key = {input_key} and output_key = {output_key}")
        self.input_key = input_key
        self.input_keys = input_keys
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        texts = []
        for idx, sample in tqdm(enumerate(dataframe.to_dict(orient='records')), desc=f"Implementing {self.__class__.__name__}", total=len(dataframe)):
            if input_keys is not None and len(input_keys) > 1:
                text = '\n'.join([f"{k}:\n{sample[k]}" for k in input_keys])
            else:
                text = sample[self.input_key]
            texts.append(text)

        # Compute embeddings
        embeddings = get_text_embedding(texts, self.tokenizer, self.model, self.device)
        embeddings = normalize(torch.tensor(embeddings), dim=1)

        # Compute cosine similarity matrix
        cos_sim_matrix = cosine_similarity(embeddings.numpy())
        cos_sim_matrix = torch.tensor(cos_sim_matrix)
        cos_sim_matrix.fill_diagonal_(0)

        # Dynamic threshold adjustment
        if self.use_dynamic_threshold:
            threshold = np.percentile(cos_sim_matrix[cos_sim_matrix > 0], 95)
        else:
            threshold = 1 - self.eps

        # Mark duplicates
        labels = [1] * len(dataframe)
        for i in range(len(dataframe)):
            for j in range(i + 1, len(dataframe)):
                if cos_sim_matrix[i][j] >= threshold:
                    labels[j] = 0

        dataframe[self.output_key] = labels
        filtered_dataframe = dataframe[(dataframe[self.output_key] > 0)]
        output_file = storage.write(filtered_dataframe)

        # Output similarity matrix if enabled
        if self.output_similarity_matrix:
            sim_matrix_key = f"{output_key}_sim_matrix"
            storage.write(cos_sim_matrix.numpy(), key=sim_matrix_key)
            self.logger.info(f"Similarity matrix saved with key: {sim_matrix_key}")
            return [self.output_key, sim_matrix_key]

        self.logger.info(f"Deduplication completed. Total unique items: {sum(labels)}")
        return [self.output_key]


def get_text_embedding(texts, tokenizer, model, device, batch_size=32):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            # Use the [CLS] token embedding as the sentence embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)


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
 
# 3. Instantiate operator
operator = NgramScorer(ngrams=5)

# 4. Run
operator.run(storage=storage.step())
