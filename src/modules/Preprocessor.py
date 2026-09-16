from FlagEmbedding import BGEM3FlagModel
import torch
import unicodedata

class Preprocessor:
    def __init__(self, embedded_model_path: str) -> None:
        self.embedded_model = BGEM3FlagModel(embedded_model_path, use_fp16=True)

    def predict(self, input_data: str) -> torch.Tensor:
        pre = self.preprocess(input_data)
        return self.embedder(pre)

    def preprocess(self, input_data: str) -> tuple:
        cleaned = self._cleanQuery(input_data)
        normalized = self._normalizeQuery(cleaned)
        tokens = self._tokenizeQuery(input_data)
        keywords = self._extractKeywords(tokens)
        return normalized, keywords

    def __call__(self, input_data: str) -> tuple:
        return self.preprocess(input_data)

    def embedder(self, input_data: str) -> torch.Tensor:
        result = self.embedded_model.encode(input_data)
        return result

    def _cleanQuery(self, query: str) -> str:
        # Implement any necessary cleaning or preprocessing of the query here
        return query.strip().lower()

    def _normalizeQuery(self, query: str) -> str:
        # Implement normalization logic here if needed
        return query

    def _tokenizeQuery(self, query: str) -> list[str]:
        # Implement tokenization logic here if needed
        return query.split()  # Example: simple split by whitespace

    def _extractKeywords(self, query: list[str]) -> list[str]:
        # Implement keyword extraction logic here if needed
        return query  # Example: return the tokenized query


