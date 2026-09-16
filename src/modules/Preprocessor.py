from FlagEmbedding import BGEM3FlagModel
import torch
import unicodedata
import re
from nltk.stem import PorterStemmer

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
        if not query:
            return ""
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query)
        query = re.sub(r'[\U0001F600-\U0001F64F]', '', query)
        query = re.sub(r'([!?！？。，,.])\1{1,}', r'\1', query)
        return query.strip()

    def _normalizeQuery(self, query: str) -> str:
        # Implement normalization logic here if needed
        query = unicodedata.normalize('NFKC', query)
        
        query = query.replace('"', '"').replace('"', '"')
        query = query.replace(''', "'").replace(''', "'")
        query = query.replace('「', '"').replace('」', '"')
        
        query = query.replace('—', '-').replace('-', '-')
        query = re.sub(r'\s+([，。！？；：,.!?;:])', r'\1', query)
        return query

    def _tokenizeQuery(self, query: str) -> list[str]:
        # Implement tokenization logic here if needed
        stemmer = PorterStemmer()
        return " ".join([stemmer.stem(w) for w in query.split()])

    def _extractKeywords(self, query: list[str]) -> list[str]:
        # Implement keyword extraction logic here if needed
        tokens = self._tokenizeQuery(query)
        
        keywords = [
            t for t in tokens
            if len(t) > 1
            and not re.match(r'^[\W_]+$', t)  
        ]
        
        return keywords


