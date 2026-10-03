import unicodedata
import re

class Preprocessor:
    def __init__(self, max_length: int = 1024) -> None:
        self.max_length = max_length

    def predict(self, input_data: str) -> str:
        return self.preprocess(input_data)

    def preprocess(self, input_data: str) -> str:
        cleaned = self._cleanQuery(input_data)
        normalized = self._normalizeQuery(cleaned)
        return normalized

    def _cleanQuery(self, query: str) -> str:
        if not query:
            return ""
        query = query[:self.max_length]
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



