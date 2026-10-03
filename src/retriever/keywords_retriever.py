from nltk.tokenize import word_tokenize
from ..dataclass.result import RetrieverResult
from .base_retriever import BaseRetriever
from typing import List, Dict, Any, Optional
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keybert import KeyBERT

class KeywordsRetriever(BaseRetriever):
    def __init__(self, 
                 
                 name: str = "KeywordsRetriever", 
                 top_k: int = 5
                 ):
        super().__init__(name=name, top_k=top_k)
        self._kw_model = KeyBERT()
        self.stemmer = PorterStemmer()
        self._doc_tokens: List[List[str]] = []

    def _retrieve_impl(self, 
                       query: str, 
                       top_k: int, 
                       **kwargs) -> List[RetrieverResult]:
        
        return 

    def _extract_keywords(self, query: str) -> List[str]:
        tokens = word_tokenize(query)
        return tokens

    def tokenize(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalnum()]
        return tokens

    def _score(self, query_keywords: List[str], chunks_keywords: List[str]) -> Dict[str, float]:
        # Placeholder for scoring logic
        return {}
