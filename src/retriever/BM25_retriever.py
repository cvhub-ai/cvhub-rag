from .base_retriever import BaseRetriever
from ..dataclass.result import RetrieverResult, EmbeddedDoc
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

class BM25Retriever(BaseRetriever):
    def __init__(
            self,
            corpus: List[str],
            tokenizer = word_tokenize,
            name: str = "BM25Retriever",
            top_k: int = 5,
            k1: float = 1.5,
            b: float = 0.75,
            epsilon: float = 0.25,
            doc_idx: List[str] = None
            ):
        super().__init__(name=name, top_k=top_k)
        self.corpus = corpus
        self.tokenizer = tokenizer or self._default_tokenizer
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._bm25 = None
        self.doc_idx = doc_idx
        self._stemmer = PorterStemmer()
        self._doc_tokens: List[List[str]] = []

    def _ensure_ready(self):
        if self.corpus:
            self._build_index()
        self.ready = True

    def _build_index(self):
        self._doc_tokens = [self.tokenize(doc) for doc in self.corpus]
        self._bm25 = BM25Okapi(tokenizer=self._doc_tokens, k1=self.k1, b=self.b, epsilon=self.epsilon)

    def _retrieve_impl(self, 
                       query:str, 
                       top_k:int, 
                       **kwargs) -> List[RetrieverResult]:
        if self._bm25 is None:
            return []

        tokens = self.tokenize(query)

        scores = self._bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:top_k * 2]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue

            res = EmbeddedDoc(source=self.name, result=self._doc_tokens[idx])

            results.append(RetrieverResult(
                doc_id=self.doc_idx[idx],
                content=self.corpus[idx],
                score=score,
                results=res
            ))

            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _default_tokenizer(self, text: str, ) -> List[str]:
        # Simple whitespace tokenizer
        import re
        return re.findall(r'[\w\u4e00-\u9fff]+', text.lower())

    def tokenize(self, 
                 query: str, 
                 lowercase: bool = False, 
                 remove_stopwords: bool = True, 
                 stem: bool = True
                 ) -> List[str]:
        tokens = self.tokenizer(query)
        if lowercase:
            tokens = [token.lower() for token in tokens]
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        if stem:
            tokens = [self._stemmer.stem(token) for token in tokens]
        return tokens

    # def _extract_keywords(self, query: str) -> List[str]:
    #     idf_dict = dict(zip(self._bm25.idf.keys(), self._bm25.idf.values()))
    #     sorted_keywords = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)
    #     return 
