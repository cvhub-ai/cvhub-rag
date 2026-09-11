from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import FlagReranker
import torch

class Inference:
    def __init__(self, embedded_model_path: str, rerank_model_path: str):
        self.embedded_model = BGEM3FlagModel(embedded_model_path, use_fp16=True)
        self.rerank_model = FlagReranker(rerank_model_path, use_fp16=True)

    def predict(self, input_data: list):
        return 

    def __call__(self, input_data: list):
        return self.predict(input_data)

    def Embedder(self, input_data: list):
        return

    def Retriever(self, input_data: list):
        return 

    def Decoder(self, input_data: list):
        return
