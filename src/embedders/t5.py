from sentence_transformers import SentenceTransformer
from src.embedders.embedder import Embedder


class T5(Embedder):
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    def embed(self, sentences):
        return self.model.embed(sentences)