from sentence_transformers import SentenceTransformer
from src.embedders.embedder import Embedder


class MiniLM(Embedder):
    """all-MiniLM-L6-v2 is an extremely fast LM embedding scheme that still provides good results."""
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
 
    def embed(self, sentences):
        return self.model.embed(sentences)