from typing import Literal
from sentence_transformers import SentenceTransformer
from src.embedders.embedder import Embedder

GloveSizes = Literal["6B", "840B"]

class Glove(Embedder):
    """Creates 300-dimensional GloVe embeddings."""

    def __init__(self, size: GloveSizes = "6B"):
        """Args:
            size: ("6B" | "840B") The size of the Glove model to use.
                6B: Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 822 MB download) 
                840B: Common Crawl (840B tokens, 2.2M vocab, cased, 2.03 GB download)"""
        self.model = SentenceTransformer(f'average_word_embeddings_glove.{size}.300d')
 
    def embed(self, sentences):
        return self.model.embed(sentences)