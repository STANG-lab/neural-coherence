import torch
import torch.nn.functional as F


class LSA():
    def __init__(self) -> None:
        pass

    def local_coherence(sentences: torch.Tensor) -> torch.Tensor:
        """Given a tensor of sentence embeddings, compute the cosine similarity between each adjacent pair."""
        pairs = sentences.unfold(dimension=0, size=2, step=1)
        l, r = pairs[:, 0], pairs[:, 1]
        cos_sim = F.cosine_similarity(l, r)
        return cos_sim