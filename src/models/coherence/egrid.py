import torch


class EntityGrid:
    def __init__(self, sentences: torch.Tensor) -> None:
        num_sentences = sentences.size(dim=0)
        entity_grid = torch.zeros(
            num_sentences,
        )

    def local_coherence(self):
        pass
