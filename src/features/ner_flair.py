from flair.data import Sentence
from flair.models import SequenceTagger
import torch


class FlairEntityRecognizer:
    def __init__(self, sentences) -> None:
        self.tagger = SequenceTagger.load("flair/ner-english")

    def build_sentences(sentences) -> list:
        """Convert a tensor of sentence strings into a list of Flair sentence objects"""
