import torch
from abc import ABC, abstractmethod

class TextProcessor(ABC):
    @abstractmethod
    def text2int(self, data):
        pass

    @abstractmethod
    def int2text(self, data):
        pass

    def __init__(self):
        self.blank_label = 0

    def decode(self, arg_maxes: torch.Tensor):
        decode = []
        for i, index in enumerate(arg_maxes):
            if index != self.blank_label:
                if i != 0 and index == arg_maxes[i - 1]:
                    continue
                decode.append(index.item())
        return self.int2text(decode)

class CharacterBased(TextProcessor):
    aux_vocab = ["<p>", "<s>", "<e>", " ", ":", "'"]

    origin_list_vocab = {
        "en": aux_vocab + list("abcdefghijklmnopqrstuvwxyz"),
    }

    origin_vocab = {
        lang: dict(zip(vocab, range(len(vocab))))
        for lang, vocab in origin_list_vocab.items()
    }

    def __init__(self, lang: str = "en"):
        super().__init__()
        self.lang = lang
        self.vocab = self.origin_vocab[lang]
        self.list_vocab = self.origin_list_vocab[lang]

    def text2int(self, s: str) -> torch.Tensor:
        return torch.Tensor([self.vocab[i] for i in s.lower()])

    def int2text(self, s: torch.Tensor) -> str:
        return "".join([self.list_vocab[i] for i in s if i > 2])