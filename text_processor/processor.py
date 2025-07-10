import torch

class CharTextTransform:
    """
    A transform to convert text to a sequence of integers and back.
    """
    def __init__(self):
        self.char_map = {
            "'": 0, ' ': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8,
            'h': 9, 'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16,
            'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21, 'u': 22, 'v': 23, 'w': 24,
            'x': 25, 'y': 26, 'z': 27
        }
        self.index_map = {v: k for k, v in self.char_map.items()}

    def text_to_int(self, text):
        return [self.char_map[c] for c in text.lower()]

    def int_to_text(self, labels):
        return "".join([self.index_map[i] for i in labels])

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i.item()] for i in indices])    