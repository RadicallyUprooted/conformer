class CharTextTransform:
    """
    A transform to convert text to a sequence of integers and back.
    """
    def __init__(self):
        self.char_map = {
            "'": 0, ' ': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8,
            'h': 9, 'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16,
            'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21, 'u': 22, 'v': 23, 'w': 24,
            'x': 25, 'y': 26, 'z': 27, '|': 28, '<blank>': 29
        }
        self.index_map = {v: k for k, v in self.char_map.items()}
        self.blank = self.char_map['<blank>']

    def text_to_int(self, text):
        return [self.char_map[c] for c in text.lower()]

    def int_to_text(self, labels):
        return "".join([self.index_map[i] for i in labels])