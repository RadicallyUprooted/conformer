class CharTextTransform:
    """
    A transform to convert text to a sequence of integers and back.
    """
    def __init__(self):
        self.char_map = {
            "'": 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7,
            'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15,
            'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23,
            'x': 24, 'y': 25, 'z': 26, '|': 27, '<blank>': 28
        }
        self.index_map = {v: k for k, v in self.char_map.items()}
        self.blank = self.char_map['<blank>']

    def text_to_int(self, text):
        return [self.char_map[c] for c in text.lower()]

    def int_to_text(self, labels):
        return "".join([' ' if self.index_map[i] == '|' else self.index_map[i] for i in labels])