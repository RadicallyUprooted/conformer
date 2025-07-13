import torch
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder import download_pretrained_files
from text_processor.processor import CharTextTransform

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self,
                text_transform: CharTextTransform,
                blank: str = '<blank>'):
        super().__init__()
        self.text_transform = text_transform
        self.blank = self.text_transform.char_map[blank]

    def forward(self, emission: torch.Tensor) -> str:

        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        transcript = "".join([self.text_transform.index_map[i.item()] for i in indices])
        transcript = transcript.replace("|", " ").strip()

        return transcript

class BeamSearchDecoder(torch.nn.Module):
    def __init__(self,
                text_transform: CharTextTransform,
                blank: str = '<blank>',
                beam_size: int = 25):
        super().__init__()
        self.text_transform = text_transform
        self.blank = self.text_transform.char_map[blank]
        self.beam_size = beam_size
        self.vocab = [self.text_transform.index_map[i] for i in range(len(self.text_transform.index_map))]
        self.files = download_pretrained_files("librispeech-4-gram")
        
        self.decoder = ctc_decoder(
            lexicon=self.files.lexicon,
            tokens=self.vocab,
            beam_size=self.beam_size,
            blank_token=blank,
            lm=self.files.lm
        )

    def forward(self, emission: torch.Tensor) -> str:

        emission = emission.unsqueeze(0)
        beam_search_result = self.decoder(emission)

        tokens_str = "".join(self.decoder.idxs_to_tokens(beam_search_result[0][0].tokens))
        transcript = " ".join(tokens_str.split("|"))

        return transcript
