import torch
from torchaudio.models.decoder import ctc_decoder

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, text_transform, blank=0):
        super().__init__()
        self.text_transform = text_transform
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
        transcript = "".join([self.text_transform.index_map[i.item()] for i in indices])
        transcript = transcript.replace("|", " ").strip()
        return transcript

class BeamSearchDecoder(torch.nn.Module):
    def __init__(self, text_transform, blank=0, beam_size=5):
        super().__init__()
        self.text_transform = text_transform
        self.blank = blank
        self.beam_size = beam_size
        
        self.vocab = [self.text_transform.index_map[i] for i in range(len(self.text_transform.index_map))]
        
        self.decoder = ctc_decoder(
            lexicon=None,
            tokens=self.vocab,
            beam_size=self.beam_size,
            blank_token='<blank>'
        )

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        emission = emission.unsqueeze(0)
        beam_search_result = self.decoder(emission)

        if not beam_search_result or not beam_search_result[0]:
            return ""

        best_hypotheses = beam_search_result[0][0]
        transcript = "".join(best_hypotheses.words).replace("|", " ").strip()
        return transcript
