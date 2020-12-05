# modify from clovaai

import torch

from .base_convert import BaseConverter
from .registry import CONVERTERS


@CONVERTERS.register_module
class CTCConverterIndex(BaseConverter):
    def __init__(self, character: str, batch_max_length: int, *args, **kwargs):
        list_token = ['[blank]']
        list_character = list(character)
        self.batch_max_length = batch_max_length
        super(CTCConverterIndex, self).__init__(character=list_token + list_character)

    def encode(self, text: list):
        text = text
        length = [len(s) for s in text]
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(0)
        for i, t in enumerate(text):
            # 直接索引
            text = list(map(int, list(t.split(' ')))) if t else [13,]
            batch_text[i][:len(text)] = torch.LongTensor(text)

        return batch_text, torch.IntTensor(length), batch_text

    def decode(self, text_index):
        texts = []
        batch_size = text_index.shape[0]
        length = text_index.shape[1]
        for i in range(batch_size):
            t = text_index[i]
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy().tolist()
            char_list = []
            for idx in range(length):
                if t[idx] != 0 and (not (idx > 0 and t[idx - 1] == t[idx])):
                    # char_list.append(self.character[t[idx]])
                    char_list.append(str(t[idx]))
            text = ' '.join(char_list)
            texts.append(text)
        return texts

    def train_encode(self, text):
        return self.encode(text)

    def test_encode(self, text):
        if isinstance(text, int):
            text = ['' for idx in range(text)]
            
        return self.encode(text)
