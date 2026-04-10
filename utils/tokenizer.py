from collections import Counter


class SimpleTokenizer:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.token_to_id = {}
        self.id_to_token = {}

    @property
    def pad_id(self):
        return self.token_to_id[self.pad_token]

    @property
    def sos_id(self):
        return self.token_to_id[self.sos_token]

    @property
    def eos_id(self):
        return self.token_to_id[self.eos_token]

    @property
    def unk_id(self):
        return self.token_to_id[self.unk_token]

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def _basic_tokenize(self, text):
        return text.strip().lower().split()

    def build_vocab(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            counter.update(self._basic_tokenize(text))

        tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        for token, freq in counter.items():
            if freq >= min_freq:
                tokens.append(token)

        self.token_to_id = {tok: i for i, tok in enumerate(tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text, add_sos=False, add_eos=False, max_len=None, pad_to_max=False):
        tokens = self._basic_tokenize(text)
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

        if add_sos:
            ids = [self.sos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        if max_len is not None:
            ids = ids[:max_len]
            if pad_to_max:
                ids = ids + [self.pad_id] * (max_len - len(ids))

        return ids

    def decode(self, ids, skip_special=True):
        special_ids = {self.pad_id, self.sos_id, self.eos_id}
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.unk_token)
            if skip_special and int(idx) in special_ids:
                continue
            tokens.append(token)
        return " ".join(tokens)
