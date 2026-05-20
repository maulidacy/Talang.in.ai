import re
import tensorflow as tf

class BillTokenizer:
    def __init__(self):
        # matches words, numbers with punctuation, and stray symbols
        self.token_pat = re.compile(r'[a-zA-Z0-9]+(?:[-.,][a-zA-Z0-9]+)*|[^\w\s]')

        # placeholders for vocabulary building
        self.word2idx = {}
        self.char2idx = {}
        self.tag2idx = {}

    def _tokenize_spans(self, text):
        """method to get tokens + spans"""
        return [(m.group(), m.start(), m.end()) for m in self.token_pat.finditer(text)]

    def extract_tokens(self, text):
        """pure text tokenization"""
        # handle tf.string tensors if mapped via tf.data
        if isinstance(text, tf.Tensor):
            text = text.numpy().decode('utf-8')
        return [m.group() for m in self.token_pat.finditer(text)]

    def align_bio(self, text, entities):
        """aligns text and entity dictionaries into token/tag lists"""
        # handle tf.string tensors
        if isinstance(text, tf.Tensor):
            text = text.numpy().decode('utf-8')

        spans = self._tokenize_spans(text)
        tokens = [s[0] for s in spans]
        tags = ['O'] * len(spans)

        for ent in entities:
            e_start, e_end, e_label = ent['start'], ent['end'], ent['label']
            first = False
            for i, (_, t_start, t_end) in enumerate(spans):
                if t_start < e_end and t_end > e_start:
                    if not first:
                        tags[i] = f"B-{e_label}"
                        first = True
                    else:
                        tags[i] = f"I-{e_label}"

        return tokens, tags
