from nltk import word_tokenize


class Snippet:
    def __init__(self, snippet_text: str, target_phrase: str):
        self.snippet = self._tokenize(snippet_text)
        self.target_phrase = self._tokenize(target_phrase)
        self.masked_snippet, self.target_indices = self._hide_target_phrase(self.snippet)

    def _hide_target_phrase(self, tokenized_text, masking_text="<UNKNOWN>"):
        tokenized_text = tokenized_text[:]
        indices = []
        i = 0
        length = len(tokenized_text)
        phrase_length = len(self.target_phrase)
        while i < length - 1:
            for i in range(i, length):
                if tokenized_text[i:i+phrase_length] == self.target_phrase:
                    tokenized_text[i:i + phrase_length] = [masking_text]
                    indices.append(i)
                    length = len(tokenized_text)
                    break
        return tokenized_text, indices

    def _tokenize(self, text):
        return word_tokenize(text.lower())

