from typing import List, Iterable
import bisect
import itertools
import numpy as np
import vocabulary

embeddings_path = "/media/marcin/USB-HDD/nasari/NASARI_embed_english.txt"
embeddings_path = "./first10proc_NASARI_embed_english.txt"

n_rows = 4420300
vector_size = 300


def parse_bn_vectors(embeddings_path, n_rows=None, vector_size=None):
    if not vector_size or not n_rows:
        with open(embeddings_path) as embeddings_file:
            header = next(embeddings_file)
            first = next(embeddings_file)
            vector_size = len(first.split()) - 1
            print("vector size:", vector_size)
            n_rows = 1
            for row in embeddings_file:
                n_rows += 1
                print(n_rows)

        print("vocabulary size:", n_rows)

    ids = []
    names = []
    vectors = np.ndarray((n_rows, vector_size))
    with open(embeddings_path) as embeddings_file:
        head = next(embeddings_file)
        for i, line in enumerate(embeddings_file):
            elements = line.strip().split()
            bn_id, name = elements[0].split("__")
            ids.append(bn_id)
            names.append(name.lower())
            vectors[i] = [float(f) for f in elements[1:]]
            vectors[i] = [float(f) for f in elements[1:]]

    return ids, names, vectors


class SemanticVocabulary(vocabulary.Vocabulary):
    UNKNOWN_TAG = ("<UNKNOWN>", None)

    def __init__(self, sense_ids: List[str], ids: List[int], vectors: np.ndarray, words: List[str]):
        if ids is None:
            ids = list(range(1, len(words)+1))
        super().__init__(words, ids, vectors)
        self.sense_ids = sense_ids
        self.sorted_wors = list(sorted(zip(words, sense_ids)))

    def possible_meanings(self, word: str):
        underscored_word = word.replace(" ", "_")
        index_min = bisect.bisect_left(self.sorted_wors, (underscored_word, ""))
        index_max = bisect.bisect_right(self.sorted_wors, (underscored_word+"_)", ""))
        candidates = []
        for voc_word_raw, bn_id in self.sorted_wors[index_min:index_max+1]:
            voc_word = voc_word_raw.replace("_", " ")
            if voc_word.find(word) == 0:
                if len(voc_word) == len(word) or voc_word[len(word):len(word) + 2] == " (":
                    candidates.append((voc_word_raw, bn_id))
                    continue
            else:
                pass#break
        return candidates

    def possible_meaning_vectors(self, word: str, allow_unknown=False):
        candidates = []
        for voc_word, bn_id in self.possible_meanings(word):
            candidates.append((bn_id, self.word2vec(bn_id)))
        if not candidates and allow_unknown:
            return [(self.id2word(0), self.id2vec(0))]
        return candidates

    def cart_meanings_of_snetence(self, sent: List[str]):
        """
        Generates every possible combination of meandning of a sentence
        :param sent: list of words representing sentence
        :return: list of iterables of senses (List[List[(babelnet_identifier, babelnet_vector)]])
        """
        return list(itertools.product(*[self.possible_meaning_vectors(word, allow_unknown=True) for word in sent]))


if __name__ == "__main__":
    ids, names, vectors = parse_bn_vectors(embeddings_path, 420300, 300)
    sv = SemanticVocabulary(ids, None, vectors, names)
