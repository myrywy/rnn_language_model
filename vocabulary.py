import numpy as np
import pickle
import tensorflow as tf
from typing import List, Iterable
from collections import defaultdict

strict_mode = True


class Vocabulary:
    """
    Instancje tej klasy reprezentują mapownie między słowami, wektorami i identyfikatorami używanymi przez sieć

    :param words: Lista wyrazów
    :param ids: Lista identyfikatorów odpowiadających wyrazom, przy czym id=0 jest zarezerwowane dla nieznanego słowa
    :param vectors: Tablica w której vecotrs[0] to wektor domyślny, który jest przypisywany nieznanym wyrazom oraz
        vectors[i] to wektor reprezentujący i-ty wyraz licząc od 1.
    """
    UNKNOWN_TAG = "<UNKNOWN>"
    def __init__(self, words: List[str], ids: List[int], vectors: np.ndarray):
        vectors = np.array(vectors)
        if strict_mode:
            if len(words) != len(ids):
                raise ValueError("There must be the same number of words and their ids")
            for i, (word, identifier) in enumerate(zip(words, ids)):
                if not isinstance(word, str):
                    raise ValueError("Words must all be strings, {}-th word is not str:\n{}".format(i, word))
                if not isinstance(identifier, int):
                    raise ValueError("Identifiers must all be ints, {}-th id is not int:\n{}".format(i, identifier))
        self.lookup = tf.constant(vectors)
        self._vector_length = vectors.shape[1]
        self.ids = defaultdict(lambda: 0, zip(words, ids))
        self.words = defaultdict(lambda: Vocabulary.UNKNOWN_TAG, zip(ids, words))

    def export_to_file_obj(self, file_obj):
        with tf.Session() as sess:
            pickle.dump(
                {"ids": [*self.ids.values()], "words": [*self.ids.keys()], "vectors": self.lookup.eval(session=sess)},
                file_obj)

    def export_to_file(self, path):
        with open(path, "wb") as f:
            self.export_to_file_obj(f)

    @classmethod
    def import_from_file_obj(cls, file_obj):
        data = pickle.load(file_obj)
        return cls(data["words"], data["ids"], data["vectors"])

    @classmethod
    def import_from_file(cls, path):
        with open(path, "rb") as f:
            return cls.import_from_file_obj(f)

    def to_vocab_file(self, filename):
        """
        Exports vocab in a format where i-th line (counting from 0) contains a word with id=i
        :return: None
        """
        with open(filename, "w") as file:
            size = max(self.words)
            file.writeln(Vocabulary.UNKNOWN_TAG)
            for i in range(size):
                try:
                    file.writeln(self.id2word(i))
                except KeyError:
                    file.writeln("")

    def vector_length(self):
        return self._vector_length

    def word2id(self, word: str):
        """
        Tłumaczy słowo w fomie stringa na id.
        """
        return self.ids[word]

    def word2vec(self, word: str):
        """
        Tłumaczy słowo w formie stringa na wektor. Ta funkcja ma charakter pomocniczy, sieci korzystają bezpośrednio
        z tensora lookup_tensor i operacji tf.nn.embedding_lookup.
        """
        return self.id2vec(self.word2id(word))

    def id2vec(self, identifier: int):
        """
        Tłumaczy id na wektor.
        """
        with tf.Session() as sess:
            return sess.run(tf.nn.embedding_lookup(self.lookup, identifier))

    def id2word(self, identifier: int):
        """
        Tłumaczy id na słowo w formie stringa.
        """
        return self.words[identifier]

    def vec2id(self, vec: np.ndarray):
        """
        Tłumaczy wektor na id.
        """
        raise NotImplementedError

    def vec2word(self, vec: np.ndarray):
        """
        Tłumaczy wektor na słowo w formie stringa.
        """
        raise NotImplementedError

    def get_lookup_tensor(self):
        """Zwraca tensor o wymiarach n+1 na m, gdzie ity wiersz zawiera wektor odpowiadający wyrazowi o id=i,
        n - rozmiar słownika, m - rozmiar wektorów."""
        return self.lookup

    def one_hot_to_id(self, one_hot_vector: np.ndarray):
        """
        Zamienia wektor one hot na id czyli jeśli na i-tej pozycji jest 1, to zwróci i. Ogólnie jeśli będzie to
        numer pozycji dla której wartość elementu wektora jest najwyższa, nie musi to być jeden.
        """
        return np.argmax(one_hot_vector)

    def one_hot_to_word(self, one_hot_vector: np.ndarray):
        """
        Zamienia wektor one hot na odpowiadający mu wyraz.
        """
        return self.id2word(self.one_hot_to_id(one_hot_vector))

    def size(self):
        """
        Returns size of vocabulary - number of embeding vectors i.e. number of words + 1 (where '+1' is for the unknow word).
        :return: int - number of ids/vectors in vocabulary = number of words + 1
        """
        return len(self.words)+1

    @staticmethod
    def create_vocabulary(words: Iterable[str], minimal_count, create_one_hot_embeddings=False):
        def count_words(words):
            counts = defaultdict(lambda: 0)
            for word in words:
                counts[word] += 1
            return counts

        def filter_vocab(word_counts, min_count):
            return {word for word in word_counts if word_counts[word] >= min_count}

        def prepare_words_ids_lookup(vocab):
            ''' założenie jest takie, że pierwszy, zerowy id to nieznane słowo'''
            return {word: i + 1 for i, word in enumerate(vocab)}

        def prepare_word_lookup(vocab):
            ''' założenie jest takie, że pierwszy, zerowy id to nieznane słowo'''
            ids_dict = prepare_words_ids_lookup(vocab)
            params = np.identity(len(ids_dict) + 1, dtype=np.float32)
            return ids_dict, params

        vocab = filter_vocab(count_words(words), minimal_count)
        if create_one_hot_embeddings:
            ids_words, lookup = prepare_word_lookup(vocab)
            words, ids = list(zip(*ids_words.items()))
            return Vocabulary(words, ids, lookup)
        else:
            ids, words = prepare_words_ids_lookup(vocab).items()
            return Vocabulary(words, ids, None)


