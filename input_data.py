import numpy as np
import tensorflow as tf
from typing import List, Iterable, Generator
import tempfile
import vocabulary as vc


class InputData:
    """
    Ta klasa ma za zadanie przetworzyć oryginalne zdania na tensory z identyfikatorami słów (kolejne wiersze to
    kolejne zdania). W tej wersji dane są zapisywane do plików w formacie TFRecord.
    TODO: udostępnić metodę tworzącą kolejkę FIFO (jako operację tensorflow), którą możnaby bezpośrednio karmić sięć
    (bez pośrednictwa pliku)
    """
    def __init__(self, vocabulary, sents=None):
        self.vocabulary = vocabulary
        self.sents = sents
        self.records_file = None

    def sents_to_id_lists(self, sents: Iterable[Iterable[str]]) -> Generator[List[int], None, None]:
        """
        Converts lists of words (strings) into lists of ids (ints)

        :param sents: Iterable of iterables of words where the inner iterable is interpreted as containing one sentence
        :return: generator yielding lists of ids - each list is one sentence
        """
        for sent in sents:
            word_ids = []
            for word in sent:
                try:
                    word_ids.append(self.vocabulary.ids[word.lower()])
                except KeyError:
                    word_ids.append(0)
            yield word_ids

    def _write_to_file(self, sents_as_ids: List[List[int]], filename: str):
        writer = tf.python_io.TFRecordWriter(filename)
        for sentence in sents_as_ids:
            ex = self.make_example(sentence)
            writer.write(ex.SerializeToString())
        writer.close()

    def write(self, filename, sents: Iterable[Iterable[str]]=None):
        """
        Writes data (as TFRecords, with words converted to ids as specified in self.Vocabulary).

        :param filename: path to a file where data is are to be stored
        :param sents: (optional) Iterable of iterables of words (inner iterable containing one sentence) if not provided
            then sentences provided in init are used
        :return: None
        """
        if sents:
            self._write_to_file(self.sents_to_id_lists(sents), filename)
        else:
            self._write_to_file(self.sents_to_id_lists(self.sents), filename)

    def dataset(self):
        if not self.recores_file:
            self.records_file = tempfile.mktemp()
            self.write(self.records_file)

        filenames = [self.records_file]
        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(input_data.get_single_example)
        return dataset

    @staticmethod
    def make_example(sentence: List[int]) -> tf.train.SequenceExample:
        """
        Makes SequenceExample representing the sentence. It contains "length" (number of words) in context data
        and ids as featire list "tokens".

        :param sentence: iterable of ids representing words of a sentence
        :return: SequenceExample that may be written to file or enqueued
        """
        ex = tf.train.SequenceExample()
        sentence_length = len(sentence)
        ex.context.feature["length"].int64_list.value.append(sentence_length)
        fl_tokens = ex.feature_lists.feature_list["tokens"]
        for token in sentence:
            fl_tokens.feature.add().int64_list.value.append(token)
        return ex

    @staticmethod
    def get_single_example(value):
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=value,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return context_parsed['length'],sequence_parsed['tokens']

    @staticmethod
    def prepeare_one_hot_input(*sents_sources, min_word_count=4):
        words = [word for sents in sents_sources for sent in sents for word in sent]
        vocabulary = vc.Vocabulary.create_vocabulary(words, min_word_count, create_one_hot_embeddings=True)
        data = [InputData(vocabulary, sents) for sents in sents_sources]
        return vocabulary, data
