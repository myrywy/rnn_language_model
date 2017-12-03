import numpy
import candidates


class VectorDistanceClassifier:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def weighted(self, predictions, candidates):
        """
        Każdy kandydat zbiera głosy postaci - odległość od przewidzianego słoa / prawdopodobieństwo tego słowa
        :param predictions:
        :param candidates:
        :return:
        """
        raise NotImplementedError

    def most_probable(self, predictions, candidates):
        """

        :param predictions: List[(babelnet_id: str, probability: float)]
        :param candidates: List[(ort form, babalnet_id)]
        :return:
        """
        return self.choose_closest(predictions[0][0], candidates)

    def choose_closest(self, predicted: str, candidates):
        """
        :param predicted: A babelnet id of a predicted word
        :param candidates: List[(ort form, babalnet_id)]
        :return: babelnet_id, distance
        """
        predicted_vec = self.vocabulary.word2vec(predicted)
        distances = []
        for candidate in candidates:
            distances.append((candidate[0], numpy.linalg.norm(self.vocabulary.word2vec(candidate[1]) - predicted_vec)))
        return list(sorted(distances, key=lambda x: x[1]))[0]


if __name__ == "__main__":
#def test_distance():
    embeddings_path = "./first10proc_NASARI_embed_english.txt"
    ids, names, vectors = candidates.parse_bn_vectors(embeddings_path, 420300, 300)
    sv = candidates.SemanticVocabulary(ids, None, vectors, names)
    # some meanings of the word "bank" ((id, vector) tuples)
    banks = sv.possible_meanings("bank")
    # one meaning of word insurance (babelnet id)
    insurance = sv.possible_meanings("insurance")[0][1]
    money = sv.possible_meanings("money")[0][1]
    # we expect that basic meaning ('bank', 'bn:00008364n') ("a financial institution that...") is closest to insurance
    # symulujemy sytuację, w której przepowiedziano słowo "insurance" na miejscu słowa "bank" więc chcemy wybrać
    # znaczenie słowa bank, które jest najbliżej przewidzianego
    classifier = VectorDistanceClassifier(sv)
    l1 = classifier.choose_closest(insurance, banks)
    l2 = classifier.choose_closest(money, banks)
