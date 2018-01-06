from collections import Counter
import numpy
import candidates
import snippets


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


class LMClosestClassifier:
    def __init__(self, vocabulary, language_model):
        self.distance_classifier = VectorDistanceClassifier(vocabulary)
        self.vocabulary = vocabulary
        self.lm = language_model

    def classify(self, text, target_word):
        s = snippets.Snippet(text, target_word)
        possible_interpretations = sv.cart_meanings_of_snetence(s.masked_text)
        best_interpretation = np.argmax(
            [self.lm.get_sentence_probability(interpretation for interpretation in possible_interpretations)]
        )
        senses = []
        probabilities = []
        for target_index in s.target_indices:
            sense, probability = best_interpretation[target_index]
            senses.append(sense)
            probabilities.append(probability)
        sense_counts = Counter(senses)
        predicted_sense = sense_counts.most_common(1)[0]
        candidates = self.vocabulary.possible_meanings(target_word)
        return self.distance_classifier.choose_closest(predicted_sense, candidates)[0]
        words_probs = self.lm.predict_words_in_sentece(s.masked_text, s.target_indices, sort=True)
        results = Counter(probs[-1] for probs in words_probs)
        self.distance_classifier.choose_closest(results.most_common(1)[0][0], candidates)



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
