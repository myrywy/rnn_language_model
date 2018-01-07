import vsmlib
import vocabulary


def vsm_embeddings_from_dir_vocabulary(path):
    embeddings = vsmlib.model.load_from_dir(path)
    return vsm_embeddings_to_vocabulary(embeddings)


def vsm_embeddings_to_vocabulary(embedding, filter_nsigned=True):
    ids, words, vectors = [], [], []
    for i, word in enumerate(embedding.vocabulary.lst_words):
        if filter_nsigned:
            if word[:1] == "#" and len(word) > 1:
                continue
        words.append(word)
        ids.append(i)
        vectors.append(embedding.get_row(word))
    return vocabulary.Vocabulary(words, ids, vectors)

