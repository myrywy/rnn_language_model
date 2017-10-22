import os
import sys
import re
from collections import deque, namedtuple
import graf

parser = graf.GraphParser()


class NoSentenceAnnotationsFound(Exception):
    pass


def get_naive_annotated_text(file_name):

    graph = parser.parse("{}.hdr".format(file_name))

    f = open("{}.txt".format(file_name))
    text = f.read()
    annotated_text = ""
    labels = [(node.links[0][0].anchors, node.annotations.get_first(('sense')).features['synset']) for node in graph.nodes if [*node.annotations.select('sense')]]

    labels_left = deque(labels)

    for i, char in enumerate(text):
       annotated_text = annotated_text + char
       for anchors, synset_id in labels_left:
           if i+1 == anchors[1]:
               annotated_text = annotated_text + "/" + synset_id
               labels_left.popleft()
               break
           if anchors[1] > i:
               break

    f.close()


def get_senses(graph):
    senses_regions = []
    for node in graph.nodes:
        if not [*node.annotations.select('sense')]:
            continue
        starts = [link.anchors[0] for link in node.links[0]]
        stops = [link.anchors[1] for link in node.links[0]]
        range = [min(starts), max(stops)]

        senses_regions.append((range, node.annotations.get_first(('sense')).features['synset']))
    return senses_regions


def get_tokens(graph, text):
    try:
        return get_tokens_ptb(graph, text)
    except NoSentenceAnnotationsFound:
        return get_tokens_penn(graph, text)


def get_tokens_penn(graph, text):
    tokens = []
    for node in graph.nodes:
        if node.id[:4] == "penn" and list(node.annotations.select("tok")):
            try:
               tokens.append(node.links[0][0].anchors)
            except IndexError:
                continue
    tokens = sorted(tokens)
    sentences_regions = []
    sentence = []
    for region in tokens:
        sentence.append(region)
        if re.search("[?!.] ?", text[region[0]: region[1]]):
            sentences_regions.append(sentence)
            sentence = []
    sentences = [[text[start: stop] for start, stop in sent] for sent in sentences_regions]
    return sentences, sentences_regions


def get_tokens_ptb(graph, text):
    ptb_root = graph.nodes.get("ptb-n00000")
    if not ptb_root:
        raise NoSentenceAnnotationsFound
    sentenes_regions = []
    for sent_node in ptb_root.iter_children():
        path = deque()
        path.append(sent_node)
        tokens = []
        #import pdb; pdb.set_trace()
        while path:
            current = path.popleft()
            if current.visited:
                continue
            current.visit()
            children = list(current.iter_children())
            if children:
                path.extendleft(children)
            else:
                try:
                    tokens.append(tuple(current.links[0][0].anchors))
                except IndexError as err:
                    pass
        tokens = list(sorted(tokens))
        sentenes_regions.append(tokens)
    sentenes_regions = list(sorted(sentenes_regions))
    sentences = [[text[start: stop] for start, stop in sent] for sent in sentenes_regions]
    return sentences, sentenes_regions


# remove tokens overlapped by some bigger tokens
def remove_overlapped(labels):
    out = []
    for a in labels:
        overlapped = False
        for b in labels:
            if b[0] <= a[0] and a[1] <= b[1] and (a[0] != b[0] or a[1] != b[1]):
                overlapped = True
                break
        if not overlapped:
            out.append(a)
    return out


def get_sense_annotated_text(file_name):
    """path to file without extension. It is assumed that text is in .txt file and header in .hdr file.

    for example: file_name = '/home/marcin/NLP/MASC-3.0.0-bn/data/written/journal/Article247_66'"""
    with open("{}.txt".format(file_name)) as f:
        text = f.read()
    try:
        graph = parser.parse("{}.hdr".format(file_name))
    except Exception as err:
        raise RuntimeError("Unable to parse a file", file_name) from err
    try:
        sents, sents_regions = get_tokens(graph, text)
    except NoSentenceAnnotationsFound as err:
        raise NoSentenceAnnotationsFound(file_name) from err
    senses_regions = get_senses(graph)

    sense_labels = [(label[0][0], label[0][1], label[1]) for label in senses_regions]
    j = 0
    for sent in sents_regions:
        for token_start, token_stop in sent:
            u = 0
            while True:
                end_of_sense_regions = j + u >= len(senses_regions)
                if not end_of_sense_regions and senses_regions[j+u][0][0] <= token_start and \
                        senses_regions[j+u][0][1] >= token_stop:
                    j += u
                    break
                else:
                    if end_of_sense_regions or senses_regions[j+u][0][0] > token_stop:
                        sense_labels.append((token_start, token_stop, None))
                        break
                    else:
                        u += 1

    sense_labels = list(sorted(sense_labels, key=lambda x: (x[0], x[1])))
    sense_labels = remove_overlapped(sense_labels)
    sense_labels = [(text[start:stop], start, stop, sense) for start, stop, sense in sense_labels]
    return sense_labels


class CorpusText:
    def __init__(self, corpus_path, file_path):
        self._corpus_path = corpus_path
        self.file_path = file_path
        self.sentences = get_sense_annotated_text(os.path.join(corpus_path, file_path))


def read_masc(path):
    """Reads sense annotated MASC corpus.

    :param path: path to main dir of MASC corpus
    :return: List of sentences, each as a list of tuples of a form:
        (token:str, starting_index:int, end_index:int, babelnet_synset_idstr)
    """
    header_files = []
    for subdir, dirs, files in os.walk(os.path.join(path, "data")):
        for file in files:
            if file[-4:] == ".hdr":
                header_files.append(os.path.join(subdir[len(path):], file)[1:-4]) # strip leading slash and extension
    corpus_files = []
    read = 0
    for header_subpath in set(header_files):
        try:
            corpus_files.append(CorpusText(path, header_subpath))
            read += 1
            print("\r{} files read".format(read), file=sys.stderr, end="")
        except NoSentenceAnnotationsFound as err:
            print("\nCorpus file is not readable:", err.args[0], file=sys.stderr)
            continue
        except RuntimeError as err:
            print("\nCorpus file is not readable - parsing error:", err.args[0], file=sys.stderr)
    return corpus_files
