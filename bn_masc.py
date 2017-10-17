from collections import deque, namedtuple
import graf

parser = graf.GraphParser()

file_name = "/home/marcin/NLP/MASC-3.0.0-bn/data/written/journal/Article247_66"

def get_naive_annotated_text(file_name):

    graph = parser.parse("{}.hdr".format(file_name))

    f = open("{}.txt".format(file_name))
    text = f.read()
    annotated_text = ""
    labels = [(node.links[0][0].anchors, node.annotations.get_first(('sense')).features['synset'] ) for node in graph.nodes if [*node.annotations.select('sense')]]

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
    return [(node.links[0][0].anchors, node.annotations.get_first(('sense')).features['synset']) for node in graph.nodes if [*node.annotations.select('sense')]]

def get_tokens(graph, text):
    ptb_root = graph.nodes.get("ptb-n00000")

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
                    # If it is a Trace node then it is ok to ignore it
                    try:
                        current.annotations.get_first('Trace')
                    except ValueError:
                        raise err from None
        tokens = list(sorted(tokens))
        sentenes_regions.append(tokens)
    sentenes_regions = list(sorted(sentenes_regions))
    #token_nodes = [node for node in graph.nodes if node.id[:3] == "ptb"]

    sentences = [[text[start: stop] for start, stop in sent] for sent in sentenes_regions]
    return sentences, sentenes_regions


with open("{}.txt".format(file_name)) as f:
    text = f.read()
graph = parser.parse("{}.hdr".format(file_name))
sents, sents_regions = get_tokens(graph, text)
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
sense_labels = [(text[start:stop], start, stop, sense) for start, stop, sense in sense_labels]

'''
sense_labels = [[None for _ in range(len(sent))] for sent in sents]
sentences_regions = deque(deque((region, t, s) for t, region in enumerate(sent)) for s, sent in enumerate(sents_regions))
current_sent = sentences_regions.popleft()
(token_start, token_stop), t, s = current_sent.popleft()
for region, sense in senses_regions:
    while token_start <= region[1]:
        if token_start >= region[0] and token_stop <= region[1]:
            sense_labels[s][t] = sense
        try:
            current_sent = sentences_regions.popleft()
            (token_start, token_stop), t, s = current_sent.popleft()
        except IndexError:
            continue
'''

'''senses_regions = deque(senses_regions)
sense_annotated_tokens = []
current_sense_tokents = []
for sent, sent_regions in zip(sents, sents_regions):
    for token, token_region in zip(sent, sent_regions):
        sense_region, sense_id = senses_regions[0]
        if token_region[0] >= sense_region[0] and token_region[1] <= sense_region[1]:
            current_sense_tokents.append((token, token_region))
        elif token_region[0] >= sense_region[1]:
            current_sense = senses_regions.popleft()
            sense_annotated_tokens.append((current_sense_tokents, sense_id))
        if token_region[1] < sense_region[0]:
            sense_annotated_tokens.append((current_sense_tokents, None))'''


'''
for n in token_nodes:
    try:
        n.annotations.get_first('S')
    except ValueError:
        continue
    else:
        sents.append(n)
'''