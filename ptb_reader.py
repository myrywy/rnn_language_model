from os.path import join
import input_data
import vocabulary


class GenIt:
    def __init__(self, gen_f, *args, **kwargs):
        self.gen_f = gen_f
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self.gen_f(*self.args, **self.kwargs)


def read_ptb(path=join("ptb","data")):
    def read_sents(filename):
        with open(filename) as file:
            for line in file:
                yield line.lower().split() + ["</snt>"]
    file_reader = lambda filename: GenIt(read_sents, filename)
    return input_data.InputData.prepeare_one_hot_input(
        file_reader(join(path, "ptb.train.txt")),
        file_reader(join(path, "ptb.valid.txt")),
        file_reader(join(path, "ptb.test.txt")))
