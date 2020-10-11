import os
import sys
from itertools import islice

from preprocessing_interface import PreprocessingInterface
import enchant
from dict import Dict
import multiprocessing as mp
import threading
from math import floor

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '../embed'))
from mmst import MMST
from embeddings import Loader


class SpellingCorrectionMMST(PreprocessingInterface):

    def __init__(self):
        self.nb = 0
        self.cores = mp.cpu_count()
        self.load = Loader()
        self.load.loadGloveModel()


    def prep_input(self):
        sentences = []
        with open(self.input, mode='r') as input:
            for line in input:
                sentences.append(line)
                self.nb += 1

        split_size = int(self.nb/self.cores) + 1
        for i in range(self.cores):
            with open(self.input + '_{}'.format(i), "w+") as f:
                start = min(i*split_size, self.nb)
                end = min((i+1)*split_size, self.nb)
                for j in range(start, end):
                    f.write(sentences[j])


    def merge_and_delete(self, delete_output=True):
        # merge split output files (and delete split files)
        out = open(self.output, "w+")
        for i in range(self.cores):
            with open(self.output + '_{}'.format(i), "r") as f:
                for line in f:
                    out.write(line)
            if delete_output:
                os.remove(self.output + '_{}'.format(i))

        # delete split input files
        for i in range(self.cores):
            os.remove(self.input + '_{}'.format(i))


    def checker(self, id, slang_dict, stop_words, emoji_dict, restart=False):
        # get enchant dict, instantiate MMSt
        broker = enchant.Broker()
        d = broker.request_dict("en_US")
        g = MMST(d, slang_dict, stop_words, emoji_dict)

        # open input file, get output file path
        input = open(self.input + '_{}'.format(id), "r")
        output_path = self.output + '_{}'.format(id)

        # check how many lines have already been processed in a previous run
        if restart or not os.path.isfile(output_path):
            start = 0
            open_mode = "w+"
        else:
            start = sum(1 for line in open(output_path))
            open_mode = "a+"

        # process remaining lines
        with open(output_path, open_mode) as f:
            for line in islice(input, start, None):
                try:
                    tmp = g.input_sentence(line, self.load, verbose=False)
                    f.write(tmp)
                except IndexError:
                    print("ERROR: " + line)


    def run(self):
        super().run()

        # split input file into num_core many files
        self.prep_input()

        # get slang, stop words and emoticon dict
        # NOTE: For now, we load these dicts here (shared between threads)
        # but we load one enchant dict per thread. This has concurrency reasons.
        # We could load these dicts also one per thread, but we need to do
        # some adjustements.
        dict = Dict()
        slang_dict = dict.get_slang()
        stop_words = dict.get_stopwords()
        emoji_dict = dict.get_emoticon()

        # process input files
        ts = [threading.Thread(target=self.checker, args=(i, slang_dict, stop_words, emoji_dict)) for i in range(self.cores)]

        for t in ts:
            t.start()

        for t in ts:
            t.join()

        # merge num_core output files into one, delete the split files
        self.merge_and_delete()
