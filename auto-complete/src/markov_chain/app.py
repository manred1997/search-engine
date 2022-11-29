import pickle
import collections

from utils import chunks, get_possible_word_from_fat_finger


class AutoComplete_MC(object):
    def __init__(self, model_path=None):
        WORDS = []

        WORD_TUPLES = []

        self.WORDS_MODEL = {}

        self.WORD_TUPLES_MODEL = {}
        if model_path is not None:
            model = pickle.load(open(model_path, 'rb'))
            self.WORD_TUPLES_MODEL = model['word_tuples_model']
            self.WORDS_MODEL = model['words_model']
        else:
            corpus = []
            with open('../data/corpus.txt', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    corpus.append(line.strip())
            for i in corpus:
                if len(i) < 5:  continue
                WORDS.extend(i.split())
            self.WORDS_MODEL = collections.Counter(WORDS)
            WORD_TUPLES = list(chunks(WORDS, 2))
            self.WORD_TUPLES_MODEL = {first:collections.Counter() for first, second in WORD_TUPLES}
            for tup in WORD_TUPLES:
                try:
                    self.WORD_TUPLES_MODEL[tup[0]].update([tup[1]])
                except:
                    # hack-y fix for uneven # of elements in WORD_TUPLES
                    pass

    def predict(self, word, top_k=10):
        word = word.split()
        if len(word) < 2:
            first_word, second_word = word[-1], word[-1]
        else:
            first_word, second_word = word[-2], word[-1]

        possible_second_words = get_possible_word_from_fat_finger(second_word)
        possible_second_words.append(second_word)


        probable_words = {w:c for w, c in self.WORD_TUPLES_MODEL[first_word.lower()].items()
                                            for sec_word in possible_second_words if w.startswith(sec_word)}
        return collections.Counter(probable_words).most_common(top_k)
