import collections
# from operator import itemgetter
import time
from vncorenlp import VnCoreNLP
annotator = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g') 

def segment_sentence(text):
    text = annotator.tokenize(text)
    sentences = []
    for words in text:
        words = " ".join(words)
        # words = words.replace('_', ' ')
        sentences.append(words)
    return " ".join(sentences)

def parse(query):
    query = query.split()
    if len(query) == 1:
        prefix = ['']
    else:
        prefix = query[:-1]
    suffix = [query[-1]]
    return prefix, suffix

def intersection(lst_of_lst):
    return list(set.intersection(*map(set,lst_of_lst)))


def Union(lst_of_list):
    final_list = list(set().union(*lst_of_list))
    return final_list

class Node:
    def __init__(self):
        self.children = {}
        self.word = None
    
    def insert(self, word):
        node = self
        for letter in word:
            if letter not in node.children:
                node.children[letter] = Node()
            node = node.children[letter]
        node.word = word
        return node
    def traverse(self, query):
        node = self
        for letter in query:
            child = node.children.get(letter)
            if child:
                node = child
            else:
                break

        return node
    def __repr__(self):
        return f'< children: {list(self.children.keys())}, word: {self.word} >'
    def get_descendants_nodes(self):
        que = collections.deque()
        for letter, child_node in self.children.items():
            que.append((letter, child_node))
        while que:
            letter, child_node = que.popleft()
            if child_node.word:
                yield child_node
            for letter, grand_child_node in child_node.children.items():
                que.append((letter, grand_child_node))

class ConjuctiveSearch(object):
    def __init__(self, vocabulary_file, inverted_file, corpus_file):
        data = []
        with open(vocabulary_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data.append(line.strip())

        inverted_list = []
        with open(inverted_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split()
                
                inverted_list.append(list(map(int, line[1:])))


        self.corpus_docid = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.corpus_docid.append(line.strip())


        self.root = Node()
        WORDS = data
        for word in WORDS:
            self.root.insert(word)

        self.term2id = {}
        self.term2inverted = {}
        id = 1
        for term, ivt in zip(data, inverted_list):
            self.term2id[term] = id
            self.term2inverted[term] = ivt
            id += 1
    
    def get_candidate_suffix(self, suffix):
        candidate_id = []
        candidate_term = []

        inverted_lst = []

        found_node = self.root.traverse(query=suffix)
        for i in list(found_node.get_descendants_nodes()):
            term = i.word
            candidate_term.append(term)
            candidate_id.append(self.term2id[term])
            inverted_lst.append(self.term2inverted[term])
        l = min(candidate_id)
        r = max(candidate_id)
        return [l, r], candidate_term, Union(inverted_lst)
    
    def parsePrefix(self, query):
        query = query.split()
        if len(query) == 1:
            prefix = ['']
        else:
            prefix = ' '.join(query[:-1])
            prefix = segment_sentence(prefix)
            prefix = prefix.split()
        suffix = [query[-1]]
        return prefix, suffix
    
    def IntersectionIterator(self, prefix):
        lst_of_lst = []
        try:
            for i in prefix:
                try:
                    lst_of_lst.append(self.term2inverted[i.replace('_', ' ')])
                except:
                    print(i)
            # print(lst_of_lst)
            return intersection(lst_of_lst=lst_of_lst)
        except:
            return []

    
    def run(self, query):
        prefix, suffix = self.parsePrefix(query)
        [l, r], candidate_term, union_suffix_inverted_lst = self.get_candidate_suffix(suffix[0])
        # print(union_suffix_inverted_lst)
        # check complete suffix #TODO
        if len(prefix) < 3:
            return candidate_term
        x = self.IntersectionIterator(prefix)
        # print(x)
        if not x:
            return candidate_term
        docid = intersection(lst_of_lst=[x, union_suffix_inverted_lst])
        if not docid:
            return candidate_term
        return list(map(self.corpus_docid.__getitem__, docid))
    
    
