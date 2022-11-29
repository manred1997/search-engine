from trie_tree.app import Node
from markov_chain.app import AutoComplete_MC
import time

from utils import get_possible_word_from_fat_finger

def main(query):

    result = []

    markov_chain = AutoComplete_MC('./markov_chain/models_compressed.pkl')

    data = []
    # with open('../data/symptom.txt', 'r', encoding='utf-8') as f:
    with open('./notebook/key_words_ehr.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.strip())
    root = Node()
    WORDS = data
    for word in WORDS:
        root.insert(word)
    
    # check fat-finger-error

    
    found_node = root.traverse(query=query)
    start_time = time.time()
    for i in list(found_node.get_descendants_nodes()):
        print(i.word)
        relevant_word = i.word.split()
        try:
            first_word, second_word = relevant_word[0], relevant_word[1]
            score = markov_chain.WORD_TUPLES_MODEL[first_word][second_word]
        except:
            score = 0
        result.append((i.word, score))
    result = sorted(result, key=lambda tup: tup[1], reverse=True)
    print(result)
    print("--- %s mili-seconds ---" % ((time.time() - start_time)*1000))
        

if __name__ == "__main__":
    main('albumin n')