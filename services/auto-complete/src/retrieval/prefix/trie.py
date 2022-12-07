

class TrieNode(object):
    def __init__(self) -> None:
        # Initialising one node for trie
        self.children = {}
        self.last = False

class Trie():
    def __init__(self) -> None:
        # Initialising the trie structure.
        self.root = TrieNode()

    def _insert(self, query):
        # Inserts a query into trie if it does not exist already.
        # And if the key is a prefix of the trie node, just marks it as leaf node.

        node = self.root
        for c in query:
            if not node.children.get(c):
                node.children[c] = TrieNode()
            
            node = node.children[c]
        
        node.last = True
    
    def formTrie(self, queries):
        # Forms a trie structure with the given set of strings if it does not exists already else it merges the key into it by extending the structure as required
        for query in queries:
            self._insert(query)

    
    def suggestionsRec(self, node, word):
        # Method to recursively traverse the trie and return a whole word
        if node.last:
            print(word)
        
        for c, n in node.children.items():
            self.suggestionsRec(n, word + c)
    
    def printAutoSuggestions(self, query):
        # Returns all the words in the trie whose common
        # prefix is the given key thus listing out all
        # the suggestions for autocomplete.
        node = self.root

        for c in query:
            # no string in the Trie has this prefix
            if not node.children.get(c):
                return 0
            node = node.children[c]
        
        # If prefix is present as a word, but
        # there is no subtree below the last
        # matching node.

        if not node.children:
            return -1
 
        self.suggestionsRec(node, query)
        return 1
