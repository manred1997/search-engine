import collections


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


def main(query):
    data = []
    with open('../../data/symptom.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.strip())
    root = Node()
    WORDS = data
    for word in WORDS:
        root.insert(word)
    
    found_node = root.traverse(query=query)
    for i in list(found_node.get_descendants_nodes()):
        print(i.word)

if __name__ == "__main__":
    main('đau')

# stage - 1: tri - tree
# stage -2 : ranking

# dataset 

# latency < 200ms
