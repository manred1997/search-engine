NEARBY_KEYS = {
    'a': 'qwsz',
    'b': 'vghn',
    'c': 'xdfv',
    'd': 'erfcxs',
    'e': 'rdsw',
    'f': 'rtgvcd',
    'g': 'tyhbvf',
    'h': 'yujnbg',
    'j': 'uikmnh',
    'k': 'iolmj',
    'l': 'opk',
    'm': 'njk',
    'n': 'bhjm',
    'o': 'iklp',
    'p': 'ol',
    'q': 'wa',
    'r': 'edft',
    's': 'wedxza',
    't': 'rfgy',
    'u': 'yhji',
    'v': 'cfgb',
    'w': 'qase',
    'x': 'zsdc',
    'y': 'tghu',
    'z': 'asx'
    }

def chunks(l, n):
    for i in range(0, len(l) - n + 1):
        yield l[i:i+n]

def get_possible_word_from_fat_finger(word):
    #Hidden step
    possible_words = [word[:-1]+char
                             for char in NEARBY_KEYS[word[-1]]
                             if len(word) > 2]

    possible_words.append(word)
    return possible_words