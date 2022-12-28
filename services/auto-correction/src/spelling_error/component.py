import random

from src.spelling_error.typographic import (
    TELEX_ERROR,
    VNI_ERROR,
    FAT_FINGER_ERROR
)

from src.utils.constants import (
    ALPHABET,
    ALPHABET_LOWERCASE,
    ALPHABET_UPPERCASE,
    DIGITS,
    PUNCTUATION,
)

def get_possible_word_from_fat_finger_error(word):
    """Create all possible words from fat-finger error
    :param word
    :return possible words list from fat-finger error
    """
    while True:
        position_error = random.randint(0, len(word)-1)
        if word[position_error] in ALPHABET:
            break
    
    possible_words = [word[:position_error] + char + word[position_error+1:]
                        for char in FAT_FINGER_ERROR[word[position_error]]]

    possible_words.append(word)
    
    return possible_words

def get_possible_word_from_telex_error(word):
    """Create all possible words from telex error
    :param word
    :return possible words list from telex error
    """
    possible_words = ''
    for char in word:
        if char not in TELEX_ERROR.keys():
            possible_words += char
        else:
            if random.gauss(0, 1) < 1 or len(TELEX_ERROR[char]) == 1:
                possible_words += TELEX_ERROR[char][0]
            else:
                possible_words += TELEX_ERROR[char][1]
    
    return [possible_words]

def get_possible_word_from_vni_error(word):
    """Create all possible words from vni error
    :param word
    :return possible words list from vni error
    """
    possible_words = ''
    for char in word:
        if char not in VNI_ERROR.keys():
            possible_words += char
        else:
            if random.gauss(0, 1) < 1 or len(VNI_ERROR[char]) == 1:
                possible_words += VNI_ERROR[char][0]
            else:
                possible_words += VNI_ERROR[char][1]
    
    return [possible_words]
