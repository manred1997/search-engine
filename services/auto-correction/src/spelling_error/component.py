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
    ALPHABET_VN,
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


def get_possible_word_from_edit_error(word):
    """Create all possible words from edit error
    :param word
    :return possible words list from edit error
    """
    if len(word) == 1:
         return [word]

    type_edit = random.choices(['delete', 'insert', 'permute', 'replace'], weights=(35, 17, 39, 9), k=1)[0]

    position_error = random.randint(0, len(word)-1)

    if type_edit == 'delete':  # Delete Error
        possible_words = [word[:position_error] + word[position_error+1:]]
    elif type_edit == 'insert':    # Insert Error
        possible_words = [word[:position_error] + random.choice(ALPHABET_VN) + word[position_error:]]
    elif type_edit == 'permute': # Permute Error
        if position_error == 0:
            possible_words = [word[position_error+1] + word[position_error] + word[position_error+2:]]
        elif position_error == len(word) - 1:
            possible_words = [word[:position_error-1] + word[position_error] + word[position_error-1]]
        else:
            if random.gauss(0, 1) < 0:
                possible_words = [word[:position_error-1] + word[position_error] + word[position_error-1] + word[position_error+1:]]
            else:
                possible_words = [word[:position_error] + word[position_error+1] + word[position_error] + word[position_error+2:]]
    else: # Replace Error/ Fat-Finger Error
        possible_words = random.choice(get_possible_word_from_fat_finger_error(word)[:-1])

    return possible_words