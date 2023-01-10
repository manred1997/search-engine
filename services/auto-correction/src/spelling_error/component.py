import re
import random

from src.spelling_error.typographic import (
    TELEX_ERROR,
    VNI_ERROR,
    FAT_FINGER_ERROR
)
from src.spelling_error.orthographic import (
    HOMOPHONE_LETTER_ERROR,
    HOMOPHONE_SINGLE_WORD_ERROR,
    HOMOPHONE_DOUBLE_WORD_ERROR

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

    if not set(word).intersection(set(ALPHABET)):
        return [word]

    while True:
        position_error = random.randint(0, len(word)-1)
        if word[position_error] in ALPHABET:
            break
    
    possible_words = [word[:position_error] + char + word[position_error+1:]
                        for char in FAT_FINGER_ERROR[word[position_error]]]
    
    return possible_words

def get_possible_word_from_telex_error(word):
    """Create all possible word from telex error
    :param word
    :return possible word list from telex error
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
    """Create all possible word from vni error
    :param word
    :return possible word list from vni error
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
    """Create all possible word from edit error
    :param word
    :return possible word list from edit error
    """
    if len(word) == 1:
         return [word]

    type_edit = random.choices(['delete', 'insert', 'permute', 'replace'], weights=(35, 17, 39, 39), k=1)[0]

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
        possible_words = [random.choice(get_possible_word_from_fat_finger_error(word))]
    return possible_words

def get_possible_word_from_accent_error(word):
    """Create all possible word from accent error
    :param word
    :return possible word list from accent error
    """
    word = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', word)
    word = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', word)
    word = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', word)
    word = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', word)
    word = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', word)
    word = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', word)
    word = re.sub(r'[ìíịỉĩ]', 'i', word)
    word = re.sub(r'[ÌÍỊỈĨ]', 'I', word)
    word = re.sub(r'[ùúụủũưừứựửữ]', 'u', word)
    word = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', word)
    word = re.sub(r'[ỳýỵỷỹ]', 'y', word)
    word = re.sub(r'[ỲÝỴỶỸ]', 'Y', word)
    word = re.sub(r'[Đ]', 'D', word)
    word = re.sub(r'[đ]', 'd', word)
    return [word]

def get_possible_word_from_miss_space_error(pairwords):
    """Create all possible word from miss space error
    :param pair words
    :return possible word list from miss space error
    """

    pairwords = pairwords.strip().split()
    return "".join(pairwords)


def get_possible_word_from_split_error(word):
    """Create all possible word from accent error
    :param word
    :return possible word list from accent error
    """
    if len(word) < 3:
        return word
    
    list_char = list(word)
    index_split = random.randint(1, len(list_char)-1)

    word = "".join(list_char[:index_split]) + " " + "".join(list_char[index_split:])
    return word

def get_homophone_letter_error(word):
    """Create all possible word from homophone letter error
    :param word
    :return possible word list from homophone letter error
    """

    if not set(word).intersection(set(HOMOPHONE_LETTER_ERROR.keys())):
        return [word]


    while True:
        position_error = random.randint(0, len(word)-1)
        if word[position_error] in HOMOPHONE_LETTER_ERROR:
            break
    
    possible_words = [word[:position_error] + char + word[position_error+1:]
                        for char in HOMOPHONE_LETTER_ERROR[word[position_error]]]
    
    possible_words.append(word)
    
    return possible_words


def get_homophone_single_word_error(word):
    """Create all possible word from homophone single word error
    :param word
    :return possible word list from homophone single word error
    """
    return HOMOPHONE_SINGLE_WORD_ERROR.get(word, [word])

def get_homophone_double_word_error(double_word):
    """Create all possible word from homophone double word error
    :param word
    :return possible word list from homophone double word error
    """
    pass
    return

# TODO: Hypernation errors, Capitalisation errors (e.g Apple TM -> Apple branch), 