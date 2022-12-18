import string

def remove_accent(text):
    """
    clean accent
    params:
        text has accent
    return:
        text non accent
    """
    list_a = ["à","á","ạ","ả","ã","â","ầ","ấ","ậ","ẩ","ẫ","ă","ằ","ắ","ặ","ẳ","ẵ"]
    list_A = ["À","Á","Ạ","Ả","Ã","Â","Ầ","Ấ","Ậ","Ẩ","Ẫ","Ă","Ằ","Ắ","Ặ","Ẳ","Ẵ"]
    list_e = ["è","é","ẹ","ẻ","ẽ","ê","ề","ế","ệ","ể","ễ"]
    list_E = ["È","É","Ẹ","Ẻ","Ẽ","Ê","Ề","Ế","Ệ","Ể","Ễ"]
    list_i = ["ì","í","ị","ỉ","ĩ"]
    list_I = ["Ì","Í","Ị","Ỉ","Ĩ"]
    list_o = ["ò","ó","ọ","ỏ","õ","ô","ồ","ố","ộ","ổ","ỗ","ơ","ờ","ớ","ợ","ở","ỡ"]
    list_O = ["Ò","Ó","Ọ","Ỏ","Õ","Ô","Ồ","Ố","Ộ","Ổ","Ỗ","Ơ","Ờ","Ớ","Ợ","Ở","Ỡ"]
    list_u = ["ù","ú","ụ","ủ","ũ","ư","ừ","ứ","ự","ử","ữ"]
    list_U = ["Ù","Ú","Ụ","Ủ","Ũ","Ư","Ừ","Ứ","Ự","Ử","Ữ"]
    list_y = ["ỳ","ý","ỵ","ỷ","ỹ"]
    list_Y = ["Ỳ","Ý","Ỵ","Ỷ","Ỹ"]
    list_d = ["đ"]
    list_D = ["Đ"]

    return_text = []
    for c in text:
        if c in list_a:
            return_text.append('a')
        elif c in list_A:
            return_text.append('A')
        elif c in list_e:
            return_text.append('e')
        elif c in list_E:
            return_text.append('E')
        elif c in list_i:
            return_text.append('i')
        elif c in list_I:
            return_text.append('I')
        elif c in list_o:
            return_text.append('o')
        elif c in list_O:
            return_text.append('O')
        elif c in list_u:
            return_text.append('u')
        elif c in list_U:
            return_text.append('U')
        elif c in list_y:
            return_text.append('y')
        elif c in list_Y:
            return_text.append('Y')
        elif c in list_d:
            return_text.append('d')
        elif c in list_D:
            return_text.append('D')
        else:
            return_text.append(c)
    
    return "".join(return_text)


def remove_punctuation(text):
    """
    clean punctuation
    params:
        text has punctuation
    return:
        text non punctuation
    """
    return text.translate(str.maketrans('', '', string.punctuation))