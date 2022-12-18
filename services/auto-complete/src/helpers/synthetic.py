import re

def suffixSynthesize(query: str, lowercase=False) -> list:
    """Create all possible suffixes
    :param query
    :return suffixes list

    Example:
    Query "How to cook CHICKEN" becomes:
        - How to cook CHICKEN
        - to cook CHICKEN
        - cook CHICKEN
        - CHICKEN
    Creating 4 suffixes.
    """

    suffixes = []

    if lowercase:
        query = query.lower()
    

    # replace punctuation with white space
    query = re.sub('[\.,?!@#$%^&*()_\-=+\[\]{}:;<>~`\/]', ' ', query)

    words = query.split()
    for index in range(0, len(words)):
        suffix =  " ".join(words[index: ])
        suffixes.append(suffix)
    return suffixes

def prefixSyntheszie(query: str, prefixLength=3, termPrefix=1, lowerCase=False) -> list:
    """Create all possible prefixes
    :param query. Query of user
    :param prefixLength. Minimum no.Character of prefix query
    :param termPrefix. Minimum term in prefix query
    :param lowerCase. Query in lower mode
    :return prefixes list

    Example:
    Query "How to cook CHICKEN" becomes:
        - How
        - How
        - How t
        - ...
        - How to cook chicken
    Creating 17 prefixes.
    """
    prefixes = []

    if lowerCase:
        query = query.lower()
    
    # replace punctuation with white space
    query = re.sub('[\.,?!@#$%^&*()_\-=+\[\]{}:;<>~`\/]', ' ', query)
    
    if termPrefix != 0:
        prefixWord = " ".join(query.split()[:termPrefix])
        prefixLength = len(prefixWord)

    for index in range(prefixLength, len(query)+1):
        prefix = query[:index]
        prefixes.append(prefix)
    return prefixes