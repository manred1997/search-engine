import connexion
import six

from swagger_server.models.health_check import HealthCheck  # noqa: E501
from swagger_server.models.prefix_retrieve import PrefixRetrieve  # noqa: E501
from swagger_server import util

import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../../..'))
print(sys.path)
from src.utils.utils import _read_text_file
from src.retrieval.prefix.trie import Trie

from src.retrieval.config import settings

query_logs = _read_text_file(settings.default.retrieval.prefix.FOLDER_QUERY_LOGS)
trie = Trie()
trie.formTrie(query_logs)


def find_completions_by_charater(body):  # noqa: E501
    """Finds completion candidates

    Retrieve completion candidates by keystrokes of user # noqa: E501

    :param body: 
    :type body: dict | bytes

    :rtype: PrefixRetrieve
    """
    if connexion.request.is_json:
        keystroke = connexion.request.get_json()
        trie.printAutoSuggestions(keystroke)
        # body = str.from_dict()  # noqa: E501
    return 'do some magic!'


def health_check():  # noqa: E501
    """Health check

     # noqa: E501


    :rtype: HealthCheck
    """
    return 'do some magic!'
