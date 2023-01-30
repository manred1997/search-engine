import os
import sys

AUTO_COMPLETE_PATH = os.environ.get("AUTO_COMPLETE_PATH")
SEARCH_ENGINE_PATH = os.environ.get("SEARCH_ENGINE_PATH")

sys.path.append(AUTO_COMPLETE_PATH)

import connexion
from settings.config import settings
from src.retrieval.prefixtrie import Trie
from src.utils.utils import _read_text_file

query_logs = _read_text_file(
    os.path.join(
        SEARCH_ENGINE_PATH, settings.default.retrieval.prefix.FOLDER_QUERY_LOGS
    )
)
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
        candidate_startsWith_prefix = trie.query(keystroke)
        candidate_startsWith_prefix = [c[0] for c in candidate_startsWith_prefix]
        # body = str.from_dict()  # noqa: E501
    return candidate_startsWith_prefix


def health_check():  # noqa: E501
    """Health check

     # noqa: E501


    :rtype: HealthCheck
    """
    return "do some magic!"
