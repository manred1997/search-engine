import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../..'))
# print(sys.path)



from src.utils.utils import _read_text_file
from src.retrieval.prefix.trie import Trie

from src.retrieval.config import settings

query_logs = _read_text_file(settings.default.retrieval.prefix.FOLDER_QUERY_LOGS)
trie = Trie()
trie.formTrie(query_logs)

from flask import Flask

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, f'{__name__}.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/health_check')
    def health_check():
        return 'Ô tô kê!'

        

    return app