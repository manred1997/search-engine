from sanic import Sanic
from sanic.response import (
    json,
    text
)

from config import settings

from utils.utils import _read_text_file

from prefix.trie import Trie

query_logs = _read_text_file(settings.default.retrieval.prefix.FOLDER_QUERY_LOGS)

# Creat Trie structure data
trie = Trie()
trie.formTrie(query_logs)

app = Sanic("search-engine-retrieval")

@app.route('/health_check')
async def heath_check(request):
    return text('OK')

@app.get('/getComps')
async def get_completion(request):
    print(request)
    # completions = trie.printAutoSuggestions(request)

if __name__ == '__main__':
    app.run()
