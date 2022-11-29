from algoliasearch.search_client import SearchClient


from constants import FILE_MAPPER
from utils import read_json



ALGOLIA_APP_ID = 'IPCD9865X5'
ALGOLIA_API_KEY = '46994d460c978a227a2d2c6ef6aefca3'
ALGOLIA_INDEX_NAME = 'autocomplete_18_4_2022_symptoms'


client = SearchClient.create(ALGOLIA_APP_ID, ALGOLIA_API_KEY)

# mapper = read_json(FILE_MAPPER)
# config_synonym = {
#         'forwardToReplicas': True,
#         'replaceExistingSynonyms': True
#     }

index = client.init_index(ALGOLIA_INDEX_NAME)
import json

# # index.save_synonym(
# #     mapper,config_synonym
# #     )

# searchParameters = {} ##type: key/value mapping default: No search parameters Optional
# requestOptions = {} ##type: key/value mapping default: No search parameters Optional
res = index.search(
    'dau d',
    # searchParameters,
    # requestOptions
)
print(res)
# client.get_logs()
# # with open('./response_template.json', 'w', encoding='utf-8') as f:
# #     json.dump(res, f, ensure_ascii=False)
# # for obj in objects:
# #     print(obj['name'])

# objects = index.search('đau châ')['hits']
# for obj in objects:
#     print(obj['name'])

# objects = index.search('đau chi duo')['hits']
# for obj in objects:
#     print(obj['name'])

# import time

# from algoliasearch.search_client import SearchClient

# res = client.add_api_key(['search'])
# print(res['key'])


# # generate a public API key that is valid for 1 hour:
# valid_until = int(time.time())  + 3600
# public_key = SearchClient.generate_secured_api_key(
#     'YourSearchOnlyApiKey',
#     {'validUntil': valid_until}
# )
# print(public_key)