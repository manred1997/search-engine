from conjuctive_search.search import ConjuctiveSearch


from flask import Flask, request, jsonify, render_template
import random
import time


app = Flask(__name__)

search = ConjuctiveSearch('./build_db/completions.dict', './build_db/completions.inverted', './build_db/completions_collect_word_level.txt')
# print(search.run('thụ tinh nhân tạ'))

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/autocomplete', methods=["POST"])
def autocomple():
    start = time.time()
    query = request.form.get('keyword')
    response = []
    
    try:
        predict = search.run(query)
        for p in predict:
            response.append(
                {
                    "name": p
                }
            )
    except Exception as e:
        print('Fail: {}'.format(str(e)))
        #logging
    print(f'Time inference: {time.time() - start}')
    return jsonify(response)




if __name__ == "__main__":
    app.run(debug=True)