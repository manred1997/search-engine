from flask import Flask, request

# from spellchecker import SpellChecker

app = Flask(__name__)


@app.route("/spell_check", methods=["POST"])
def spell_check():
    text = request.json["text"]
    # spell = SpellChecker()
    # corrected_text = spell.correction(text)
    corrected_text = None
    return {"corrected_text": corrected_text}


if __name__ == "__main__":
    app.run()
