from flask import Flask, jsonify, request

from core import TopicModeler

app = Flask(__name__)

topic_modeler = TopicModeler(None, None) # JUST AS MOCK! WILL GIVE ERROR BECAUSE OF NONES!


@app.route('/topic_model', methods=['POST'])
def topic_model():
    content = request.json
    topics = topic_modeler.topic_models(content)
    return jsonify(topics)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True)