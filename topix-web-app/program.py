from flask import Flask, jsonify, request
from core import BertClusterer
from core import TopicModeler

app = Flask(__name__)

# JUST AS MOCK! WILL GIVE ERROR BECAUSE OF NONES!
topic_modeler = TopicModeler(clusterer=BertClusterer(), descriptor=None)


@app.route('/topic_model', methods=['POST'])
def topic_model():
    content = request.json
    topics = topic_modeler.topic_models(content)
    return jsonify(topics)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
