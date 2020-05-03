from flask import Flask, jsonify, request
from core import BertClusterer
from core import TopicModeler
from core import TFIDFTopicDescriptor
from nltk.tokenize import word_tokenize

app = Flask(__name__)


topic_modeler = TopicModeler(clusterer=BertClusterer(), descriptor=TFIDFTopicDescriptor(
    tokenize=word_tokenize,
    min_df=3,
    max_df_ratio=0.5,
    topn_words_per_topic=6,
    phrases_min_count=5,
    phrases_threshold=10
))


@app.route('/topic_model', methods=['POST'])
def topic_model():
    content = request.json
    topics = topic_modeler.topic_models(content, depth=1)
    return jsonify(topics)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
