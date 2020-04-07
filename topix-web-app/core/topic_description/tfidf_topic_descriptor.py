from .interfaces import TopicDescriptor
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFTopicDescriptor(TopicDescriptor):
    def __init__(self, min_df, max_df_ratio, tokenizer, topn_words_per_topic=10):
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.tokenizer = tokenizer
        self.topn_words_per_topic = topn_words_per_topic

    def generate_descriptions(self, topics: List[List[str]]) -> List[str]:
        all_docs = []
        for topic in topics:
            all_docs.extend(topic)

        corpus_vectorizer, id2token = learn_corpus(
                                        self.min_df,
                                        self.max_df_ratio,
                                        all_docs,
                                        self.tokenizer)

        topics_description = []
        for topic in topics:
            top_words_per_topic = get_top_words(
                                    corpus_vectorizer, 
                                    id2token,
                                    topic,
                                    self.topn_words_per_topic)

            topics_description.append(top_words_per_topic)

        return topics_description


def get_top_words(corpus_vectorizer, id2token, topic, topn_words_per_topic):
    # create a single document of topic's docs - tf for whole topic
    combined_doc = ' '.join(topic)

    tfidf = corpus_vectorizer.transform([combined_doc])

    # sort by tfidf
    sorted_indices = tfidf.data.argsort()[-topn_words_per_topic:][::-1]

    # tokens
    topn_tokens_id = tfidf.indices[sorted_indices]
    topn_tokens = [id2token[token_id] for token_id in topn_tokens_id]

    return topn_tokens


def learn_corpus(min_df, max_df_ratio, docs, tokenizer):
    # adjust cut-offs for too small topic
    min_df = min(len(docs), min_df)
    max_df = max_df_ratio if len(docs) * max_df_ratio >= min_df else 1.0

    # learn idf over all docs
    corpus_vectorizer = TfidfVectorizer(lowercase=False, 
                                        tokenizer=tokenizer,
                                        token_pattern=None,
                                        min_df=min_df,
                                        max_df=max_df)

    corpus_vectorizer.fit(docs)

    # token mapping
    id2token = [None] * len(corpus_vectorizer.vocabulary_)
    for token, token_id in corpus_vectorizer.vocabulary_.items():
        id2token[token_id] = token

    return corpus_vectorizer, id2token
