from .interfaces import TopicDescriptor
from typing import List, Callable
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.phrases import Phrases, Phraser
import itertools


class TFIDFTopicDescriptor(TopicDescriptor):

    _tokenize: Callable[[str], List[str]]
    _min_df: int
    _max_df_ratio: float
    _topn_words_per_topic: int
    _phrases_min_count: int
    _phrases_threshold: float

    def __init__(   self,
                    tokenize: Callable[[str], List[str]],
                    min_df: int,
                    max_df_ratio: float,
                    topn_words_per_topic: int,
                    phrases_min_count: int,
                    phrases_threshold: float):
        self._tokenize = tokenize
        self._min_df = min_df
        self._max_df_ratio = max_df_ratio
        self._topn_words_per_topic = topn_words_per_topic
        self._phrases_min_count = phrases_min_count
        self._phrases_threshold = phrases_threshold

    def generate_descriptions(self, topics: List[List[str]]) -> List[str]:
        preprocessed_topics = preprocess_topics(
                    topics,
                    self._tokenize,
                    self._phrases_min_count,
                    self._phrases_threshold)

        all_docs = list(itertools.chain.from_iterable(preprocessed_topics))

        corpus_vectorizer, id2token = learn_corpus(
                                        self._min_df,
                                        self._max_df_ratio,
                                        all_docs)

        topics_description = []
        for topic in preprocessed_topics:
            top_words_per_topic = get_top_words(
                                    corpus_vectorizer, 
                                    id2token,
                                    topic,
                                    self._topn_words_per_topic)

            topics_description.append(top_words_per_topic)

        return topics_description


def get_top_words(  corpus_vectorizer: TfidfVectorizer,
                    id2token: List[str],
                    topic: List[List[str]],
                    topn_words_per_topic: int) -> List[List[str]]:
    # create a single document of topic's docs - tf for whole topic
    combined_doc = ' '.join(itertools.chain.from_iterable(topic))

    tfidf = corpus_vectorizer.transform([combined_doc])

    # sort by tfidf
    sorted_indices = tfidf.data.argsort()[-topn_words_per_topic:][::-1]

    # tokens
    topn_tokens_id = tfidf.indices[sorted_indices]
    topn_tokens = [id2token[token_id] for token_id in topn_tokens_id]

    # fix after pharsing
    topn_words = [token.replace('_', ' ') for token in topn_tokens]

    return topn_words


def preprocess_topics(  topics: List[List[str]],
                        tokenize: Callable[[str], List[str]],
                        phrases_min_count: int,
                        phrases_threshold: float) -> List[List[List[str]]]:
    tokenized_topics = []
    for topic in topics:
        tokenized_topics.append([tokenize(doc) for doc in topic])

    all_docs = itertools.chain.from_iterable(tokenized_topics)

    phrases = Phrases(all_docs, min_count=phrases_min_count, threshold=phrases_threshold)
    bigram_phraser = Phraser(phrases)

    preprocessed_topics = []
    for topic in tokenized_topics:
        preprocessed_topic = [bigram_phraser[doc] for doc in topic]
        preprocessed_topics.append(preprocessed_topic)

    return preprocessed_topics


def learn_corpus(   min_df: int,
                    max_df_ratio: float,
                    docs: List[List[str]]) -> [TfidfVectorizer, List[str]]:
    # adjust cut-offs for too small topic
    min_df = min(len(docs), min_df)
    max_df = max_df_ratio if len(docs) * max_df_ratio >= min_df else 1.0

    # learn idf over all docs
    corpus_vectorizer = TfidfVectorizer(lowercase=False, 
                                        tokenizer=lambda t: t.split(),
                                        token_pattern=None,
                                        min_df=min_df,
                                        max_df=max_df)

    corpus_vectorizer.fit([' '.join(doc) for doc in docs])

    # token mapping
    id2token = [None] * len(corpus_vectorizer.vocabulary_)
    for token, token_id in corpus_vectorizer.vocabulary_.items():
        id2token[token_id] = token

    return corpus_vectorizer, id2token
