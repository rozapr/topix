from .interfaces import TopicClusterer
from typing import List
from transformers import BertModel, BertTokenizer
from .k_means import k_means
from datetime import datetime
import pandas as pd
import numpy as np
import torch

PATH_TO_MODEL = '/datadrive/topix/exploration/bert/checkpoint-2216000/'
MIN_WORDS_IN_SENTENCE = 3
MAX_TOKENS_SIZE = 512


class BertClusterer(TopicClusterer):
    _tokenizer = BertTokenizer.from_pretrained(PATH_TO_MODEL)
    _embedding_model = BertModel.from_pretrained(PATH_TO_MODEL)
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self):
        self._embedding_model = self._embedding_model.to(self._device)

    def get_document_embeddings(self, documents: List[str]) -> List[np.ndarray]:
        document_embeddings = []
        len_docs = len(documents)
        print(f'{datetime.now()} num of docs: {len_docs}') # TODO - print it to a logger
        for index, document in enumerate(documents):
            sentences = document.split('\n')
            sentence_embeddings = [self.get_sentence_embedding(sentence) for sentence in sentences]
            sentence_embeddings = np.array([sentence for sentence in sentence_embeddings if sentence is not None])
            mean_document_embedding = np.mean(sentence_embeddings, axis=0) # the document embedding is the average of all the sentence embeddings
            document_embeddings.append(mean_document_embedding)
            if index % 100 == 0: # for logging purposes
                present = round(100 * index / len_docs)
                print(f'{datetime.now()} finished {present}%') # TODO - print it to a logger
        print(f'{datetime.now()} finished embedding docs') # TODO - print it to a logger
        return document_embeddings

    def get_sentence_embedding(self, sentence: str) -> List[float]:
        if sentence.strip() == '' or len(sentence.split()) < MIN_WORDS_IN_SENTENCE:
            return
        sentence = f'[CLS] {sentence} [SEP]'
        tokens = self._tokenizer.encode(sentence)
        if len(tokens) > MAX_TOKENS_SIZE:
            tokens = tokens[:MAX_TOKENS_SIZE] # if the sentence is too big, we takes only the first MAX_TOKENS_SIZE tokens
        input_ids = torch.tensor(tokens).unsqueeze(0)  # batch size is 1
        input_ids = input_ids.to(self._device)
        outputs = self._embedding_model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state, the token embeddings, is the first element of the output tuple
        mean_of_words = torch.mean(last_hidden_states, axis=1).squeeze(0).tolist() # the sentence embedding is the average of all the bert token embeddings
        return mean_of_words

    def cluster(self, documents: List[str]) -> List[List[str]]:
        document_embeddings = self.get_document_embeddings(documents)
        cluster_assignment = k_means(document_embeddings)
        df = pd.DataFrame({'vectors': document_embeddings, 'cluster': cluster_assignment, 'text': documents})
        cluster_to_docs = df.groupby('cluster')['text'].apply(list).reset_index(name='docs')
        return cluster_to_docs['docs'].tolist()

