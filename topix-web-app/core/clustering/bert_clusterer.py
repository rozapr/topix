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
        print(f'{datetime.now()} num of docs: {len(documents)}')
        for index, document in enumerate(documents):
            sentences = document.split('\n')
            sentence_embeddings = [self.get_sentence_embeddings(sentence) for sentence in sentences]
            sentence_embeddings = np.array([sentence for sentence in sentence_embeddings if sentence is not None])
            mean_document_embedding = np.mean(sentence_embeddings, axis=0)
            document_embeddings.append(mean_document_embedding)
            if index % 100 == 0:
                print(f'{datetime.now()} finished {index}%')
        print(f'{datetime.now()} finished embedding docs')
        return document_embeddings

    def get_sentence_embeddings(self, sentence: str) -> List[float]:
        if sentence.strip() == '' or len(sentence.split()) < MIN_WORDS_IN_SENTENCE:
            return
        sentence = f'[CLS] {sentence} [SEP]'
        tokens = self._tokenizer.encode(sentence)
        if len(tokens) > MAX_TOKENS_SIZE:
            tokens = tokens[:MAX_TOKENS_SIZE]
        input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size 1
        input_ids = input_ids.to(self._device)
        outputs = self._embedding_model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        mean_of_words = torch.mean(last_hidden_states, axis=1).squeeze(0).tolist()
        return mean_of_words

    def cluster(self, documents: List[str]) -> List[List[str]]:
        document_embeddings = self.get_document_embeddings(documents)
        cluster_assignment = k_means(document_embeddings)
        df = pd.DataFrame({'vectors': document_embeddings, 'cluster': cluster_assignment, 'text': documents})
        cluster_to_docs = df.groupby('cluster')['text'].apply(list).reset_index(name='docs')
        return cluster_to_docs['docs'].tolist()

