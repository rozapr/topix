from .interfaces import TopicClusterer
from typing import List, Dict
from transformers import BertModel, BertTokenizer
from sklearn.cluster import KMeans
from datetime import datetime
import pandas as pd
import numpy as np
import torch

PATH_TO_MODEL = '/datadrive/topix/exploration/bert/checkpoint-2216000/'
MAX_TOKENS_SIZE = 512


class BertClusterer(TopicClusterer):
    _tokenizer = BertTokenizer.from_pretrained(PATH_TO_MODEL)
    _embedding_model = BertModel.from_pretrained(PATH_TO_MODEL)

    def get_sentence_embeddings(self, sentence):
        if sentence.strip() == '':
            return None
        sentence = f'[CLS] {sentence} [SEP]'
        tokens = self._tokenizer.encode(sentence)
        if len(tokens) > MAX_TOKENS_SIZE:
            tokens = tokens[:MAX_TOKENS_SIZE]
        input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size 1
        outputs = self._embedding_model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        mean_of_words = torch.mean(last_hidden_states, axis=1).squeeze(0).tolist()
        return mean_of_words

    def k_means(self, vectors, num_clusters=6):
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(vectors)
        cluster_assignment = clustering_model.labels_
        return cluster_assignment

    def cluster(self, documents: List[str]) -> List[List[str]]:
        document_embeddings = []
        print(f'{datetime.now()} num of docs: {len(documents)}')
        for index, document in enumerate(documents):
            sentences = document.split('\n')
            sentence_embeddings = [self.get_sentence_embeddings(sentence) for sentence in sentences]
            sentence_embeddings = np.array([sentence for sentence in sentence_embeddings if sentence is not None])
            mean_document_embedding = np.mean(sentence_embeddings, axis=0)
            document_embeddings.append(mean_document_embedding)
            print(f'{datetime.now()} finished doc {index}')

        cluster_assignment = self.k_means(document_embeddings)
        df = pd.DataFrame({'vectors': document_embeddings, 'cluster': cluster_assignment, 'text': documents})
        cluster_to_docs = df.groupby('cluster')['text'].apply(list).reset_index(name='docs')

        return cluster_to_docs['docs'].tolist()
