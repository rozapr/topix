from .interfaces import TopicClusterer
from typing import List, Dict
from transformers import BertModel, BertTokenizer
from sklearn.cluster import KMeans
import pandas as pd
import torch

PATH_TO_MODEL = '/datadrive/topix/exploration/bert/'
MAX_TOKENS_SIZE = 510


class BertClusterer(TopicClusterer):
    _tokenizer = BertTokenizer.from_pretrained(PATH_TO_MODEL)
    _embedding_model = BertModel.from_pretrained(PATH_TO_MODEL)

    def get_sentence_embeddings(self, sentence):
        tokens = self._tokenizer.encode(sentence)
        if len(tokens) > MAX_TOKENS_SIZE:
            tokens = tokens[:MAX_TOKENS_SIZE]
        input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size 1
        outputs = self._embedding_model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        mean_of_words = torch.mean(last_hidden_states, axis=2)
        return mean_of_words

    def k_means(self, vectors, num_clusters=6):
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(vectors)
        cluster_assignment = clustering_model.labels_
        return cluster_assignment

    def cluster(self, documents: List[str]) -> List[List[str]]:
        document_embeddings = []
        for document in documents:
            sentences = document.split('\n')
            sentence_embeddings = [get_sentence_embeddings(sentence) for sentence in sentences]
            sentence_embeddings = np.array([sentence for sentence in sentence_embeddings if sentence is not None])
            mean_document_embedding = np.mean(sentence_embeddings, axis=1)
            document_embeddings.append(mean_document_embedding)

        cluster_assignment = k_means(document_embeddings)
        df = pd.DataFrame({'vectors': document_embeddings, 'cluster': cluster_assignment, 'text': documents})
        cluster_to_docs = df.groupby('cluster')['text'].apply(list).reset_index(name='docs')

        return cluster_to_docs['docs'].tolist()

