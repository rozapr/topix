from typing import List, Dict, Any

from .clustering import TopicClusterer
from .topic_description import TopicDescriptor

class TopicModeler:

    _clusterer: TopicClusterer
    _descriptor: TopicDescriptor

    def __init__(self, clusterer: TopicClusterer, descriptor: TopicDescriptor):
        self._clusterer = clusterer
        self._descriptor = descriptor

    def topic_models(self, documents: List[str]) -> List[Dict[str, Any]]:
        clusters = self._clusterer.cluster(documents=documents)
        topic_descriptions = self._descriptor.generate_descriptions(clusters)

        topics = []

        for documents, description in zip(clusters, topic_descriptions):
            topic = {
                "documents": documents,
                "description": description
            } 
            topics.append(topic)
        
        return topics