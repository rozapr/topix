from typing import List, Dict, Any

from .clustering import TopicClusterer
from .topic_description import TopicDescriptor

class TopicModeler:

    _clusterer: TopicClusterer
    _descriptor: TopicDescriptor

    def __init__(self, clusterer: TopicClusterer, descriptor: TopicDescriptor):
        self._clusterer = clusterer
        self._descriptor = descriptor

    def topic_models(self, root_documents: List[str], depth : int = 0, subtopic_ratio : float = 0.2) -> List[Dict[str, Any]]:
        clusters = self._clusterer.cluster(documents=root_documents)
        topic_descriptions = self._descriptor.generate_descriptions(clusters)

        total_size = len(root_documents)

        topics = []

        for documents, description in zip(clusters, topic_descriptions):
            topic = {
                "documents": documents,
                "description": description
            }
            topic_size = len(documents)
            if depth > 0 and (topic_size / total_size) > subtopic_ratio:
                topic["subtopics"] = self.topic_models(documents, depth - 1, subtopic_ratio)
            topics.append(topic)
        
        return topics