from abc import abstractmethod, ABC
from typing import List, Dict

class TopicClusterer(ABC):

    @abstractmethod
    def cluster(self, documents: List[str]) -> List[List[str]]:
        pass