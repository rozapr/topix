from abc import abstractmethod, ABC
from typing import List, Dict

class TopicDescriptor(ABC):

    @abstractmethod
    def generate_descriptions(self, topics: List[List[str]]) -> List[str]:
        pass