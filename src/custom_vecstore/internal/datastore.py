from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from typing import Any, Dict, List, Literal, Optional, Tuple


class CustomData:

    def __init__(self, ref_id: str, segment_index: int,
                 text: str, embedding: List[float], metadata: dict) -> None:
        pass



SearchType = Literal[
    "KEYWORD",
    "SIMILARITY",
]

DistanceStrategy = Literal[
    "COSINE",
    "DOT_PRODUCT",
    "EUCLIDEAN_DISTANCE",
    "MAX_INNER_PRODUCT",
]

SimilarityStrategy = Literal[
    "EXACT",
    "APPROXIMATE",
]


class CustomDataStore:

    @property
    def embedding(self) -> Embeddings:
        return self._embedding
    
    def __init__(self) -> None:
        self._embedding = None

    def store(custom_data: CustomData, **kwargs: Any):
        pass

    def delete(ref_ids: List[str], **kwargs: Any):
        pass

    def search(self,
               search_type: SearchType,
                query: str, query_filters: Optional[List[Dict]] = None,
                k: int = 4, similarity_strategy: Optional[SimilarityStrategy] = None,
                distance_strategy: Optional[DistanceStrategy] = None,
                **kwargs: Any) -> List[Tuple[Document, float]]:
        if similarity_strategy is None:
            similarity_strategy == "APPROXIMATE" if distance_strategy is None else "EXACT"
        if similarity_strategy == "APPROXIMATE" and distance_strategy is not None:
            raise ValueError("APPROXIMATE strategy not compatible with distance_strategy")
        elif distance_strategy is None:
            distance_strategy = "COSINE"
