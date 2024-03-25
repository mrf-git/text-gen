import logging
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

from custom_vecstore.retrievers import CustomRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.vectorstores import VectorStore

from custom_vecstore.internal.datastore import CustomData, CustomDataStore, DistanceStrategy, SearchType, SimilarityStrategy
from custom_vecstore.internal.texts import CustomTextBuilder


log = logging.getLogger(__name__)


class CustomVectorStore(VectorStore):

    def __init__(
        self,
        custom_params: dict,
        data_store: CustomDataStore,
        text_builder: CustomTextBuilder,
        default_search_type: Optional[SearchType] = "SIMILARITY",
        distance_strategy: Optional[DistanceStrategy] = None,
        similarity_strategy: Optional[SimilarityStrategy] = None,
    ):
        self.text_builder = text_builder
        self.data_store = data_store
        self.custom_params = custom_params
        self.embedding = data_store.embedding
        self.default_search_type = default_search_type
        self.similarity_strategy = similarity_strategy
        self.distance_strategy = distance_strategy
        
    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding
    
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return lambda score: score

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        
        embeddings = self.embedding.embed_documents(list(texts))
        ref_ids = kwargs.get("ref_ids")
        if not ref_ids:
            ref_ids = [str(uuid.uuid4()) for _ in texts]
        if len(ref_ids) != len(texts):
            raise ValueError()

        datas = []
        for i, text in enumerate(texts):
            ref_id = ref_ids[i]
            metadata = metadatas[i] if metadatas else {}
            custom_data = CustomData(ref_id, 0, text, embeddings[i], metadata)
            datas.append(custom_data)
        self.data_store.store(custom_data)

        return ref_ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            raise ValueError("No ids provided to delete.")

        self.data_store.delete(ids)
        

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        for document in documents:
            metadata = document.metadata
            texts = self.text_builder.document_to_texts(document, **kwargs)
            embeddings = self.embedding.embed_documents(list(texts))
            ref_id = kwargs.get("ref_id")
            if not ref_id:
                ref_id = str(uuid.uuid4())

            datas = []
            for i, text in enumerate(texts):
                custom_data = CustomData(ref_id, i, text, embeddings[i], metadata)
                datas.append(custom_data)
            self.data_store.store(custom_data)

    def search(self, query: str, search_type: Optional[str] = None,
               query_filters: Optional[List[Dict]] = None,
               k: int = 4, **kwargs: Any) -> List[Document]:
        if not search_type:
            search_type = self.default_search_type
        docs_and_scores = self.data_store.search(search_type=search_type, query=query,
                                                 query_filters=query_filters,
                                                 k=k, similarity_strategy=self.similarity_strategy,
                                                 distance_strategy=self.distance_strategy,
                                                 **kwargs)
        documents = [d[0] for d in docs_and_scores]
        return documents

    def similarity_search(
        self, query: str, query_filters: Optional[List[Dict]] = None, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(query, query_filters, k, **kwargs)
        documents = [d[0] for d in docs_and_scores]
        return documents
    
    def similarity_search_with_score(
        self, query: str, query_filters: Optional[List[Dict]] = None, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return self.data_store.search(search_type="SIMILARITY", query=query, 
                                      query_filters=query_filters,
                                      k=k, similarity_strategy=self.similarity_strategy,
                                      distance_strategy=self.distance_strategy,
                                      **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        custom_params: Optional[dict] = None,
        data_store: Optional[CustomDataStore] = None,
        add_kwargs: Optional[Dict] = {},
        **kwargs: Any,
    ) -> "CustomVectorStore":
        if not custom_params or not data_store:
            raise ValueError("custom_params and data_store required")
        if embedding is not data_store.embedding:
            raise ValueError("wrong embedding")
        vector_store = CustomVectorStore(custom_params=custom_params,
                                         data_store=data_store, **kwargs)
        vector_store.add_documents(documents, **add_kwargs)

        return vector_store

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        custom_params: Optional[dict] = None,
        data_store: Optional[CustomDataStore] = None,
        add_kwargs: Optional[Dict] = {},
        **kwargs: Any,
    ) -> "CustomVectorStore":
        if not custom_params or not data_store:
            raise ValueError("custom_params and data_store required")
        if embedding is not data_store.embedding:
            raise ValueError("wrong embedding")
        
        vector_store = CustomVectorStore(custom_params=custom_params,
                                         data_store=data_store, **kwargs)
        vector_store.add_texts(texts, metadatas=metadatas, **add_kwargs)

        return vector_store

    def as_retriever(self, search_type: Optional[SearchType] = None, **kwargs: Any) -> CustomRetriever:
        if not search_type:
            search_type = self.default_search_type
        return CustomRetriever(self.data_store, search_type, **kwargs)

