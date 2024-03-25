import logging
from typing import Any, List, Optional

from custom_vecstore.internal.datastore import CustomDataStore, SearchType
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.vectorstores import BaseRetriever

logger = logging.getLogger(__name__)


class CustomRetriever(BaseRetriever):

    data_store: CustomDataStore
    """CustomDataStore to use for retrieval."""
    default_search_type: SearchType
    """Default type of search to perform."""
    search_kwargs: Optional[dict] = {}
    """Keyword arguments to pass to the search function."""

    # def __init__(self, **kwargs: Any) -> None:
    #     super().__init__(**kwargs)

    # def __init__(self, data_store: CustomDataStore, default_search_type: SearchType,
    #              search_kwargs: Optional[dict] = {}, **kwargs: Any) -> None:
    #     # super().__init__(vectorstore=data_store, **kwargs)
    #     self.data_store = data_store
    #     self.default_search_type = default_search_type
    #     self.search_kwargs = search_kwargs


    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # docs_and_scores = self.data_store.search(self.default_search_type, query, **self.search_kwargs)
        # documents = [d[0] for d in docs_and_scores]
        # return documents
        run_manager.metadata
        print("retrieval query:", query, "\n-----")
        return []




