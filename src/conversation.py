from custom_vecstore.combine import create_combining_chain
from custom_vecstore.internal.datastore import CustomDataStore
from custom_vecstore.memory import create_summarizing_memory, ChatHistoryWithRoles
from custom_vecstore.vecstores import CustomVectorStore
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

from langchain_elasticsearch.vectorstores import ElasticsearchStore

from langchain_core.prompts import PromptTemplate

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from models import MAX_TOKEN_LIMIT, MODEL

from custom_vecstore.generators import create_function_json_generator_chain
from custom_vecstore.retrievers import CustomRetriever



def load_conversation_chain():

    verbose = True

    combine_docs_chain = create_combining_chain(MODEL, verbose=verbose)
    # vectorstore = CustomVectorStore()
    # retriever = vectorstore.as_retriever()
    data_store = CustomDataStore()
    retriever = CustomRetriever(data_store=data_store, default_search_type="SIMILARITY")


    max_tokens_limit = None

    history_with_roles = ChatHistoryWithRoles()

    



    json_generator = create_function_json_generator_chain(MODEL, verbose=verbose)
    # retriever.get_relevant_documents("something returned")


    chain = ConversationalRetrievalChain(
        combine_docs_chain=combine_docs_chain,
        get_chat_history=history_with_roles.get_chat_history,
        max_tokens_limit=max_tokens_limit,
        memory=create_summarizing_memory(MODEL),
        question_generator=json_generator,
        rephrase_question=False,
        retriever=retriever,
        verbose=verbose,
        )
    
    print("Loaded chain.")
    
    from langchain.docstore.document import Document
    from langchain_core.messages.chat import ChatMessage

    docs = [
        Document(page_content="Jesse loves red but not yellow"),
        Document(page_content = "Jamal loves green but not as much as he loves orange")
    ]

    history = [
        ChatMessage(role="human", content="hello there."),
        ChatMessage(role="ai", content="How can I help?"),
    ]

    outputs = chain.invoke({
        "context": docs,
        "chat_history": history,
        "question": "what color does Jesse not like?",
    })

    
    print(outputs)
    


    print("Done.")

    # return chain
