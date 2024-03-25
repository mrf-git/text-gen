from typing import Any, Dict
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.llms.base import LLM
from langchain.prompts.prompt import PromptTemplate
from langchain_core.runnables import Runnable


COMBINE_TEMPLATE = """<s>[INST] The following is a conversation between a Human and AI.

context: {context}

{chat_history}

{question}
"""

COMBINE_PROMPT = PromptTemplate(template=COMBINE_TEMPLATE,
                input_variables=["context", "chat_history", "question"],
                )


DOCUMENT_TEMPLATE = "{page_content}"


DOCUMENT_PROMPT = PromptTemplate(template=DOCUMENT_TEMPLATE,
                input_variables=["page_content"],
                )



def create_combining_chain(llm: LLM, **kwargs: Any) -> StuffDocumentsChain:
    kwargs.pop("prompt", None)

    llm_chain = LLMChain(llm=llm, prompt=COMBINE_PROMPT,
                         
                         **kwargs)

    return StuffDocumentsChain(llm_chain=llm_chain,
                               document_prompt=DOCUMENT_PROMPT,
                               document_separator="\n\n",
                               document_variable_name="context",
                               **kwargs)

