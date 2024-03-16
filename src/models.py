import os

from langchain.chains.conversation.base import ConversationChain
from langchain.llms.llamacpp import LlamaCpp
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate


MODEL = None


def load_models():
    global MODEL
    if not MODEL:
        MODEL = LlamaCpp(model_path=os.environ["MODEL_PATH"],
                        verbose=True,
                        stop=["</s>", "Human:"],
                        max_tokens=1000,
                        n_batch=2000,
                        n_ctx=4096,
                        n_threads=20,
                        )
        print("Loaded model.")



CONVERSATION = None  # TODO store different sessions
MAX_TOKEN_LIMIT = 1000

def load_conversation():
    global MODEL
    global CONVERSATION
    if CONVERSATION:
        del CONVERSATION

    summary_template = """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

EXAMPLE
Current summary:
AI would like to assist Human, who desires help with something unknown.

New lines of conversation:
Human: Hello.
AI: Hello. How can I help you?
Human: I need your help with something.

New summary:
AI and Human greeted each other and Human asked for AI's help, who would like to assist.
END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:"""

    summary_prompt = PromptTemplate(template=summary_template,
                   input_variables=["summary", "new_lines"],
                   )
    

    memory = ConversationSummaryBufferMemory(llm=MODEL, prompt=summary_prompt,
                                             max_token_limit=MAX_TOKEN_LIMIT)


    chat_template = """The following is a friendly conversation between a Human and AI.
Current conversation:
{history}
Human: {input}
AI:"""

    chat_prompt = PromptTemplate(template=chat_template,
                   input_variables=["history", "input"],
                   )


    CONVERSATION = ConversationChain(
        llm=MODEL,
        memory=memory,
        prompt=chat_prompt,
        verbose=True,
    )
    print("Loaded conversation.")


