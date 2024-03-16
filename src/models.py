import transformers
import torch

from langchain.chains.conversation.base import ConversationChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate


MODEL_PIPELINE = None
TOKENIZER = None
LLM = None


def load_models():
    global MODEL_PIPELINE
    global TOKENIZER
    global LLM
    if not MODEL_PIPELINE:
        MODEL_PIPELINE = transformers.pipeline(
            "text-generation",
            model="/models/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.bfloat16,
            device="cuda",
        )
        TOKENIZER = MODEL_PIPELINE.tokenizer
        LLM = HuggingFacePipeline(pipeline=MODEL_PIPELINE)
        print("Loaded model.")


CONVERSATION = None  # TODO store different sessions
MAX_TOKEN_LIMIT = 200

def load_conversation():
    global LLM
    global TOKENIZER
    global CONVERSATION
    if CONVERSATION:
        del CONVERSATION

    # if TOKENIZER.chat_template is not None:
    #     chat_template = TOKENIZER.chat_template
    # else:
    #     chat_template = TOKENIZER.default_chat_template


    # summary_template = """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

    # EXAMPLE
    # Current summary:
    # The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

    # New lines of conversation:
    # Human: Why do you think artificial intelligence is a force for good?
    # AI: Because artificial intelligence will help humans reach their full potential.

    # New summary:
    # The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
    # END OF EXAMPLE

    # Current summary:
    # {summary}

    # New lines of conversation:
    # {new_lines}

    # New summary:"""



    messages = [
        {
            "role": "system",
            "content": """
            Your task is to progressively update the summary of a conversation between
            <|user|> and <|assistant|> by first understanding the current summary and then
            reading the next part of the conversation, which you will use to write a new
            summary that includes details from the current summary and its progression
            over the next part. The current summary is:
            {summary}
            """,
        },
        # {
        #     "role": "user",
        #     "content": "user_prompt",
        # },
        # {
        #     "role": "assistant",
        #     "content": "assistant_prompt",
        # },
    ]

    tok_prompt = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    tok_prompt += "{new_lines}"



    messages = [
        {
            "role": "system",
            "content": """
            Write a summary of the ongoing conversation between
            <|user|> and <|assistant|>. The summary must be true, concise,
            and detailed.
            """,
        },
    ]

    tok_prompt += TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    


    print(tok_prompt)

    summary_prompt = PromptTemplate(template=tok_prompt,
                   input_variables=["summary", "new_lines"],
                   )
    

    memory = ConversationSummaryBufferMemory(llm=LLM, prompt=summary_prompt,
                                             max_token_limit=MAX_TOKEN_LIMIT)





    # prompt = PromptTemplate(template=chat_template, template_format="jinja2",
    #                input_variables=["messages", "add_generation_prompt"],
    #                )


    chat_template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    Current conversation:
    {history}
    Human: {input}
    AI:"""

    chat_prompt = PromptTemplate(template=chat_template,
                   input_variables=["history", "input"],
                   )


    CONVERSATION = ConversationChain(
        llm=LLM,
        memory=memory,
        prompt=chat_prompt,
        verbose=True,
    )
    print("Loaded conversation.")


    ret = CONVERSATION.predict(input="hello world")
    print(ret)

    ret = CONVERSATION.predict(input="nice to meet you.")
    print(ret)

