from typing import Any, List
from langchain.llms.base import LLM
from langchain_core.memory import BaseMemory
from langchain.prompts.prompt import PromptTemplate

from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages.chat import ChatMessage


SUMMARY_TEMPLATE = """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

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

SUMMARY_PROMPT = PromptTemplate(template=SUMMARY_TEMPLATE,
                                input_variables=["summary", "new_lines"],
                                )
    

def create_summarizing_memory(llm: LLM, **kwargs: Any) -> BaseMemory:
    kwargs.pop("prompt", None)
    memory = ConversationSummaryBufferMemory(llm=llm, prompt=SUMMARY_PROMPT,
                                             input_key="chat_history",
                                            #  memory_key="chat_memory",
                                            output_key="answer",
                                            **kwargs)
    return memory


class ChatHistoryWithRoles:

    def __init__(self) -> None:
        self.role_map = {"human": "Human: ", "ai": "AI: "}

    def get_chat_history(self, chat_history: List[ChatMessage]) -> str:
        buffer = ""
        for dialogue_turn in chat_history:
            role_prefix = self.role_map.get(dialogue_turn.role)
            buffer += f"\n{role_prefix}{dialogue_turn.content}"
        return buffer

