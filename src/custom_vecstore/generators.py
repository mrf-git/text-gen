from typing import Any, Dict, Union
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from langchain.llms.base import LLM



GENERATE_FUNCTION_JSON_TEMPLATE = """<s>[INST] Read the conversation between Human and AI, who are discussing
one or more documents, then determine which document-related function Human is currently requesting
from the set ["retrieve", "analyze", "transform", "other"], identify all parameters and their
values that would be required criteria to satisfy the request.
Output only a JSON object that contains fields for the function and the criteria. If the
number of results is specified, include a field for it also.

Parameters:
The function "retrieve" has parameters ["type", "category"]
The function "analyze" has parameters ["method"]
The function "transform" has parameters ["method"]
The function "other" has parameters ["why"]

Add extra parameters to the criteria if relevant.

Here is an example:

Background:

Current conversation:
Human: I'm looking for a book without fiction.
AI: There are 50 books without fiction. Which one would you like?
Human: Something about science, written in the last 100 years with positive reviews.

Function requested by Human:
[/INST]
{{
    "function": "retrieve",
    "criteria": [
        {{
            "parameter": "type",
            "values": ["book"]
        }},
        {{
            "parameter": "category",
            "values": ["non-fiction"]
        }},
        {{
            "parameter": "topic",
            "values": ["science"]
        }},
        {{
            "parameter": "written within",
            "values": ["last 100 years"]
        }}
    ]
}}
</s>
[INST]
Here is another:

Background:
AI is helping Human analyze several similar emails containing billing questions from
clients in various global regions.

Current conversation:
Human: How many clients are in the USA?
AI: There are emails from 3000 clients in the USA containing billing questions.
Human: What are the 3 most common issues clients want to address?
AI: Do you want to know common issues from all clients or only those in the USA?
Human: All clients who have paid at least 2 billing cycles.

Function requested by Human:
[/INST]
{{
    "function": "analyze",
    "num_results": 3,
    "criteria": [
        {{
            "parameter": "method",
            "values": ["find common topics"]
        }},
        {{
            "parameter": "global region",
            "values": ["all"]
        }},
        {{
            "parameter": "billing cycles",
            "values": ["at least 2"]
        }}
    ]
}}
</s>
[INST]
Here is another:

Background:
AI is helping Human analyze several financial reports over the period from November 12, 2021 to
February 12, 2022. The reports contain anomalies that Human is wants to investigate, such as
an increase of spending in January and apparent missing entries from several firms.

Current conversation:
Human: Who can help me make sense of these reports?
AI: The reports were filed by PartyOne, PartyTwo, and PartyThree.
Human: Send them all an email summarizing the problems and include a detailed table. Politely ask them for any assistance they can give. Copy me and BossOne.

Function requested by Human:
[/INST]
{{
    "function": "transform",
    "criteria": [
        {{
            "parameter": "method",
            "values": ["send email"]
        }},
        {{
            "parameter": "reports period",
            "values": ["November 12, 2021 to February 12, 2022"]
        }},
        {{
            "parameter": "anomalies",
            "values": ["increase of spending in January", "apparent missing entries from several firms"]
        }},
        {{
            "parameter": "send to",
            "values": ["PartyOne", "PartyTwo", "PartyThree"]
        }},
        {{
            "parameter": "instructions",
            "values": ["summarize anomalies", "include a detailed table", "politely ask for assistance", "copy Human and BossOne"]
        }}
    ]
}}
</s>
[INST]
Here is another:

Background:
AI is helping Human analyze a service agreement with CompanyOne.

Current conversation:
Human: Hi, my name is UserOne.
AI: Hi UserOne, I'm AI. Nice to meet you.
Human: Nice to meet you too.

Function requested by Human:
[/INST]
{{
    "function": "other",
    "criteria": [
        {{
            "parameter": "why",
            "values": ["The conversation is a greeting, which is unrelated to the background."]
        }}
    ]
}}
</s>
[INST]
Here is another:

{chat_history}
Human: {question}

Function requested by Human:
[/INST]
"""
GENERATE_FUNCTION_JSON_PROMPT = PromptTemplate(
    template=GENERATE_FUNCTION_JSON_TEMPLATE,
    input_variables=["chat_history", "question"],
)


def create_function_json_generator_chain(llm: LLM, **kwargs: Any) -> LLMChain:
    kwargs.pop("prompt", None)
    return LLMChain(llm=llm, prompt=GENERATE_FUNCTION_JSON_PROMPT,
                    **kwargs)





# class QuestionGeneratorChain(LLMChain):


#     def __init__(self, **kwargs: Any) -> None:
#         super().__init__(**kwargs)


#     def invoke(
#         self,
#         input: Dict[str, Any],
#         **kwargs: Any,
#     ) -> Dict[str, Any]:
        
        


#         final_outputs: Dict[str, Any] = self.prep_outputs(
#                 inputs, outputs, return_only_outputs
#             )
        
#         return final_outputs
