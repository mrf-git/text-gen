from models import load_models, load_conversation
from conversation import load_conversation_chain


def post_fork(server, worker):
    load_models()
    load_conversation_chain()
