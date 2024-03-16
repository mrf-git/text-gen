from models import load_models, load_conversation


def post_fork(server, worker):
    load_models()
