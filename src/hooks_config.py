from models import load_models


def post_fork(server, worker):
    load_models()
