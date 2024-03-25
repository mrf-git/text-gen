import os
import textwrap

import models

from datetime import datetime
from html import escape
from flask import Flask, make_response, request, abort



app = Flask(__name__)

RESP_TEMPLATE = """
<div class="chat-message-group {writer_class}">
    <div class="chat-messages">
        <div class="message custom-scroll overflowing">
            <pre class="bubble-content">{msg}</pre>
        </div>
        <div class="from">{who} {time_str}</div>
    </div>
</div>
"""

WRAP_AMOUNT = 40

with open(os.environ["INDEX_HTML_PATH"], "rb") as f_in:
    INDEX_HTML = f_in.read()


if os.getenv("PRELOAD_MODELS", "false") == "true":
    models.load_models()
    from conversation import load_conversation_chain
    load_conversation_chain()


@app.route("/", methods = ["GET"])
def index():
    models.load_conversation()
    response = make_response(INDEX_HTML)
    response.headers.set("Content-Type", "text/html")
    return response


@app.route("/format-prompt", methods = ["POST"])
def format_prompt():

    writer_message = request.form.get("writerMessage")
    if not writer_message:
        abort(400)
    
    ret = textwrap.fill(writer_message, WRAP_AMOUNT)
    out_message = escape(ret)
    cur_time_str = datetime.now().astimezone().strftime("%H:%M")

    response = make_response(RESP_TEMPLATE.format(who="You", writer_class="writer-user",
                                                  msg=out_message, time_str=cur_time_str))
    response.headers.set("Content-Type", "text/html")
    return response


@app.route("/submit-prompt", methods = ["POST"])
def submit_prompt():

    writer_message = request.form.get("writerMessage")
    if not writer_message:
        abort(400)

    ret = models.CONVERSATION.predict(input=writer_message)

    ret = textwrap.fill(ret, WRAP_AMOUNT)
    out_message = escape(ret)
    cur_time_str = datetime.now().astimezone().strftime("%H:%M")

    response = make_response(RESP_TEMPLATE.format(who="Bot", writer_class="",
                                                  msg=out_message, time_str=cur_time_str))
    response.headers.set("Content-Type", "text/html")
    return response
