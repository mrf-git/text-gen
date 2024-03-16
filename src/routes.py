from http import HTTPStatus
import json
import time
import zoneinfo
import models

from datetime import datetime, timedelta
from html import escape
from flask import Flask, make_response, request, abort, Response



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

with open("/app/src/index.html", "rb") as f_in:
    INDEX_HTML = f_in.read()


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
    
    out_message = escape(writer_message)
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
    print(ret)
    print(type(ret))
    print(dir(ret))

    out_message = escape(writer_message)
    cur_time_str = datetime.now().astimezone().strftime("%H:%M")

    response = make_response(RESP_TEMPLATE.format(who="Bot", writer_class="",
                                                  msg=out_message, time_str=cur_time_str))
    response.headers.set("Content-Type", "text/html")
    return response



    # resp = {
    #     "resp:addMessages": {
    #         "messages": [
    #             writer_message
    #         ]
    #     }
    # }

    # resp_str = json.dumps(resp)
    # print(resp_str)

    # response = Response(None, HTTPStatus.NO_CONTENT, content_type="application/json")
    # print(response)
    # response.headers.set("HX-Trigger", resp_str)
    # print(response)
    # return response
