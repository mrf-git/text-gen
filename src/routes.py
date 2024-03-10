import models

from flask import Flask, make_response, request, abort


app = Flask(__name__)


with open("/app/src/index.html", "rb") as f_in:
    INDEX_HTML = f_in.read()


@app.route("/", methods = ["GET"])
def index():
    response = make_response(INDEX_HTML)
    response.headers.set("Content-Type", "text/html")
    return response


@app.route("/prompt", methods = ["POST"])
def prompt_generate():

    system_prompt = request.form.get("system_prompt")
    user_prompt = request.form.get("user_prompt")
    if not system_prompt or not user_prompt:
        abort(400)

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    tok_prompt = models.TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = models.MODEL_PIPELINE(tok_prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    out = str(outputs[0]["generated_text"])

    text = out.split("<|assistant|>")[1]
    resp = "<pre style=\"width:600px;overflow:auto\">%s</pre>" % text

    response = make_response(resp)
    response.headers.set("Content-Type", "text/html")
    return response
