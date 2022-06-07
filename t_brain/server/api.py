from typing import List, Dict
import datetime
import hashlib
import time

from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

from ..facades import valid_model

app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = "a22923509@gmail.com"  #
SALT = "baby"  #
#########################################


def generate_server_uuid(input_string: str) -> str:
    sha = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    sha.update(data)
    server_uuid = sha.hexdigest()
    return server_uuid


def predict(sentence_list: List[str]) -> str:

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    max_score = -1
    answer_index = -1
    scores = valid_model.predict(sentence_list)
    print(scores)

    for index, score in enumerate(scores):
        if score > max_score:
            max_score = score
            answer_index = index

    prediction = sentence_list[answer_index]

    ####################################################
    if _check_datatype_to_string(prediction):
        return prediction


def _check_datatype_to_string(prediction: str) -> bool:
    if isinstance(prediction, str):
        return True
    raise TypeError("Prediction is not in string type.")


@app.route("/inference", methods=["POST"])
def inference():
    """API that return your model predictions when E.SUN calls this API."""
    data = request.get_json(force=True)
    sentence_list = data["sentence_list"]

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)

    try:
        answer = predict(sentence_list)
    except TypeError as type_error:
        # You can write some log...
        raise type_error
    except Exception as e:
        # You can write some log...
        raise e
    server_timestamp = time.time()

    return jsonify(
        {
            "esun_uuid": data["esun_uuid"],
            "server_uuid": server_uuid,
            "answer": answer,
            "server_timestamp": server_timestamp,
        }
    )
