import logging
from flask import Flask, jsonify, request
from scribble import Scribble


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module)s - %(levelname)s - %(message)s')


app = Flask(__name__)

engine_path = ''
entry = Scribble('./engine')


@app.route('/hz')
def hz():
    return 'well'


@app.route('/api/control/scribble', methods=['POST'])
def infer():
    try:
        request_body = request.get_json()
        control_b64 = request_body['image']
        prompts = request_body['prompts']
        neg_prompts = request_body['neg_prompts']
        steps = request_body.get('steps', 20)
        scale = request_body.get('scale', 7.5)
        seed = request_body.get('seed', None)
    except:
        raise "please check input params!"
    img_b64 = entry.infer(prompts=prompts,
                          neg_prompts=neg_prompts,
                          control=control_b64,
                          seed=seed,
                          scale=scale,
                          steps=steps)
    return jsonify({'image': img_b64})
