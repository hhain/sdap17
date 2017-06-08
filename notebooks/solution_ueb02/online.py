import datetime
from flask import Flask, jsonify, abort, make_response, request
from flask_restful import Api, Resource, reqparse, fields, marshal, marshal_with
from flask_cors import CORS, cross_origin
from flask import url_for
from sklearn.externals import joblib
import numpy as np

APP = Flask(__name__, static_url_path="")
CORS(APP)
API = Api(APP)
MODEL = None


class Prediction(Resource):

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('p_class', type=str, location='json')

    def post(self):
        json_data = request.get_json(force=True)
        sample = np.array(json_data['data'])
        return jsonify({'result': MODEL.predict(sample).tolist()})


API.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    MODEL = joblib.load('../../models/solution_ueb02/nb_model.plk')
    print(MODEL)
    APP.run(host='0.0.0.0', port=5444 ,debug=True)

