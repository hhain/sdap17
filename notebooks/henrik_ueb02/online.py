import datetime
from flask import Flask, jsonify, abort, make_response, request
from flask_restful import Api, Resource, reqparse, fields, marshal, marshal_with
from flask_cors import CORS, cross_origin
from flask import url_for

APP = Flask(__name__, static_url_path="")
CORS(APP)
API = Api(APP)

Prediction = {
    'row': fields.String,
    'p_class': fields.String
}

class Prediction(Resource):

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('row', type=str, location='json')
        self.reqparse.add_argument('p_class', type=str, location='json')

    def post(self):
        json_data = request.get_json(force=True)
        return json_data


API.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    APP.run(host='0.0.0.0', port=5444 ,debug=True)
