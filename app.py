from flask import Flask, jsonify
from flask_restful import Resource, reqparse
import getQTMatrixOpt

app = Flask(__name__)

parser = reqparse.RequestParser()
parser.add_argument('listOfQ',
                    type=list,
                    required=True,
                    location='json',
                    help="parameter 'listOfQ' must be in your request")
parser.add_argument('T',
                    type=int,
                    required=True,
                    help="parameter 'T' must be in your request")
parser.add_argument('listOfS',
                    type=list,
                    required=True,
                    location='json',
                    help="parameter 'listOfS' must be in your request")


@app.route('/', methods = ['POST'])
def getSTMatrix():
    data = parser.parse_args()
    matrix = getQTMatrixOpt.setSTMatrix(data)
    if matrix:
        return matrix, 200
    else:
        return {"error": "something went wrong :("}, 500


if __name__ == '__main__':
    app.run()
