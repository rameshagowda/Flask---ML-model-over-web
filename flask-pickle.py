from flask import Flask, abort, jsonify, request
import pickle
import numpy as np

from sklearn.externals import joblib

my_random_forest = pickle.load(open("iris_rfc.pkl", "rb"))

app = Flask(__name__)

@app.route('/make_predict',methods = ['POST', 'GET'])
def make_predict():
    data = request.get_json(force=True)
    # print(data)
    predict_request = [data["sl"],data["sw"],data["pl"],data["pw"]]
    predict_request = np.array(predict_request)
    y_hat = my_random_forest.predict(predict_request)
    # print("class={0}".format(y_hat))
    output = y_hat[0]
    # print(type(output))
    return jsonify(result = str(output))


if __name__ == "__main__":
    app.run(port=9000, debug=True)