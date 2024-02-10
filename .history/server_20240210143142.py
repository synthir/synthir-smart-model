from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas
#import socket
# from sklearn.preprocessing import OneHotEncoder

class Predictor:
    def __init__(self, model_path, encoder_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)
        self.feature_names = self.model.get_booster().feature_names
        self.required_arguments = ["Sex", "Age_Group", "County", "Main_Diagnosis", "Arrival_Mode", "Discharge_To", "ATC_Code", "Prescribtion_Type"]
    
    def predict(self, args):
        fixed_args = {}
        for arg in self.required_arguments:
            if arg not in args:
                fixed_args[arg] = "0"
            else:
                fixed_args[arg] = args[arg]

        df = pandas.DataFrame(fixed_args, index=[0])
        user_input = self.encoder.transform(df)
        probability = self.model.predict_proba(user_input, validate_features=False)
        return float(probability[0][0])

app = Flask(__name__)
CORS(app)
p = Predictor("data/model.pkl", "data/encoder.pkl")

@app.route('/predict')
def predict():
    result = p.predict(request.args)
    return jsonify({"prediction": result})

@app.route("/")
def index():
    return "<h1> Hello Python! </h1>"

    
if __name__ == '__main__':
    # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock.bind(('localhost', 0))
    # port = sock.getsockname()[1]
    # sock.close()
    # app.run(port=port)
    app.run(host='127.0.0.1', port=19890)
