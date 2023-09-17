import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from flask import Flask, request, jsonify
import os
import logging

logging.basicConfig(filename="std.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=4)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfccs

def predict(sound_data):
    X = np.load("Heart_Sound.npy")
    nsamples, nx, ny = X.shape
    train_dataset = X.reshape((nsamples, nx * ny))
    data = pd.read_csv("dataset_heart.csv")
    n = data["label"].value_counts()[0]
    target = data.pop("label").values
    training = train_dataset
    sound_data = sound_data.reshape((1, 40 * 173))
    le = LabelEncoder()
    target = le.fit_transform(target.astype(str))
    g = [pow(2,-15),pow(2,-14),pow(2,-13),pow(2,-12),pow(2,-11),pow(2,-10),pow(2,-9),pow(2,-8),pow(2,-7),pow(2,-6),pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3)]
    
    C = [pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7),pow(2,8),pow(2,9),pow(2,10),pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]
    param= {'gamma': g,
        'kernel': ['rbf'],
        'C': C}
    mod = SVC()
    grid_search = GridSearchCV(mod,param)
    grid_search.fit(training, target)
    clf1 = SVC(gamma = grid_search.best_params_["gamma"],kernel=grid_search.best_params_["kernel"],C=grid_search.best_params_["C"])
    clf1.fit(training, target)
    preds = clf1.predict(sound_data)
    # extracting most confident predictions
    heart_class = le.inverse_transform(preds)
    t = "The Class is : " + str(heart_class)
    return t 

# Define the Flask app
app = Flask(__name__)

# API endpoint for heart sound data prediction
@app.route('/predict', methods=['POST'])
def predict_heart_sound_data():
    try:
        sound_file = request.files['sound_data']
        if sound_file:
            # Save the uploaded sound file temporarily
            temp_file_path = 'temp.wav'
            sound_file.save(temp_file_path)

            # Load and process the sound data
            features = extract_features(temp_file_path)
            t = predict(features)

            # Clean up the temporary file
            os.remove(temp_file_path)

            return jsonify({"prediction": t})
        else:
            return jsonify({"error": "No sound data received."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)