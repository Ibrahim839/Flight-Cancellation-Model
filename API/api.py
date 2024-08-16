from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load('./flight_cancellation_predictor.pkl')  


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.get_json()
        
        # Extract the features from the JSON data
        input_data = np.array(data['features']).reshape(1, -1)
        
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
