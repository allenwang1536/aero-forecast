from aeroForecast.model import load_model
from flask import Flask, jsonify, request
import torch


app = Flask(__name__)
model = load_model()
mockWeatherResponse = torch.load('backend/mock/mock_weather_data.pth')
mockWeatherResponse = mockWeatherResponse.unsqueeze(0)


@app.route('/predict', methods=['GET'])
def predict():
    output, _ = model(mockWeatherResponse)
    
    output_list = output.tolist()

    return output_list

if __name__ == '__main__':
    app.run(debug=True)
