Produce Predictor is a machine learning model that identifies the fruit/vegetable type and determines whether it is fresh or rotten using a Convolutional Neural Network. 
The model is trained using a multi-task CNN architecture that predicts both outputs simultaneously.
The system supports Apples, Bananas, Carrots, Oranges, and Tomatoes.

This program uses tensorflow which requires python 3.10. Ensure you have pyhton 3.10 installed.

# Install dependencies
py -3.10 -m pip install -r requirements.txt

# Run Command
py -3.10 predict_single_image.py
