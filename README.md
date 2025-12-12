Produce Predictor is a machine learning model that identifies the fruit/vegetable type and determines whether it is fresh or rotten using a Convolutional Neural Network. 
The model is trained using a multi-task CNN architecture that predicts both outputs simultaneously.
The system supports Apples, Bananas, Carrots, Oranges, and Tomatoes.

This program uses tensorflow which requires python 3.10. Ensure you have pyhton 3.10 installed.

This repo contains 2 python code files, 10 test images, and the pretrained model. train_freshness_model_2 is the code that was used to create the model. fruit_freshness_multitask.keras is the pretrained, 
multi-label model. predict_single_image can be used to insert one image at a time and recieve a produce type and freshness prediction.

# Install dependencies
py -3.10 -m pip install -r requirements.txt

# running predict_single_image
you must first open the code for predict_single_image and edit line 53:

test_image = r"YOUR IMAGE PATH" # CHANGE THIS to the image you want to test

Then, run this line to run the program:

py -3.10 predict_single_image.py
