import pickle

# Load the model from the file
with open("classifier.pkl", "rb") as pickle_in:
    classifier_loaded = pickle.load(pickle_in)

# Making a prediction
sample_data = [[2, 3, 4, 1]]  # Example data point
prediction = classifier_loaded.predict(sample_data)

print("Prediction for the provided data:", prediction)