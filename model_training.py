#RUN THIS FILE INORDER TO CREATE THE PKL FILE FOR THE MODEL
#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import sklearn

# Load dataset
df = pd.read_csv('BankNote_Authentication.csv')
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train classifier
classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)

# Save the trained model to a file
with open("classifier.pkl", "wb") as pickle_out:
    pickle.dump(classifier, pickle_out)

# Print the scikit-learn version for reference
    
# print("Scikit-learn version used:", sklearn._version_)