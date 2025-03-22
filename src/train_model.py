import os
import numpy as np
from PIL.GimpGradientFile import linear
from sklearn.svm import SVC #Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  #Model Saving

def load_embeddings():
    embeddings_dir = "../data/embeddings/"
    X, y = [], []

    for person in os.listdir(embeddings_dir):  # Διατρέχουμε τους φακέλους (ονόματα ατόμων)
        person_path = os.path.join(embeddings_dir, person)

        if os.path.isdir(person_path):  # Αν είναι φάκελος

            for file in os.listdir(person_path):  # Διατρέχουμε τα αρχεία του φακέλου
                if file.endswith(".npy"):  # Αν είναι αρχείο .npy
                    file_path = os.path.join(person_path, file)
                    embedding = np.load(file_path)
                    X.append(embedding)
                    y.append(person)

    return np.array(X), np.array(y)

X, y = load_embeddings()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy: .2f}")