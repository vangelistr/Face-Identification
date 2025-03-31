import os
import numpy as np
from sklearn.svm import SVC #Classifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize
import joblib  #Model Saving

def load_embeddings():
    embeddings_dir = "../../data/embeddings/"
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

def train_svm_model():
    X, y = load_embeddings()

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Error: No embeddings found! Make sure you have run the data preprocessing step.")

    X = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    param_grid = {
        "C": [0.1, 1, 10, 100],  # Regularization parameter
        "kernel": ["linear", "rbf"]  # Linear ή Gaussian SVM
    }

    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Επιλογή του καλύτερου μοντέλου
    model = grid_search.best_estimator_
    print(f"Best SVM model: C={grid_search.best_params_['C']}, Kernel={grid_search.best_params_['kernel']}")

    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    models_dir = "../../models/"
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "face_recognition_svc.pkl")
    print(f"Trying to save model to: {os.path.abspath(models_dir)}")
    try:
        joblib.dump(model, model_path)
        print(f"Model saved at: {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

train_svm_model()