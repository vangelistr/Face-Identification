import cv2
import os
import numpy as np
from keras_facenet import FaceNet

embedder = FaceNet()

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)

        #Loading the image
        image = cv2.imread(img_path)
        if image is None:
            continue

        #Converting from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images

def detect_faces(images):
    embedded_images = []

    for image in images:
        detections = embedder.extract(image, threshold=0.95)  # Face detection
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            face = image[y1:y2, x1:x2]

            if face.size == 0:
                continue

            embedded_images.append(face)

    return embedded_images

def extract_mean_embeddings(images):
    embeddings = []

    for image in images:
        if image is None or image.size == 0:  # Αν η εικόνα είναι άδεια, αγνόησέ την
            continue

        embedding = embedder.embeddings([image])[0]
        embeddings.append(embedding)

    if len(embeddings) == 0:
        return None

    mean_embedding = np.mean(embeddings, axis=0)
    return mean_embedding


person = "Vangelis Triantafyllou"
dataset_path = f"../data/raw/{person}/"

if os.path.exists(dataset_path):

    images = load_images_from_folder(dataset_path)
    print(f"The images from the folder {dataset_path} are loaded. Number of photos: {len(images)}")

    faces = detect_faces(images)
    mean_embeddings = extract_mean_embeddings(faces)

    save_dir = os.path.join("../data/embeddings", person)
    print(f"Trying to create folder: {os.path.abspath(save_dir)}")

    try:
        os.makedirs(save_dir, exist_ok=True)  # Δημιουργία φακέλου αν δεν υπάρχει
        save_path = os.path.join(save_dir, f"{person}.npy")  # Σωστό path για το αρχείο
        if mean_embeddings is not None:
            np.save(save_path, mean_embeddings)
            print(f"The embedding was saved at {save_path}")

    except Exception as e:
        print(f"Error creating folder: {e}")

else:
    print(f"The folder {dataset_path} does not exists.")
