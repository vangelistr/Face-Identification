import cv2
import os
import numpy as np
from keras_facenet import FaceNet

embedder = FaceNet()

def augment_image(image):
    return [
        image,  # Original image
        cv2.flip(image, 1),  # Horizontal Flip
        cv2.convertScaleAbs(image, alpha=np.random.uniform(0.8, 1.2), beta=np.random.randint(-30, 30)),  # Brightness
        cv2.warpAffine(image,
                       cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), np.random.uniform(-15, 15),
                                               1), (image.shape[1], image.shape[0]))  # Rotation
    ]

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
        augmented_images = augment_image(image)
        images.extend(augmented_images)

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

def extract_embeddings(faces):
    embeddings = []

    for face in faces:
        if face is None or face.size == 0:
            continue

        embedding = embedder.embeddings([face])[0]
        embeddings.append(embedding)

    return embeddings if embeddings else None

person = "Vangelis Triantafyllou"
dataset_path = f"../data/raw/{person}/"

if os.path.exists(dataset_path):

    images = load_images_from_folder(dataset_path)
    print(f"The images from the folder {dataset_path} are loaded. Number of photos: {len(images)}")

    # Αντικαθιστούμε το mean_embeddings με embeddings
    faces = detect_faces(images)
    print(f"Detected faces: {len(faces)}")
    embeddings = extract_embeddings(faces)  # Νέα συνάρτηση που επιστρέφει όλα τα embeddings

    save_dir = os.path.join("../data/embeddings", person)
    print(f"Trying to create folder: {os.path.abspath(save_dir)}")

    try:
        os.makedirs(save_dir, exist_ok=True)  # Δημιουργία φακέλου αν δεν υπάρχει

        if embeddings is not None:
            for idx, embedding in enumerate(embeddings):
                save_path = os.path.join(save_dir, f"img_{idx + 1}.npy")  # Αποθήκευση κάθε embedding ξεχωριστά
                np.save(save_path, embedding)
                print(f"Saved embedding: {save_path}")

    except Exception as e:
        print(f"Error creating folder: {e}")

else:
    print(f"The folder {dataset_path} does not exists.")
