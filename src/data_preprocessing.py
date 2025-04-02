import cv2
import os
import numpy as np
from keras_facenet import FaceNet
from utils import ensure_directory_exists
from concurrent.futures import ThreadPoolExecutor

embedder = FaceNet()

embeddings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/embeddings")

ensure_directory_exists(embeddings_dir)

def augment_image(image):
    augs = [image]  # Original Image

    # Horizontal Flip
    augs.append(cv2.flip(image, 1))

    # Brightness Adjustment
    alpha = np.random.uniform(0.7, 1.3) #Random Brightness
    beta = np.random.randint(-30, 30) #Add/Reduce Brightness
    bright = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    augs.append(bright)

    # Rotation with Scaling
    angle = np.random.uniform(-20, 20)
    scale = np.random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, scale)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    augs.append(rotated)

    # Shear Transformation
    shear = np.random.uniform(-0.1, 0.1)
    M_shear = np.float32([[1, shear, 0], [shear, 1, 0]])
    sheared = cv2.warpAffine(image, M_shear, (image.shape[1], image.shape[0]))
    augs.append(sheared)

    # Gaussian Blur
    kernel_size = np.random.choice([3, 5])
    sigma = np.random.uniform(0.1, 1.5)
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    augs.append(blurred)

    # Color Jitter (HSV Space)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    # Adjust Saturation & Value
    hsv[..., 1] *= np.random.uniform(0.7, 1.3) #Saturation
    hsv[..., 2] *= np.random.uniform(0.7, 1.3)
    hsv = np.clip(hsv, 0, 255)
    hsv = hsv.astype(np.uint8)
    color_jittered = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    augs.append(color_jittered)

    # Gaussian Noise / Random noise
    noise = np.random.normal(0, np.random.uniform(0, 15), image.shape).astype(np.int16)
    noisy_image = cv2.add(image.astype(np.int16), noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    augs.append(noisy_image)

    # Contrast Adjustment
    contrast = np.random.uniform(0.8, 1.2)
    contrasted = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    augs.append(contrasted)

    # Grayscale Conversion
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # Convert back to 3-channel
    augs.append(gray)

    return augs

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
        """
        The extract method returns a dictionary with the following keys:
        
        - "box": Coordinates of the detected face in the image. It is a tuple (x1, y1, x2, y2), where (x1, y1) is the top-left corner of the bounding box, and (x2, y2) is the bottom-right corner.
        - "confidence": The confidence score of the detection, which represents the likelihood that the detected object is indeed a face.
        - "embedding": A vector representing the facial features of the detected face.
        
        Each detection represents a face, and you can access the bounding box to crop the face from the original image.
        """
        detections = embedder.extract(image, threshold=0.75)  # Face detection with threshold 75%
        for det in detections:
            x1, y1, x2, y2 = det["box"] #(x1, y1): Upper left| (x2,y2): Bottom right
            face = image[y1:y2, x1:x2] #Image Cropping from the original image

            if face.size == 0:
                continue

            embedded_images.append(face)

    return embedded_images

def process_face(face, embedder):
    return embedder.embeddings([face])[0] if face is not None and face.size != 0 else None #It returns a 3-dimensional array| (806 height, 602 width, 3 Color Channels)

def extract_embeddings(faces, embedder):
    with ThreadPoolExecutor(max_workers=4) as executor:
        embeddings = list(executor.map(lambda face: process_face(face, embedder), faces))
    return [emb for emb in embeddings if emb is not None]

person = input("Enter Name: ")
dataset_path = f"../../data/raw/{person}/"

if os.path.exists(dataset_path):

    images = load_images_from_folder(dataset_path)
    print(f"The images from the folder {dataset_path} are loaded. Number of photos: {len(images)}")

    faces = detect_faces(images)
    print(f"Detected faces: {len(faces)}")
    embeddings = extract_embeddings(faces, embedder)  # Νέα συνάρτηση που επιστρέφει όλα τα embeddings

    save_dir = os.path.join(embeddings_dir, person)
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
