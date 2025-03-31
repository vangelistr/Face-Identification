import cv2
import joblib
from keras_facenet import FaceNet
from sklearn.preprocessing import normalize
import traceback

# Constants
TARGET_SIZE = (160, 160)  # FaceNet's expected input size
MIN_CONFIDENCE = 0.7  # Minimum confidence threshold for recognition
FACE_DETECTION_THRESHOLD = 0.75  # Face detection confidence threshold


def main():
    try:
        # Load pre-trained model
        model = joblib.load("../../models/face_recognition_svc.pkl")

        # Initialize FaceNet embedder
        embedder = FaceNet()

        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video capture")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB and get dimensions
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame.shape[:2]

            # Detect faces with confidence threshold
            detections = embedder.extract(rgb_frame, threshold=FACE_DETECTION_THRESHOLD)
            print(f"Detections: {detections}")

            for detection in detections:
                # Get bounding box coordinates with boundary checks
                x1, y1, width, height = detection["box"]
                x2 = x1 + width
                y2 = y1 + height

                # Προσθήκη padding 10% γύρω από το πρόσωπο
                padding_x = int(0.1 * width)
                padding_y = int(0.1 * height)

                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(frame_width, x2 + padding_x)
                y2 = min(frame_height, y2 + padding_y)

                print(f"Adjusted Face coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                # Extract face region
                face = rgb_frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                try:
                    # Preprocess face for embedding
                    resized_face = cv2.resize(face, TARGET_SIZE)

                    # Get face embedding
                    embedding = embedder.embeddings([resized_face])[0]
                    embedding = normalize([embedding])  # Normalize for classifier

                    # Predict identity
                    prediction = model.predict(embedding)[0]
                    confidence = model.predict_proba(embedding).max()

                    # Format label and confidence
                    label = prediction if confidence > MIN_CONFIDENCE else "Unknown"
                    text = f"{label} ({confidence:.1%})"

                except Exception as e:
                    print(f"Error processing face: {str(e)}")
                    continue

                # Prepare drawing parameters
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                thickness = 2

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Calculate text position
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(frame, text, (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)

            # Display result
            cv2.imshow("Face Recognition", frame)

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {str(e)}\n{traceback.format_exc()}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()