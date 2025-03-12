import cv2
import os

def capture_photos(person_name, save_dir="data/raw", num_photos=10):
    person_dir = os.path.join(save_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'c' to capture photos and 'q' to quit.")

    count = 0
    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the frame in a window
        cv2.imshow('Camera', frame)

        # Wait for key press and handle it
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            img_path = os.path.join(person_dir, f"img{count + 1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved: {img_path}")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} photos for {person_name}.")

if __name__ == "__main__":
    person_name = input("Enter person's name: ")
    capture_photos(person_name, num_photos=10)