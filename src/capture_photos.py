import cv2
import os

def capture_photos(person_name, save_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/raw"), num_photos=10):
    person_dir = os.path.join(save_dir, person_name)
    print(f"Trying to create the folder: {os.path.abspath(person_dir)}")
    try:
        os.makedirs(person_dir, exist_ok=True)
    except Exception as e:
        print(f"Error: Unable to create the folder: {e}")
        return

    # Υπολογίζει το επόμενο διαθέσιμο όνομα αρχείου
    existing_files = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
    existing_files.sort()  # Ταξινόμηση των αρχείων για εύρεση του τελευταίου αριθμού
    next_index = 1

    if existing_files:
        last_file = existing_files[-1]  # Παίρνει το τελευταίο αρχείο
        last_index = int(last_file.replace("img", "").replace(".jpg", ""))  # Εξάγει τον αριθμό
        next_index = last_index + 1  # Ορίζει το επόμενο διαθέσιμο όνομα

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

        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            img_path = os.path.join(person_dir, f"img{next_index}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved: {img_path}")
            count += 1
            next_index += 1  # Αυξάνει τον μετρητή
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} new photos for {person_name}.")

# if __name__ == "__main__":
#     person_name = input("Enter person's name: ")
#     capture_photos(person_name)