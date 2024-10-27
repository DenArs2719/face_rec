import cv2
import face_recognition
import numpy as np
import psycopg2

from dotenv import load_dotenv
import os

load_dotenv()

DB_PARAMS = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

def init_db():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None

def is_static_image(face_encoding, frame, face_location):
    # Here, implement a simple heuristic to determine if the detected face is static
    # For example, check for image quality or flatness
    # This is a placeholder for more sophisticated checks
    top, right, bottom, left = face_location
    detected_face = frame[top:bottom, left:right]
    
    # Convert the detected face to grayscale
    gray_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

    # Check if the face is too uniform, which might indicate a photo
    mean_face_intensity = np.mean(gray_face)

    # A simple threshold for determining if the face is too flat/uniform
    return mean_face_intensity > 130  # You might need to adjust this threshold


def load_known_faces_from_db(conn):
    known_faces = {}
    cursor = conn.cursor()

    cursor.execute("""
        SELECT users.name, users.surname, face_images.encoding
        FROM users
        JOIN face_images ON users.id = face_images.user_id
    """)
    
    rows = cursor.fetchall()

    for name, surname, encoding_binary in rows:
        # binary data to numpy array
        encoding = np.frombuffer(encoding_binary, dtype=np.float64)
        full_name = f"{name} {surname}"

        if full_name not in known_faces:
            known_faces[full_name] = encoding

    return known_faces

# main function for face detection
def recognize_faces(known_faces):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert image from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # find face in frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Check if the detected face is likely a static image
            if is_static_image(face_encoding, frame, face_location):
                continue

            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
            name = "Unknown"
            
            if any(matches):
                first_match_index = matches.index(True)
                name = list(known_faces.keys())[first_match_index]

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Face recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Connect to the database and run face recognition
conn = init_db()

if conn:
    known_faces = load_known_faces_from_db(conn)
    recognize_faces(known_faces)
else:
    print("Issue with connection to db")
