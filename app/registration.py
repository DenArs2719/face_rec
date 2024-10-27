from flask import Flask, render_template, request
import base64
import os
import face_recognition
import psycopg2
import numpy as np
from exception import MultipleFacesDetectedError, UserAlreadyExistsError
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables from .env file
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
        print(f"Error connecting to database: {e}")
        return None

def save_image_and_encoding(name, surname, pesel, image_data):
    # Decode the image data
    header, encoded = image_data.split(',', 1)
    image_data = base64.b64decode(encoded)
    
    # Save the image
    image_path = f"images/{name}_{surname}.jpg"

    with open(image_path, 'wb') as f:
        f.write(image_data)
    
    # Load the image and get face encoding
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    # Check if there are no faces or multiple faces
    if len(encodings) == 0:
        if os.path.exists(image_path):
            os.remove(image_path)
        raise ValueError("No face detected. Please upload an image with a clear face.")
    elif len(encodings) > 1:
        if os.path.exists(image_path):
            os.remove(image_path)
        raise MultipleFacesDetectedError("Multiple faces detected. Please upload an image with only one face.")

    # If exactly one face is detected, proceed with saving the encoding
    if encodings:
        conn = init_db()
        if conn:
            cursor = conn.cursor()

            cursor.execute("""
                        SELECT users.name, users.surname, users.pesel
                        FROM users
                        WHERE users.name = %s AND users.surname = %s AND users.pesel = %s
                        """, (name, surname,pesel))
            
            userExist = cursor.fetchone()

            if userExist:
                cursor.close()
                if os.path.exists(image_path):
                    os.remove(image_path)
                raise UserAlreadyExistsError(f"User {name} {surname} already exists.")
                
            cursor.execute("""
                INSERT INTO users (name, surname, pesel)
                VALUES (%s, %s, %s) RETURNING id
            """, (name, surname, pesel))

            user_id = cursor.fetchone()[0]

            # Convert numpy array to bytes
            encoding_binary = encodings[0].tobytes()

            cursor.execute("""
                INSERT INTO face_images (user_id, encoding)
                VALUES (%s, %s)
            """, (user_id, encoding_binary))

            conn.commit()
            conn.close()


@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        surname = request.form['surname']
        pesel = request.form['pesel']
        image_data = request.form['image']

        save_image_and_encoding(name, surname,pesel, image_data)
        return "Registration successful!"

    return render_template('register.html')


@app.errorhandler(UserAlreadyExistsError)
def handle_value_error(e):
    return render_template('error.html', error_message=str(e)), 400

@app.errorhandler(MultipleFacesDetectedError)
def handle_multiple_faces_error(e):
    return render_template('error.html', error_message=str(e)), 400


@app.errorhandler(ValueError)
def handle_multiple_faces_error(e):
    return render_template('error.html', error_message=str(e)), 400


if __name__ == '__main__':
    app.run(debug=True)
