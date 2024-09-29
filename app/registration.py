from flask import Flask, render_template, request
import base64
import os
import face_recognition
import psycopg2
import numpy as np
from exception import UserAlreadyExistsError

app = Flask(__name__)

DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'toor',
    'host': 'localhost',
    'port': '5432'
}

def init_db():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def save_image_and_encoding(name, surname, image_data):
    # Decode the image data
    header, encoded = image_data.split(',', 1)
    image_data = base64.b64decode(encoded)
    
    # Save the image
    image_path = f"C:/faceDetection/face_rec/app/images/{name}_{surname}.jpg"
    with open(image_path, 'wb') as f:
        f.write(image_data)
    
    # Load the image and get face encoding
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        # Save encoding to database
        conn = init_db()
        if conn:
            cursor = conn.cursor()

            cursor.execute("""
                        SELECT users.name, users.surname
                        FROM users
                        WHERE users.name = %s AND users.surname = %s
                        """, (name, surname))
            
            userExist = cursor.fetchone()  # Fetch one result

            if userExist:
                cursor.close()
                if os.path.exists(image_path):
                    os.remove(image_path)
                raise UserAlreadyExistsError(f"User {name} {surname} already exists.")
                

            cursor.execute("""
                INSERT INTO users (name, surname)
                VALUES (%s, %s) RETURNING id
            """, (name, surname))

            user_id = cursor.fetchone()[0]

            encoding_binary = encodings[0].tobytes()  # Convert numpy array to bytes

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
        image_data = request.form['image']

        save_image_and_encoding(name, surname, image_data)
        return "Registration successful!"

    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)
