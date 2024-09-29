import cv2
import face_recognition
import numpy as np
import psycopg2

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
        print(f"Ошибка подключения к базе данных: {e}")
        return None

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
        # Преобразуем бинарные данные обратно в массив numpy
        encoding = np.frombuffer(encoding_binary, dtype=np.float64)
        full_name = f"{name} {surname}"

        if full_name not in known_faces:
            known_faces[full_name] = encoding  # Сохраняем только одно кодирование

    return known_faces

# 2. Основная функция распознавания лиц
def recognize_faces(known_faces):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразование изображения из BGR в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Поиск лиц на кадре
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Сравнение с известными лицами
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
            name = "Unknown"  # Имя по умолчанию
            
            # Проверяем, есть ли хотя бы одно совпадение
            if any(matches):
                first_match_index = matches.index(True)
                name = list(known_faces.keys())[first_match_index]

            # Рисуем рамку вокруг лица и имя
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Распознавание лиц', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 3. Загрузка известных лиц и распознавание
conn = init_db()
if conn:
    known_faces = load_known_faces_from_db(conn)
    recognize_faces(known_faces)
else:
    print("Не удалось инициализировать базу данных.")