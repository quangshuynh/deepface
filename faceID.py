from deepface import DeepFace
import cv2
from time import sleep
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))


def get_max_key(attributes):
    """Gets the key with the maximum value in a dictionary."""
    return max(attributes, key=attributes.get)


def get_largest_face(faces):
    """Gets the largest face from a list of faces."""
    if not isinstance(faces, list):
        return faces
    if not faces:
        return None
    return max(faces, key=lambda face: face[2] * face[3])


cam = cv2.VideoCapture(0)

while True:
    sleep(1)
    ret, image = cam.read()  

    if ret:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            face_classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_classifier.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) > 0:  
                objs = DeepFace.analyze(
                    img_path=image,
                    actions=['age', 'gender', 'race', 'emotion'],
                    enforce_detection=False
                )
                face_result = DeepFace.verify(
                    img1_path=r"C:\Users\Quang\Pictures\Camera Roll\WIN_20241026_00_12_56_Pro.jpg",
                    img2_path=image,
                    enforce_detection=False
                )
                color = (0, 255, 0) if face_result['verified'] else (0, 0, 255)
                for face in faces:
                    if len(face) == 4:
                        x, y, w, h = face
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        if isinstance(objs, list) and len(objs) > 0:
                            text = f'Age: {objs[0]["age"] - 10}, Gender: {get_max_key(objs[0]["gender"])}, Race: {get_max_key(objs[0]["race"])}, Emotion: {get_max_key(objs[0]["emotion"])}'
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(image, text, (x // 2, y - 10), font, 0.5, (255, 255, 255), 2)
                largest_face = get_largest_face(faces)
                if largest_face is not None and len(largest_face) == 4:
                    x, y, w, h = largest_face
                    crop_width = int(w * 2)  
                    crop_height = int(h * 2) 
                    x1 = max(0, x - crop_width // 2) + crop_width // 4
                    y1 = max(0, y - crop_height // 2) + crop_height // 4
                    x2 = min(image.shape[1], x1 + crop_width)
                    y2 = min(image.shape[0], y1 + crop_height)
                    cropped_image = image[y1:y2, x1:x2]
                    cropped_image = cv2.resize(cropped_image, (200, 200))
                    cv2.imshow('Cropped Image', cropped_image)
                    cv2.waitKey(1)
            cv2.imshow('Live Face Detection', image)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
        except Exception as e:
            print('Error while getting image:', e)
    else:
        print('No image detected')
cam.release()
cv2.destroyAllWindows()
