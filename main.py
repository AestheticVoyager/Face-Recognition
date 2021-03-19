import os
import face_recognition
import cv2

known_faces_dir = 'K'
unknown_faces_dir = 'U'
TOLERANCE=0.6
frame_thicc=3
font_thicc=2
MODEL='hog'

known_faces = []
known_names = []
for name in os.listdir(known_faces_dir):
    dir_path = os.path.join(known_faces_dir, name)
    if os.path.isdir(dir_path):
       for filename in os.listdir(dir_path):
          if not filename.startswith('.'):
              filepath = os.path.join(dir_path, filename)
              image = face_recognition.load_image_file(filepath)
              encoding=face_recognition.face_encodings(image)[0]
              known_faces.append(encoding)
              known_names.append(name)

print("Processing Unknown faces")
for filename in os.listdir(unknown_faces_dir):
    print(filename)
    image=face_recognition.load_image_file(f"{unknown_faces_dir}/{filename}")
    locations=face_recognition.face_locations(image, model=MODEL)
    encodings=face_recognition.face_encodings(image, locations)
    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results=face_recognition.compare_faces(known_faces, face_encoding,TOLERANCE)
        match=None
        if True in results:
            match=known_names[results.index(True)]
            print(f"Match Found: {match}")

            top_left=(face_location[3], face_location[0])
            bottom_right=(face_location[1], face_location[2])

            color=[0,255,255]

            cv2.rectangle(image, top_left, bottom_right, color, frame_thicc)

            top_left=(face_location[3], face_location[0])
            bottom_right=(face_location[1],face_location[2]+22)
            cv2.putText(image,match,(face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), font_thicc)

    cv2.imshow(filename, image)
    cv2.waitKey(10000)
    cv2.destroyWindow(filename)
