import cv2
import mediapipe as mp

# Initialize OpenCV
cap = cv2.VideoCapture(0)

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Create a face mesh object
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Convert the image to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe
    results = face_mesh.process(rgb)

    # Check if landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                                   circle_radius=1),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                                     circle_radius=1))

    # Show the resulting frame
    cv2.imshow('Facial Landmark Detection', frame)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
