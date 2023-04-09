from __future__ import print_function
import cv2 as cv
import argparse

def detectAndDisplay(frame, face_cascade, eyes_cascade, smile_cascade):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        faceROI = frame_gray[y:y+h,x:x+w]

        # Detect smiles
        smiles = smile_cascade.detectMultiScale(faceROI, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 0, 255), 2)

        # Detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (ex, ey, ew, eh) in eyes:
            eye_center = (x + ex + ew//2, y + ey + eh//2)
            radius = int(round((ew + eh)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

    cv.imshow('Capture - Face detection', frame)

def main():
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='./haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='./haarcascade_eye_tree_eyeglasses.xml')
    parser.add_argument('--smile_cascade', help='Path to smile cascade.', default='./haarcascade_smile.xml')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    args = parser.parse_args()

    face_cascade = cv.CascadeClassifier(args.face_cascade)
    eyes_cascade = cv.CascadeClassifier(args.eyes_cascade)
    smile_cascade = cv.CascadeClassifier(args.smile_cascade)

    camera_device = args.camera
    cap = cv.VideoCapture(camera_device)

    if not cap.isOpened():
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        detectAndDisplay(frame, face_cascade, eyes_cascade, smile_cascade)

        if cv.waitKey(10) == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
