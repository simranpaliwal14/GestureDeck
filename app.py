import os
import subprocess
import threading
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from pdf2image import convert_from_path
from cvzone.HandTrackingModule import HandDetector
from flask_socketio import SocketIO, emit
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize SocketIO
socketio = SocketIO(app)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'ppt', 'pptx'}
current_slide_index = 0
path_images = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ppt_to_pdf(ppt_path, output_folder):
    command = [
        "soffice",
        "--headless",
        "--convert-to", "pdf",
        "--outdir", output_folder,
        ppt_path
    ]
    subprocess.run(command, check=True)

def pdf_to_images(pdf_path, output_folder, size=(1280, 720)):
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        image = image.resize(size, Image.LANCZOS)
        image.save(os.path.join(output_folder, f"slide_{i+1}.png"), 'PNG')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Convert PPT to PDF and then to images
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename.replace('.pptx', '.pdf'))
            ppt_to_pdf(file_path, app.config['UPLOAD_FOLDER'])
            pdf_to_images(pdf_path, 'static/presentation_images')

            global path_images
            path_images = sorted([f for f in os.listdir('static/presentation_images') if f.endswith('.png')])

            # Start gesture control thread
            threading.Thread(target=start_gesture_control).start()

            return redirect(url_for('gesture_control'))
    return render_template('upload.html')

@app.route('/gesture-control')
def gesture_control():
    return render_template('gesture_control.html')

@app.route('/current-slide')
def current_slide():
    global current_slide_index
    slide_image = path_images[current_slide_index] if path_images else ""
    return jsonify(slide=slide_image)

# Emit slide change and annotations to the client
def emit_slide_and_annotations(slide_image, annotations):
    socketio.emit('update_slide', {
        'slide_image': slide_image,
        'annotations': annotations
    })

def start_gesture_control():
    global current_slide_index
    cap = cv2.VideoCapture(0)
    width, height = 1280, 720
    cap.set(3, width)
    cap.set(4, height)

    try:
        global path_images
        path_images = sorted([f for f in os.listdir('static/presentation_images') if f.endswith('.png')], key=len)
        if not path_images:
            print("No images found in the output folder.")
            return
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    hs, ws = int(120 * 1.2), int(213 * 1.2)
    gesture_threshold = 300
    button_pressed = False
    button_counter = 0
    button_delay = 20
    annotations = [[]]
    annotation_number = 0
    annotation_start = False

    detector = HandDetector(detectionCon=0.8, maxHands=1)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from camera.")
            break
        img = cv2.flip(img, 1)

        # Load current slide image
        if path_images:
            path_full_image = os.path.join('static/presentation_images', path_images[current_slide_index])
            img_current = cv2.imread(path_full_image)

            # Detect hands
            hands, img = detector.findHands(img)

            cv2.line(img, (0, gesture_threshold), (width, gesture_threshold), (0, 255, 0), 10)
            if hands and not button_pressed:
                hand = hands[0]
                fingers = detector.fingersUp(hand)
                cx, cy = hand['center']
                lm_list = hand['lmList']
                x_val = int(np.interp(lm_list[8][0], [width // 2, width], [0, width]))
                y_val = int(np.interp(lm_list[8][1], [150, height - 150], [0, height]))
                index_finger = (x_val, y_val)

                socketio.emit('gesture_pointer', {'x': index_finger[0], 'y': index_finger[1]})

                if cy <= gesture_threshold:
                    annotation_start = False
                    if fingers == [1, 0, 0, 0, 0]:  # Left swipe
                        if current_slide_index > 0:
                            current_slide_index -= 1
                            button_pressed = True
                            annotations = [[]]  # Clear annotations on slide change
                            annotation_number = 0
                            emit_slide_and_annotations(path_images[current_slide_index], annotations)

                    elif fingers == [0, 0, 0, 0, 1]:  # Right swipe
                        if current_slide_index < len(path_images) - 1:
                            current_slide_index += 1
                            button_pressed = True
                            annotations = [[]]  # Clear annotations on slide change
                            annotation_number = 0
                            emit_slide_and_annotations(path_images[current_slide_index], annotations)

                elif fingers == [0, 1, 1, 0, 0]:  # Pointer
                    cv2.circle(img_current, index_finger, 12, (0, 0, 255), cv2.FILLED)

                elif fingers == [0, 1, 0, 0, 0]:  # Draw
                    if not annotation_start:
                        annotation_start = True
                        annotation_number += 1
                        annotations.append([])
                    cv2.circle(img_current, index_finger, 12, (0, 0, 255), cv2.FILLED)
                    annotations[annotation_number].append(index_finger)
                    emit_slide_and_annotations(path_images[current_slide_index], annotations)

                elif fingers == [0, 1, 1, 1, 0]:  # Erase
                    if annotations and annotation_number >= 0:
                        annotations.pop(-1)
                        annotation_number -= 1
                        button_pressed = True
                        annotation_start = False
                        emit_slide_and_annotations(path_images[current_slide_index], annotations)

            if button_pressed:
                button_counter += 1
                if button_counter > button_delay:
                    button_counter = 0
                    button_pressed = False

            for i in range(len(annotations)):
                for j in range(len(annotations[i])):
                    if j != 0:
                        cv2.line(img_current, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 12)

            img_small = cv2.resize(img, (ws, hs))
            h, w, _ = img_current.shape
            img_current[0:hs, w - ws:w] = img_small

            cv2.imshow("Webcam", img)
            cv2.imshow("Slides", img_current)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    socketio.run(app, debug=True)
