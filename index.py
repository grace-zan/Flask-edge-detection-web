from flask import Flask, render_template, request, Response
import cv2
import os
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)
threshold_value = 100
rainbow_mode = False
hue = 0

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


@app.route('/about', methods=["GET"])
def about():
    return render_template('about.html')

@app.route('/convert', methods=['GET', 'POST'])
def convert():
    global threshold_value
    global rainbow_mode
    if request.method =='POST':
        action= request.form["threshhold"]
        if action == 'increase':  
            threshold_value = threshold_value + 10
            if threshold_value > 255:                    
                threshold_value = 255
        elif action == 'decrease':
            threshold_value = threshold_value - 10
            if threshold_value < 0:
                threshold_value = 0
        elif action == 'rb':
            rainbow_mode = not rainbow_mode

    gen_frames(threshold_value, rainbow_mode)
    return render_template('convert.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(threshold_value, rainbow_mode), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/uploads', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'photo' in request.files:
            photo = request.files['photo']
            filename = photo.filename
            file_path = os.path.join(app.root_path, 'static/result/og.jpg')
            photo.save(file_path)

            detection_option = request.form.get('Detection Option')

            if detection_option == 'Contour':
                edges_path = contour_edges(file_path)
            elif detection_option == 'Sobel':
                edges_path = sobel_edges(file_path)
            elif detection_option == 'Canny':
                edges_path = canny_edges(file_path)

            return render_template('results.html', original=filename, edges=os.path.basename(edges_path))
        return 'No photo uploaded.'

def canny_edges(image_path):
    image = cv2.imread(image_path, 0)
    img_blur = cv2.GaussianBlur(image, (5,5), 0)
    edges = cv2.Canny(img_blur, 140, 180)
    edges_path = os.path.join(app.root_path, 'static/result/ed.jpg')
    cv2.imwrite(edges_path, edges)
    return edges_path

def sobel_edges(image_path):
    image = cv2.imread(image_path, 0)
    img_blur = cv2.GaussianBlur(image, (5,5), 0)
    edges = cv2.Sobel(img_blur, cv2.CV_32F, 1, 1, ksize=5)
    # sobely = cv2.Sobel(img_blur, cv2.CV_32F, 0, 1, ksize=5)
    # edges = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    edges_path = os.path.join(app.root_path, 'static/result/ed.jpg')
    cv2.imwrite(edges_path, edges)
    return edges_path

def contour_edges(image_path):
    image = cv2.imread(image_path, 0)
    ret, image_threshold = cv2.threshold(image, 140, 255, 0)
    contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape
    image_outlines_only = np.zeros((height, width), np.uint8)
    # image_color = (0, 0, 0)
    # image_outlines_only[:] = image_color
    edges = cv2.drawContours(image_outlines_only, contours, -1, (255, 255, 255), 1)
    edges_path = os.path.join(app.root_path, 'static/result/ed.jpg')
    cv2.imwrite(edges_path, edges)
    return edges_path

def gen_frames(threshold_value, rainbow_mode):  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break

        else:
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, image_threshold = cv2.threshold(image_gray, threshold_value, 255, 0)
            contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            height, width, channels = frame.shape
            image_outlines = np.zeros((height,width,channels), np.uint8)
            image_color = (255,255,255)
            image_outlines[:] = image_color
            outline_thickness = 2
            outline_color = (0,0,0)
            contour_id = -1

            if rainbow_mode:
                outline_color = get_next_color()
                image_color = (0,0,0)
                image_outlines[:] = image_color
            else:
                outline_color = (0,0,0)
            
            frame_edges = cv2.drawContours(image_outlines, contours, contour_id, outline_color, outline_thickness)
            ret, buffer = cv2.imencode('.jpg', frame_edges)
            frame = buffer.tobytes()
           
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def get_next_color():  
    # https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB_alternative
    global hue
    hue = hue + 10
    r = HSV_to_RGB_aux(5) * 255
    g = HSV_to_RGB_aux(3) * 255
    b = HSV_to_RGB_aux(1) * 255

    return (r,g,b)


def HSV_to_RGB_aux(n):
    s = 1
    v = 1
    k = (n + hue / 60) % 6
    return v - v*s*max(0 , min(k, 4 - k, 1)) 

if __name__ == '___main__':
    app.run(debug= True)
