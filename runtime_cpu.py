import cv2
import numpy as np
import time
import argparse
def parser():
    parser = argparse.ArgumentParser(description="Darknet Image classification")
    parser.add_argument("--batch_size", default=1, type=int,help="number of images to be processed at the same time")
    parser.add_argument("--top", default=3, type=int,help="show label top predictions")
    parser.add_argument("--csi", action="store_true",help="Set --csi for using pi-camera")
    parser.add_argument("--webcam", type=str, default=None,help="Take inputs from webcam /dev/video*.")
    parser.add_argument('--image', type=str, default=None,help='path to image file name')
    parser.add_argument("--video",type=str,default=None,help="Path to video file.")
    parser.add_argument("--weights", default=None,help="yolo weights path")
    parser.add_argument("--config_file", default=None,help="path to config file")
    parser.add_argument("--label", default=None,help="path to labels.txt file")
    return parser.parse_args()
def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=640,
    display_height=480,
    framerate=21,
    port=0,
    flip_method=0):
    return ("nvarguscamerasrc sensor_id=%d ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink drop=True"
            %(  port,
		capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,))
def open_cam_usb(dev,width,height,USB_GSTREAMER):
    if USB_GSTREAMER:
        gst_str = ('v4l2src device=/dev/video{} ! '
                   'video/x-raw, width=(int)640, height=(int)480 ! '
                   'videoconvert ! appsink').format(dev)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        return cv2.VideoCapture('/dev/video'+str(dev))
args = parser()
if args.csi:
    print("csi using")
    cam = cv2.VideoCapture(gstreamer_pipeline(),cv2.CAP_GSTREAMER)
elif args.image:
    print("image for classification")
    print(args.image)
elif args.webcam:
    print('webcam using')
    cam = open_cam_usb(int(args.webcam),USB_GSTREAMER=True)
elif args.video:
    print('video for classification')
    cam = cv2.VideoCapture(args.video)
else:
    print('None source for input need image, video, csi or webcam')
    sys.exit()
net = cv2.dnn.readNetFromDarknet(args.config_file,args.weights)
classes = []
with open(args.label, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
while True:
    if args.image:
        frame = cv2.imread(args.image)
    else:
        _,frame = cam.read()
    blob = cv2.dnn.blobFromImage(frame,1.0/255.0, (256, 256), (0, 0, 0), True, crop=False)
    t1 = time.time()
    net.setInput(blob)
    outs = net.forward(output_layers)
    fps = 1.0/(time.time()-t1)
    classification = outs[0].reshape(len(classes))
    inx = classification.argmax()
    cv2.putText(frame, str(round(fps,2))+' fps', (11, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (32, 32, 32),4, cv2.LINE_AA)
    cv2.putText(frame, str(round(fps,2))+' fps', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1, cv2.LINE_AA)   
    cv2.putText(frame,'{} {:.2f}'.format(classes[inx], round(classification[inx]*100,2)), (11, (40)), cv2.FONT_HERSHEY_PLAIN, 1.0, (32,32, 32), 4, cv2.LINE_AA)
    cv2.putText(frame,'{} {:.2f}'.format(classes[inx], round(classification[inx]*100,2)), (10, (40)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('Inference', frame)
    cv2.moveWindow('Inference',0,0)
    if cv2.waitKey(1) == ord('q'):
        break
if args.image:
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()
    cam.release()


    