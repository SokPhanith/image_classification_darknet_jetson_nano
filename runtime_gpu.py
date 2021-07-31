import argparse
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import sys
def parser():
    parser = argparse.ArgumentParser(description="Darknet Image classification")
    parser.add_argument("--batch_size", default=1, type=int,help="number of images to be processed at the same time")
    parser.add_argument("--top", default=3, type=int,help="show label top predictions")
    parser.add_argument("--csi", action="store_true",help="Set --csi for using pi-camera")
    parser.add_argument("--webcam", type=str, default=None,help="Take inputs from webcam /dev/video*.")
    parser.add_argument('--image', type=str, default=None,help='path to image file name')
    parser.add_argument("--video",type=str,default=None,help="Path to video file.")
    parser.add_argument("--weights", default="yolov4.weights",help="yolo weights path")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",help="path to data file")
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
def check_arguments_errors(args):
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))

def main():
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
    check_arguments_errors(args)
    network, class_names, _  = darknet.load_network(args.config_file,args.data_file,args.weights,batch_size=args.batch_size)
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    while True:
        if args.image:
            frame = cv2.imread(args.image)
        else:
            _,frame = cam.read()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        prev_time = time.time()
        detections = darknet.predict_image(network, darknet_image)
        fps = 1.0/(time.time() - prev_time)
        predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
        predictions = sorted(predictions, key=lambda x: -x[1])
        cv2.putText(frame, str(round(fps,2))+' fps', (11, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (32, 32, 32),4, cv2.LINE_AA)
        cv2.putText(frame, str(round(fps,2))+' fps', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1, cv2.LINE_AA)   
        for inx,(classes,percent) in enumerate(predictions[0:args.top]):
            cv2.putText(frame,'{} {:.2f}'.format(classes, round(percent*100,2)), (11, (40+20*inx)), cv2.FONT_HERSHEY_PLAIN, 1.0, (32,32, 32), 4, cv2.LINE_AA)
            cv2.putText(frame,'{} {:.2f}'.format(classes, round(percent*100,2)), (10, (40+20*inx)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('Inference', frame)
        cv2.moveWindow('Inference',0,0)
        if cv2.waitKey(1) == ord('q'):
            break
    if args.image:
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
        cam.release()
if __name__ == "__main__":
    main()
