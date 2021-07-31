import argparse
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet
def parser():
    parser = argparse.ArgumentParser(description="Darknet Image classification")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    return parser.parse_args()


def check_arguments_errors(args):
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))
def load_images(images_path):
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))

def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])

def main():
    args = parser()
    check_arguments_errors(args)
    network, class_names, _  = darknet.load_network(args.config_file,args.data_file,args.weights,batch_size=args.batch_size)
    images = load_images(args.input)
    index = 0
    while True:
        if args.input:
            if index >= len(images):
                break
            image_name = images[index]
        else:
            image_name = input("Enter Image Path: ")
        prev_time = time.time()
        frame = cv2.imread(image_name)
        predictions = image_classification(frame,network,class_names)
        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        print(predictions[:3])
        cv2.imshow('Inference', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        index += 1
if __name__ == "__main__":
    main()
