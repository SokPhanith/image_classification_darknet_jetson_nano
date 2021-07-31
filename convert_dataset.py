import os
import cv2
import argparse
import sys
from glob import glob
def parser():
    parser = argparse.ArgumentParser(description="Convert dataset Image classification")
    parser.add_argument("--path_dataset", default=None,help="dataset path")
    parser.add_argument("--new_path_dataset", default=None,help="new dataset path")
    return parser.parse_args()
args = parser()
if args.path_dataset and args.new_path_dataset:
    print('model path: ',args.path_dataset,' new dataset path',args.new_path_dataset)
else:
    print('Give path dataset and new path dataset')
    sys.exit()
count = 0
dataset_new = args.new_path_dataset
path_dataset = args.path_dataset
if not os.path.exists(dataset_new):
    os.makedirs(dataset_new)
for type_data in os.listdir(path_dataset):
    if not os.path.exists(dataset_new+'/'+type_data):
        os.makedirs(dataset_new+'/'+type_data)
    for single_class in os.listdir(path_dataset+'/'+type_data):
        for image_name in os.listdir(path_dataset+'/'+type_data+'/'+single_class):
            img = cv2.imread(path_dataset+'/'+type_data+'/'+single_class+'/'+image_name)
            new_name = str(count)+"_"+single_class+'.jpg'
            count += 1
            cv2.imwrite(dataset_new+'/'+type_data+'/'+new_name,img)
            print(dataset_new+'/'+type_data+'/'+new_name)
    if not os.path.exists(dataset_new+'/labels.txt'):
        with open(dataset_new+'/labels.txt', 'w') as f:
            for single_class in os.listdir(path_dataset+'/'+type_data):
                f.write(single_class)
                f.write("\n")
        f.close()
    if not os.path.exists(dataset_new+'/'+type_data+'.list'):
        with open(dataset_new+'/'+type_data+'.list', 'w') as f:
            for image_path in glob(dataset_new+'/'+type_data+'/*'):
                f.write(image_path)
                f.write("\n")
        f.close()
    if not os.path.exists(dataset_new+'/custom.data'):
        with open(dataset_new+'/custom.data','w') as f:
            f.write("classes = "+str(len(os.listdir(path_dataset+'/'+type_data))))
            f.write("\n")
            f.write("train = "+dataset_new+'/train.list')
            f.write("\n")
            f.write("valid = "+dataset_new+'/test.list')
            f.write("\n")
            f.write("labels = "+dataset_new+'/labels.txt')
            f.write("\n")
            f.write("backup = "+dataset_new+'/')
            f.write("\n")
            f.write("top = 2")
            f.write("\n")
        f.close()
    
