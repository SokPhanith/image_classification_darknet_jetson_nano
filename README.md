## 1. Setup

Install darknet framework writing in C on Jetson Nano 

    cd ~
    git clone https://github.com/SokPhanith/image_classification_darknet_jetson_nano.git
    cd image_classification_darknet_jetson_nano
    git clone https://github.com/AlexeyAB/darknet
    rm darknet/Makefile
    cp Makefile darknet/
    cd darknet
    make
   
## 2. Download model 

Download Image Classification pretrain model 1000 classes from ImgaeNet dataset training in darknet framework

    cd ~
    cd image_classification_darknet_jetson_nano
    ./download_model.sh

## 3. Testing with darknet application build from C

    cd ~
    cd image_classification_darknet_jetson_nano/darknet
    ./darknet classifier predict cfg/imagenet1k.data cfg/darknet19.cfg ../weight/darknet19.weights data/dog.jpg
    ./darknet classifier predict cfg/imagenet1k.data cfg/darknet53.cfg ../weight/darknet53.weights

## 4. Testing with python by libdarknet.so librery

    cd ~
    cd image_classification_darknet_jetson_nano
    cp *.py darknet/
    cd darknet/
    python3 runtime_gpu.py --weights ../weight/darknet53.weights --config_file cfg/darknet53.cfg --data_file cfg/imagenet1k.data --csi
    python3 runtime_cpu.py --weights ../weight/darknet53.weights --config_file cfg/darknet53.cfg --data_file cfg/imagenet1k.data --image data/dog.jpg

## 5. Training with cifar dataset
    
    cd image_classification_darknet_jetson_nano/darknet/data
    wget https://pjreddie.com/media/files/cifar.tgz
    tar xzf cifar.tgz
    cd cifar
    find `pwd`/train -name \*.png > train.list
    find `pwd`/test -name \*.png > test.list
    cd ../..
    cp ../cifar.data cfg/
    cp ../cifar_small.cfg cfg/
    ./darknet classifier train cfg/cifar.data cfg/cifar_small.cfg
    ./darknet classifier valid cfg/cifar.data cfg/cifar_small.cfg backup/cifar_small_final.weights
    python3 runtime_gpu.py --weights backup/cifar_small_final.weights --config_file cfg/cifar_small.cfg --data_file cfg/cifar.data --image data/cifar/test/0_cat.png

## 6. Training with custom dataset darknet19 model

My custom dataset has 4 classes arduino, cnc, esp8266 and pyboard. In data/custom have test/ and train/. In a test/ and train/ have classes arduino, cnc, esp8266 and pyboard folder. Imgae was put in by there folder name.
If you have a different classes or more then this classes must edit [filters=4](https://github.com/SokPhanith/image_classification_darknet_jetson_nano/blob/main/darknet19.cfg) line 181 by count same your dataset.
    
    cd ~
    cd image_classification_darknet_jetson_nano/darknet
    python3 convert_dataset.py --path_dataset data/custom --new_path_dataset data/board
    cp ../darknet19.cfg data/board
    ./darknet classifier train cfg/board/custom.data cfg/board/darknet19.cfg
    ./darknet classifier valid cfg/board/custom.data cfg/board/darknet19.cfg cfg/board/darknet19_last.weights
    python3 runtime_gpu.py --weights cfg/board/darknet19_last.weights --config_file cfg/board/darknet19.cfg --data_file cfg/board/custom.data --image data/board/test/0_esp8266.jpg    
 
    
