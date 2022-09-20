# Welcome to README.md file!


## Description files and directories

> **data** - the directory with the data for classification and detection. The detection data has YOLOv5 format and contains three classes [tigers, leopards, empty]. The class empty is about 10% from all volume dataset. The classification data contains two classes [tigers, leopards] \
> **weights** - the directory with weights for classification and detection tasks. The weights for classification pretrained on big dataset that we didn't publish (29 classes for classification). \
> **utils** - the directory with additional .py files for **train_cls.py** file \
> **results_test** - the directory with confusion matrix on the test data

<br/>

> **results** - the directory with results of training
>> **config.yaml** - the configuration file with training params \
>> **mapping.yaml** - the file with classes on which the classifier was trained \
>> **.pt** - the weights of neural network \
>> **results_.json** - the results of training with losses and accuracy \
>> **learning_curve.png** - the curves of training and validation \
>> **events.out.tfevents...** - the results for tensorboard \
>
<br/>

> **test_cls.py** - the file for testing classifier \
> **train_cls.py** - the file for training neural network

<br/>

## How to use *train_cls.py*.
> If you want to reproduce the results on "tigers_vs_leopards" dataset you need to specify path to *config.yaml* file that is located in *configs/config.yaml*. In this case, the weights that are trained on 29 classes will be used.

>> `$ python train_cls.py -c configs/config.yaml` 

> You can change some parameters like number of epochs, batch_size, input_size ant etc. The path to training and validation data is specified by default. If you want to use another dataset you should change *train_dir*, *val_dir* params.
>> `$ python train_cls.py -c configs/config.yaml --epochs 10 --batch_size 32 --input_size 256 --loss smooth --train_dir path/to/your/train/data --val_dir path/to/your/val/data`
> 

> When training is completed, its results are saved in the **results** directory. Read above about the structure of the **results** directory.

<br/>

## How to use *test_cls.py*
> If you want to evaluate the performance of a neural network on test data, you need to specify the path to the directory with the weights and test data
>> `$ python test_cls.py --pt_w weights/Classification/tigers_vs_leopards/resnest101e --pt_data data/Classificationtigers_vs_leopards/test`

>> or you can use your trained weights:

>> `$ python test_cls.py --pt_w path/to/your/results --pt_data data/Classification/tigers_vs_leopards/test`

> After testing the results will be saved in **results_test**

<br/>

## How to reproduce detection results
1. You need to clone git repository with YOLOv5
>> `$ git clone https://github.com/ultralytics/yolov5.git`
2. Then you need to install all libraries using *requirements.txt* for yolov5
>> `$ pip install -r requirements.txt`
3. After that, you need to train YOLOV5 using our pretrained weights
>> `$ python train.py --imgsz 1280 --epochs 10 --data data/Detection/tigers_vs_leopards/animals.yaml --weights weights/Detection/YOLOv5_L6/weights/best.pt --single-cls --batch 2`
4. To evaluate trained network on the test dataset you need to use val.py file in YOLOv5 repository
>> `$ python val.py --test --imgsz 1280 --data data/Detection/tigers_vs_leopards/animals.yaml --weights weights/Detection/YOLOv5_L6/weights/best.pt --single-cls --batch 24 --single-cls`
