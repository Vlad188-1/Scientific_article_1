# Welcome!

## Description files and directories

<details open>
<summary>To reproduce article results you need to use this data:</summary>

- [Detection and classification datasets](https://doi.org/10.6084/m9.figshare.21162829.v3) 
- [Model weights for detection and classification](https://doi.org/10.6084/m9.figshare.20982226.v2) 🚀 RECOMMENDED

</details>

> **test_cls.py** - the file for testing classifier \
> **train_cls.py** - the file for training neural network

## How to use *train_cls.py*.

> Install all required libraries
```bash
$ pip install -r requrements_cls.txt
```

>If you want to reproduce the results on **"tigers_vs_leopards"** dataset you need to specify path to *config.yaml* file 
that is located in *configs/config.yaml*. In this case, the weights that are trained on **29** classes will be used.

```bash
$ python train_cls.py -c configs/config.yaml
```

>You can change some parameters like number of epochs, batch_size, input_size ant etc. The path to training and
validation data is specified by default. If you want to use another dataset you should change *train_dir*, *val_dir*
params.
```bash
$ python train_cls.py -c configs/config.yaml --epochs 10 --batch_size 32 --input_size 256 --loss smooth --train_dir path/to/your/train/data --val_dir path/to/your/val/data
```

>When the network training is completed, directory **results_train** will be created, which will contain the following files:
- **config.yaml** - the configuration file with training params
- **mapping.yaml** - the file with classes on which the classifier was trained
- **.pt** - the weights of neural network
- **results_.json** - the results of training with losses and accuracy
- **learning_curve.png** - the curves of training and validation
- **events.out.tfevents...** - the results for tensorboard


## How to use *test_cls.py*

If you want to evaluate the performance of a neural network on test data, you need to specify the path to the
directory with the weights and test data
```bash
$ python test_cls.py --pt_w weights/Classification/tigers_vs_leopards/resnest101e --pt_data data/Classificationtigers_vs_leopards/test
```

or you can use your trained weights:

```bash
$ python test_cls.py --pt_w path/to/your/weights --pt_data data/Classification/tigers_vs_leopards/test
```
After testing, directory results_test will be created where the confusion matrix will be stored.


## How to reproduce detection results

1. You need to clone git repository with YOLOv5

```bash
$ git clone https://github.com/ultralytics/yolov5.git
```

2. Then you need to install all libraries using *requirements.txt* for yolov5

```bash
$ pip install -r requirements.txt
```

3. After that, you need to train YOLOV5 using our pretrained weights

```bash
$ python train.py --imgsz 1280 --epochs 10 --data data/Detection/tigers_vs_leopards/animals.yaml --weights weights/Detection/YOLOv5_L6/weights/best.pt --single-cls --batch 2
```

4. To evaluate trained network on the test dataset you need to use val.py file in YOLOv5 repository

```bash
$ python val.py --test --imgsz 1280 --data data/Detection/tigers_vs_leopards/animals.yaml --weights weights/Detection/YOLOv5_L6/weights/best.pt --single-cls --batch 24 --single-cls
```
