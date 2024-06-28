# Real-time-SafetyHelmet-Detection-Yolov5-Yolov8

The use of helmets in industrial settings is one of the most crucial factors directly related to life and safety. Therefore, it is necessary to issue warnings about not wearing helmets to ensure the safety of workers in these environments. However, the industrial managers cannot always be present there. To address this issue, we have implemented a model that detects whether workers are wearing helmets in real-time using YOLO.

## Used model version
- YOLOv5
- YOLOv8 (To be uploaded)

Please note that the following contents only contain environment settings and explanations based on the YOLOv5 model, considering that the contents of the YOLOv8 model have not been uploaded yet.<br><br>
We will train both models using the same custom dataset and compare their evaluation metrics to determine which model is more suitable for use in an actual industrial setting.

To train the models with the custom dataset, we first need to accurately specify the path to the custom dataset in the YAML file.
### file_path.py
List the datasets to be used for training in a text file with their absolute paths.
```
root_path = ''		#path of root

file_path = os.path.join(root_path, 'images/train')
valid_path = os.path.join(root_path, 'images/val')


with open(os.path.join(roo_path, 'train.txt'), 'w') as f:
...
with open(os.path.join(roo_path, 'valid.txt'), 'w') as f:
...
```
```
# train.txt
/absolute/path/to/dataset1.jpg
/absolute/path/to/dataset2.jpg
/absolute/path/to/dataset3.jpg
...
```
This allows the YOLO model to easily locate the training datasets. Once the listing is complete, specify the path to this text file in the `data.yaml` file.

### data.yaml
```
train: data/train.txt
val: data/valid.txt
```
# Train
The training environment was Anaconda prompt, and the training was performed within that environment after building a virtual environment. Accelerated training processing speed using GPU. Detailed instructions on enabling GPU acceleration are provided in the 'Dependencies' section.
```
python train.py --image 640 --batch 16 --epochs 50 --data data/data.yaml --cfg models/yolov5s.yaml -- weights weights/yolov5s.pt --device 0
```

# Inference

After training is complete, a directory named `runs` is created within the `yolov5` directory. Inside the `yolov5/runs/train/` path, directories such as `exp1`, `exp2`, etc., are generated for each training session. The `weight/best.pt` file in the latest folder can be used as the weight file for inference.
```
python detect.py --source=data\images\train\img.jpg --weights=runs\train\expN\weights\best.pt --img 640 --conf 0.5 --save-txt
```
|Parameters|Details|
|------|---|
|--source|image you want to infer|
|--img|Size of the image input to the network|
|--batch|Batch size|
|--epochs|Number of epochs|
|--weights|Pre-trained model for transfer leraning (e.f., specifying 'yolov5s.pt' will automatically download the model)|
|--name|Nmae under which the trained model will be saved|

<br>

```
detect: weights=['runs\\train\\expN\\wegihts\\best.pt'], source=data\images\train\img.jpg, data=data\coco128.yaml, imgsz=[640, 640], conf_thres=0.5, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=True, save_csv=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs\detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1
YOLOv5  2024-6-5 Python-3.12.3 torch-2.3.1+cu118 CUDA:0 (NVIDIA GeForce RTX 2070 SUPER, 8192MiB)

Fusing layers...
Model summary: 157 layers, 7037095 parameters, 0 gradients, 15.8 GFLOPs
image 1/1 yolov5\data\images\train\img.jpg: 640x640 3 Hardhats, 4 No-Masks, 1 No-Safety Vest, 7 Persons, 6 Safety Vests, 1 machienry, 3 vehicles, 6.0ms
Speed: 0.0ms pre-process, 6.0ms inference, 54.7ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs\detect\exp
1 labels saved to runs\detect\exp\labels
```
<img src="https://github.com/TF-polygon/SPH-Collision-Detection/assets/111733156/d3a990ee-ce89-4b22-95c7-cce87fc023cb" style="height: 400px;"><br>
- It can be confirmed that real-time object detection is possible through learned data using a simple webcam that can be easily connected to a PC.<br><br>
<img src="https://github.com/TF-polygon/SPH-Collision-Detection/assets/111733156/b9fe0d99-8376-4410-9d5d-8647ec067e8b">

<img src="https://github.com/TF-polygon/SPH-Collision-Detection/assets/111733156/102631b9-300f-4a05-a082-11d39c5a8d53"> <br>

- F1-Confidence Curve Comparison (value range: 0.0~1.0)<br>
The results showed that model v8 produced slightly better outcomes compared to model v5, with only a marginal difference. This similarity in results can be attributed to the fact that v8 is a modified version of v5, with approximately five key alterations such as module changes, backbone modifications, addition/removal of convolutional layers, changes in convolutional layer size, and separation of the head module.

# Conclusion
### Model Comparison
Comparison between YOLOv8 and YOLOv5 indicates that v8 performs slightly better.

### Core Significance of the Project
The most crucial aspect of this project is demonstrating that it is possible to create a highly usable model even with limited datasets in a rapidly changing practical environment.

### Future Applications
- Developing domain expansion models using the YOLO model.<br>
- As of now (as of 6/11), version upgrades have reached v10, suggesting that future models with even better performance can be trained and used with smaller datasets.

## Virtual Environment Setup for Experiment
Create a virtual environment
```
# Anaconda Prompt
conda create -n [venv name] python=necessary_version
```
Connect the virtual environment
```
conda activate [venv name]
```
In case it causes pip error in virtual environment
```
pip install [package]
Unable ~ processing Path\python.exe
```
- It's because of collision between base and virtual environment.
- you should delete pip environment variable of base and use virtual environment's pip.

## Dependencies
```
CUDA 11.8
CUDNN 9.1
Python 3.12.3
```
```
torch, torchaudio 2.3.1+cu118
torchaudio 0.18.1+cu118
ultralytics 8.2.28
opencv-python 4.10.0.82
tensorboard 2.16.2
tensorflow 2.16.1
```

### Reference
- https://github.com/ultralytics/yolov5

