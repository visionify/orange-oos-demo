# Orange OOS Training 
## Repo Cloning

```
git clone https://github.com/visionify/orange-oos-demo.git
cd orange-oos-demo
git checkout training
```

## Setup
### Anaconda Installation [link](https://github.com/hrnbot/All-in-one/tree/main/Anaconda%20Install)
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash ~/Anaconda3-2021.05-Linux-x86_64.sh
source ~/.bashrc
```
|Note: This must activate ```(base)``` in terminal
|---|

### Create Environment
``` 
conda create -n yolo python=3.8 -y
conda activate yolo
```
|Note: This must activate ```(yolo)``` in terminal
|---|

### Install pip Dependencies
```
pip install -r requirements.txt
```

### Install Pytorch Dependencies [link](https://github.com/hrnbot/All-in-one/tree/main/Pytorch%20with%20Conda)
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
```
|Note: This Dependency is only for GPU CUDA 10.2+ for other check link
|---|

## Data Preparation
If your data is having any other annotation then Yolov5 Pytorch [Sample Dataset is here](https://drive.google.com/drive/folders/1PG-QEnzIUzmcYkxtnlnao0OebI04kIAt?usp=sharing)

### Convert Any Dataset to YOLO Dataset easily
- Sign up and login https://app.roboflow.com/
- Create New Project
- Select Images Folder with annotation and Upload
- Click on Upload 

|Note: Images and Annotation must show, Train(70%), val(20%) and test (10%)
|---|

- Generate New Version
  - None Preprocessing
  - Augmentation
    - Between -15° and +15°
    - Between -10% and +10%
    - Between -7% and +7%
    - Blurring 0 to 1px
  - Generate 3x
- Click on Export
  - Select Yolo v5 PyTorch
    - show download code
  - Click on Continue
- Terminal
- Copy the Code
  - Similar to ```curl -L "https://app.roboflow.com/ds/9VymdQ4dgl?key=JGPTA5BMvR" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip```

### Dataset Download
```
mkdir data_v1
cd data_v1
# Paste Above Copied URL
cd ..
```

### Modification of Yaml
```
sudo nano data_v1/data.yaml
```
Sample yaml_data file
```
train: data_v1/train/images
val: data_v1/valid/images
test: data_v1/test/images

nc: 12
names: ['bottle', 'box', 'can', 'candy', 'carton', 'container', 'eggs', 'empty-shelf', 'medical-box', 'medicine', 'pouch', 'shelf']
```

## Training
```
python train.py --data data_v1/data.yaml --cfg models/yolov5m.yaml --weights yolov5m.pt --batch-size 8
```
|Note: use --imgsz for image_size 640(default)
|---|

|Note: use --batch-size based on GPU Ram 8(default)
|---|

##