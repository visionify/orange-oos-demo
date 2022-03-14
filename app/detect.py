import argparse
import os
import sys
from pathlib import Path
from urllib.request import urlopen
import cv2
import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.general import (check_img_size,non_max_suppression, scale_coords, xyxy2xywh)
from utils.torch_utils import select_device
from utils.augmentations import letterbox



class Prediction:
    @torch.no_grad()
    def __init__(self):
        self.conf_thres = 0.2
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = True
        self.max_det = 1000
        self.device = ''
        self.device = select_device(self.device)
        self.model = DetectMultiBackend("weight/best.pt", device=self.device, dnn=False, data="weight/data.yaml")
        self.stride, self.names, self.pt, self.jit, self.onncmx, self.engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size((640, 640), s=self.stride)
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup


    @torch.no_grad()
    def predict(self,img0):

        # img0 = cv2.imread(img_source)
        img = letterbox(img0, 640, stride=self.model.stride, auto=self.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img=img.float()
        print(img)
        img /= 255
        if len(img.shape) == 3:
            img = img[None]

        pred = self.model(img)

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        s=""
        result={}
        prediction = []
        for i, det in enumerate(pred):  # per image
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    c = int(cls)
                    label =self.names[c]
                    prediction.append({"label_index":c,"label_name":label,"confidence":float(conf),"bounding_box":{"x":xywh[0],"y":xywh[1],"width":xywh[2],"height":xywh[3]}})
        result['predictions']=prediction
        print(result)
        return result

    def predict_url(self,image_url):
        with urlopen(image_url) as binary:
            arr = np.asarray(bytearray(binary.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
            return self.predict(img)

    def predict_image(self,image):
        # img = cv2.imread(image_path)
        # print(image)
        return self.predict(image)



if __name__ == "__main__":
    # opt = parse_opt()
    run("test_data/qwinix-2022-02-03-16h36m12s843_jpg.rf.fd139a615612d0624bc424200b893718.jpg")
