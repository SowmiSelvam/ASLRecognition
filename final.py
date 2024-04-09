import cv2, pickle
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import os
import sqlite3
import string
from PIL import Image

label_map = {char: index for index, char in enumerate(string.ascii_uppercase, start=0)}
label_map['del'] = 26
label_map['space'] = 27
label_map['nothing'] = 28

label_class = {index: char for index, char in enumerate(string.ascii_uppercase, start=0)}
label_class[26] = 'del'
label_class[27] = 'space'
label_class[28] = 'nothing'

label_tensor = torch.tensor([label_map[char] for char in label_map])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

criterion = nn.CrossEntropyLoss()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        modules = list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1]
        
        self.resnet50_feature_extractor = nn.Sequential(*modules)
        for param in self.resnet50_feature_extractor.parameters():
            param.requires_grad = True
        
        self.fc_layer = nn.Linear(2048, 29)

    def forward(self, images, labels):
        loss = None
        x = self.resnet50_feature_extractor(images)
        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)
        loss = criterion(x, labels)
        return loss

    def predict(self, images):
        labels_pred = None
        x = self.resnet50_feature_extractor(images)
        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)
        _, labels_pred = torch.max(x, dim=1)
        return labels_pred

model = Model()
model.load_state_dict(torch.load('/Users/sowmiyanarayanselvam/Documents/AirCloud/Rutgers/Spring24/ML2/Project/ML/model.pth', map_location=torch.device('cpu')))
model.eval()
# transform = ResNet50_Weights.DEFAULT.transforms()


def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def get_image_size():
    return (200, 200)

hist = get_hand_hist()
x, y, w, h = 0, 0, 640, 480
is_voice_on = False

def torch_process_image(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(get_image_size()),
        transforms.ToTensor(),
    ])
    
    img = transform(img)
    img = img.unsqueeze(0)  
    return img

def torch_predict(model, image):
    processed = torch_process_image(image)
    #print(type(processed))
    #processed = np.array(processed)
    with torch.no_grad():
        label_pred = model.predict(processed)
        pred_class = label_class[label_pred.item()]
    return pred_class

def get_pred_from_contour(contour, img):
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    save_img = img[y1:y1+h1, x1:x1+w1]
    pred_class = torch_predict(model, save_img)
    return pred_class

def get_img_contour_thresh(img):
    #img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    cv2.filter2D(dst, -1, disc, dst)
    blur = cv2.GaussianBlur(dst, (11,11), 0)
    blur = cv2.medianBlur(blur, 15)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.merge((thresh,thresh,thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    
    return img, contours, thresh

def recognize():
    cam = cv2.VideoCapture(1)
    if cam.read()[0] == False:
        cam = cv2.VideoCapture(0)
    
    # Define the bounding box size and position
    bbox_width = 800
    bbox_height = 800
    bbox_x = 200
    bbox_y = 200
    
    while True:
        img = cam.read()[1]
        
        # Extract the bounding box region
        bbox_img = img[bbox_y:bbox_y+bbox_height, bbox_x:bbox_x+bbox_width]
        
        img, contours, thresh = get_img_contour_thresh(bbox_img)
        
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                pred_class = get_pred_from_contour(contour, img)
                #print(pred_class)
                
                # Display the predicted character with the bounding box
                blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blackboard, "Predicted text- " + pred_class, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
                cv2.rectangle(img, (bbox_x, bbox_y), (bbox_x+bbox_width, bbox_y+bbox_height), (0, 255, 0), 2)
                res = np.zeros((max(img.shape[0], blackboard.shape[0]), img.shape[1] + blackboard.shape[1], 3), dtype=np.uint8)
                res[:img.shape[0], :img.shape[1]] = img
                res[:blackboard.shape[0], img.shape[1]:] = blackboard
                cv2.imshow("Recognizing gesture", res)
                cv2.imshow("thresh", thresh)
        
        keypress = cv2.waitKey(1)
        if keypress == ord('q'):
            break

recognize()