from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# VisionTransformer Class
class VisionTransformer:
    def __init__(self) -> None:
        self.clf = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to("cuda")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.optimizer = Adam(self.clf.parameters(),lr=1e-5)
        self.clf.classifier = torch.nn.Linear(in_features=768,out_features=2,bias=True)
        self.clf.num_labels = 2
    
    def fit(self, X_train, y_train) -> None:
        self.clf.train()
        template_imgs = np.reshape(X_train,(412,3,224,224))

        for img in tqdm(range(template_imgs.shape[0])):
            label = torch.tensor(y_train[img],dtype=torch.long).to('cuda')
            img = self.feature_extractor(images=template_imgs[img], return_tensors="pt")
            img = img.to('cuda')
            outputs = self.clf(**img,labels=label)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        


    def predict(self, query_img) -> np.array:
        self.clf.eval()
        query_img = np.reshape(query_img,(3,224,224))
        with torch.no_grad():
            img = self.feature_extractor(images=query_img, return_tensors="pt")
            img = img.to('cuda')
            outputs = self.clf(**img)
            logits = outputs.logits
            y_pred = logits.argmax(-1).item()
            return y_pred


class CNN:
    def __init__(self) -> None:
        self.device = 'cuda'
        self.clf = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(self.device)
        self.clf.fc = torch.nn.Linear(in_features=512,out_features=2,bias=True).to(self.device)
        self.optimizer = Adam(self.clf.parameters(),lr=1e-5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def fit(self, X_train, y_train) -> None:
        self.clf.train()
        template_imgs = X_train

        for img in tqdm(range(template_imgs.shape[0])):
            label = torch.tensor(y_train[img],dtype=torch.long).to(self.device)
            img = self.preprocess(Image.fromarray(np.uint8(template_imgs[img]))).unsqueeze(0)
            img = img.to(self.device)

            outputs = self.clf(img)
            loss = self.criterion(outputs,label.unsqueeze(0))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def predict(self, query_img) -> np.array:
        self.clf.eval()
        with torch.no_grad():
            img = self.preprocess(Image.fromarray(np.uint8(query_img))).unsqueeze(0)
            img = img.to(self.device)
            outputs = self.clf(img)
            probabilities = torch.nn.functional.softmax(outputs, dim=0)
            y_pred = torch.argmax(probabilities)
            return y_pred


class FCC:
    # TODO implement a simple MLP to use as a baseline
    def __init__(self,img_shape) -> None:
        self.device = 'cuda'
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(img_shape*img_shape*3,256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256,256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256,256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256,256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256,2),
            torch.nn.Softmax()
        )
        self.optimizer = Adam(self.clf.parameters(),lr=0.0005)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.clf.to('cuda')
    
    def fit(self, X_train, y_train):
        self.clf.train()
        template_imgs = X_train

        for img in tqdm(range(template_imgs.shape[0])):
            label = torch.tensor(y_train[img],dtype=torch.long).to(self.device)
            img = torch.flatten(torch.Tensor(template_imgs[img]))
            img = img.to(self.device)

            outputs = self.clf(img)
            loss = self.criterion(outputs.unsqueeze(0),label.unsqueeze(0))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def predict(self, query_img):
        self.clf.eval()
        with torch.no_grad():
            img = torch.flatten(torch.Tensor(query_img))
            img = img.to(self.device)
            probabilities = self.clf(img)
            # probabilities = torch.nn.functional.softmax(outputs, dim=0)
            y_pred = torch.argmax(probabilities)
            return y_pred
