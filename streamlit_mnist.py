import io
import os
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms
import wget
import torch.nn as nn

device = torch.device('cpu')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(28**2, 100, bias=False)
        self.bn1 = nn.BatchNorm1d(100)
        self.linear2 = nn.Linear(100, 50, bias=False)
        self.bn2 = nn.BatchNorm1d(50)
        self.linear3 = nn.Linear(50, 10, bias=False)
        self.R = nn.ReLU()

        self.history = []
        self.batch_history = []
        self.val_history = []

    def forward(self, x):
        self.linear1ins, self.linear1outs = [], []
        self.linear2outs = []

        x = x.to(dtype=torch.float32)
        x = x.to(device)
        x = x.view(-1, 28**2)
        self.linear1ins.append(x)
        x = self.linear1(x)
        x = self.R(self.bn1(x))
        self.linear1outs.append(x)
        x = self.linear2(x)
        x = self.R(self.bn2(x))
        self.linear2outs.append(x)
        x = self.linear3(x)

        return x

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def load_model():
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
        
    model = torch.load('mnist_model.pt', map_location=map_location)
    model.eval()
    return model

def predict(model, categories, image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        st.write(categories[top5_catid[i]], top5_prob[i].item())

def main():
    st.title('Pretrained model demo')
    model = load_model()
    model = model.to(device)
    categories = [0,1,2,3,4,5,6,7,8,9]
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(model, categories, image)

if __name__ == '__main__':
    main()
