# https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/


import torch
from torchvision import models,transforms
from PIL import Image

# Model
model = models.resnet18(pretrained=True)

transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                    )
                                ])


'''
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O /tmp/cats_and_dogs_filtered.zip
'''
img = Image.open("cats_and_dogs_filtered/train/dogs/dog.1.jpg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
model.eval()
out = model(batch_t)
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print("Predictions: {}, Confidence: {:.2f}%".format(labels[index[0]], percentage[index[0]].item()))


