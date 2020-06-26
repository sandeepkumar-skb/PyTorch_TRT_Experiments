# https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/

import torch
from torchvision import models,transforms
from PIL import Image
import time
import argparse
import os



os.environ["TORCH_HOME"] = "."


# Model

def transform_image(image):
    transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                        )
                                    ])
    return transform(image)

def create_model(model_name=None):
    model = getattr(models, model_name)
    return model(pretrained=True)



'''
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O /tmp/cats_and_dogs_filtered.zip
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run torchvision models")
    parser.add_argument("--model", \
            default="resnet18", \
            choices= ["resnet18", "resnet50", "vgg16", "vgg19"], \
            help='Choose a model among ["resnet18", "resnet50", "vgg16", "vgg19"]')
    args = parser.parse_args()

    model = create_model(args.model)
    model.eval().cuda()

    for i in range(10):
        img = Image.open("cats_and_dogs_filtered/train/dogs/dog.{}.jpg".format(i))
        img_t = transform_image(img)
        batch_t = torch.unsqueeze(img_t, 0)
        batch_t = batch_t.cuda()
        t1 = time.time()
        out = model(batch_t)
        print("Iter {}: {:.4f}sec".format(i, time.time()-t1))
        with open('imagenet_classes.txt') as f:
            labels = [line.strip() for line in f.readlines()]
        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        print("Predictions: {}, Confidence: {:.2f}%".format(labels[index[0]], percentage[index[0]].item()))


