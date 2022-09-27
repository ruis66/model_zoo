import os
import sys
sys.path.append(".")

import cv2
import torch
from torchvision import transforms

from model_zoo.classification.colornet import ColorNet
from model_zoo.classification.coatnet import coatnet_5
idx_to_class = ["green", "red", "white", "yellow"]


class ClsNet:
    def __init__(self, model_f, net, gpu_ids=0):
        device = torch.device(f"cuda:{int(gpu_ids)}")
        net.load_state_dict(torch.load(model_f, map_location=device))
        net.eval()
        net.to(device)
        
        self.net = net
        self.device = device
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
    def infer(self, img):
        with torch.no_grad():
            img = self.transform(img).unsqueeze(0).to(self.device)
            outputs = self.net(img)
            return outputs

def gen_color_net(model_f, gpu_ids=0):
    net = ColorNet(num_classes=4)
    return ClsNet(model_f, net, gpu_ids)

def gen_coat_net(model_f, gpu_ids=0):
    net = coatnet_5()
    return ClsNet(model_f, net, gpu_ids)

if __name__ == '__main__':
    model_f = "checkpoints/colornet_epoch_15_acc_0.99706_loss_0.74674.pth"
    img_p = "tests/data/cls/yellow/211116155319214_camera7_0_0.jpg"
    img = cv2.imread(img_p, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    net = gen_color_net(model_f)
    outputs = net.infer(img)
    color = idx_to_class[outputs.argmax(1).item()]
    print(color)


    model_f = "checkpoints/CoAtNet_epoch_10.pth"
    net = gen_coat_net(model_f)
    outputs = net.infer(img)
    color = idx_to_class[outputs.argmax(1).item()]
    print(color)


