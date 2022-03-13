import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize,ToPILImage,ToTensor
import random

class DI_NIFGSM:
    def __init__(self,esp,steps,u,prob=0.5,resize_rate=0.9,device=torch.device('cpu')) -> None:
        self.esp=esp
        self.u=u
        self.steps=steps
        self.device=device
        self.prob=prob
        self.resize_rate=resize_rate

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.prob else x



    def attack(self,model,inputs,labels,clip_min=-1,clip_max=1):
        noise=torch.zeros_like(inputs,requires_grad=True)
        g=torch.zeros_like(inputs)
        a=self.esp/self.steps
        for i in range(0,self.steps):
            adv=noise.clone()+inputs
            adv=torch.autograd.Variable(adv,requires_grad=True)
            #计算获得nes变量
            nes=adv+a*self.u*g
            logits=model(self.input_diversity(nes))
            ce_loss=nn.CrossEntropyLoss()
            loss=ce_loss(logits,torch.autograd.Variable(labels))
            loss.backward()

            adv_grad=adv.grad.clone()
            adv.grad.data.zero_()
            g=self.u*g+(adv_grad/(torch.mean(torch.abs(adv_grad),[1,2,3],keepdim=True)))

            adv=adv+a*torch.sign(g)
            adv=torch.clip(adv,clip_min,clip_max)
            diff=adv-inputs
            noise=torch.clip(diff,-self.esp,self.esp)
            model.zero_grad()
        return adv