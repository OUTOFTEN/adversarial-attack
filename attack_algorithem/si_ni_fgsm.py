import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from ReTransform import Re_transforms

class SI_NI_FGSM:
    def __init__(self,esp,steps,u,m=5,floor=1/2,device=torch.device('cpu')) -> None:
        self.esp=esp
        self.u=u
        self.steps=steps
        self.device=device
        self.m=m
        self.floor=floor

    def scale(self,x,i):
        nes=x*((self.floor)**i)
        return nes

    def attack(self,model,inputs,labels,clip_min=-1,clip_max=1):
        noise=torch.zeros_like(inputs,requires_grad=True)
        g=torch.zeros_like(inputs)
        a=self.esp/self.steps
        loss_velue=[0 for i in range(0,self.steps)]
        # transN = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for i in range(0,self.steps):
            adv=noise.clone()+inputs
            adv = torch.autograd.Variable(adv, requires_grad=True)
            #计算获得nes变量
            nes=adv+a*self.u*g
            #logits=model(nes)
            ce_loss=nn.CrossEntropyLoss()
            g_1=torch.zeros_like(inputs)
            for j in range(self.m):
                tnes=self.scale(nes,j)
                logits=model(tnes)
                loss=ce_loss(logits,labels)
                loss.backward()
                loss_velue[i]+=loss.item()
                g_1+=adv.grad.clone()
                adv.grad.data.zero_()
                model.zero_grad()
            g_avg=g_1/self.m
            # noise=g_1/torch.mean(torch.abs(g_1),[1,2,3],keepdim=True)
            g=self.u*g+(g_avg/(torch.mean(torch.abs(g_avg),[1,2,3],keepdim=True)))
            adv=adv+a*torch.sign(g)
            adv=torch.autograd.Variable(adv,requires_grad=False)
            adv=torch.clip(adv,clip_min,clip_max)
            diff=adv-inputs
            noise=torch.clip(diff,-self.esp,self.esp)
        # adv_=Re_transforms(adv,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # adv=torch.clip(adv_,0,1)
        # adv=transN(adv)
        return adv,loss_velue