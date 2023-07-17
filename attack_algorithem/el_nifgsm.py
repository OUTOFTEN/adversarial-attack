import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from ReTransform import Re_transforms

class EL_NIFGSM:
    def __init__(self,esp,steps,u,m=5,floor=1/2,device=torch.device('cpu')) -> None:
        self.esp=esp
        self.u=u
        self.steps=steps
        self.device=device
        self.m=m
        self.floor=floor

    def u_scale(self,x,i,min,max):
        nes=x*i
        nes=torch.clip(nes,min,max)
        return nes
    def scale(self,x,i):
        nes=x*((self.floor)**i)
        return nes

    def attack(self,model,inputs,labels,targeted=False,clip_min=-1,clip_max=1):
        noise=torch.zeros_like(inputs,requires_grad=True)
        g=torch.zeros_like(inputs)
        a=1.0/255
        multiplier = -1 if targeted else 1
        # transN = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for i in range(0,self.steps):
            adv=noise.clone()+inputs
            adv=torch.autograd.Variable(adv,requires_grad=True)
            #计算获得nes变量
            nes=adv+a*self.u*g
            logits=model(nes)
            logits_1=model(self.scale(nes,1))
            logits_2=model(self.scale(nes,2))
            logits_3 = model(self.scale(nes,3))
            logits_4 = model(self.scale(nes,4))
            f_logits=(logits+logits_1+logits_2+logits_3+logits_4)/5
            ce_loss=nn.CrossEntropyLoss()
            loss=ce_loss(f_logits,labels)
            loss.backward()

            adv_grad=adv.grad.clone()
            adv.grad.data.zero_()
            g=self.u*g+(adv_grad/(torch.mean(torch.abs(adv_grad),[1,2,3],keepdim=True)))

            adv=adv+a*torch.sign(multiplier*g)
            adv=torch.clip(adv,clip_min,clip_max)
            diff=adv-inputs
            noise=torch.clip(diff,-self.esp,self.esp)
            model.zero_grad()
        # adv_=Re_transforms(adv,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # adv=torch.clip(adv_,0,1)
        # adv=transN(adv)
        return adv,noise