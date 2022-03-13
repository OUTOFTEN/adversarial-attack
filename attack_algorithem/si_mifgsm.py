import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class SI_MIFGSM:
    def __init__(self,esp,steps,u,m=5,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        self.esp=esp
        self.u=u
        self.steps=steps
        self.device=device
        self.m=m

    def scale(self,x,i):
        nes=x/(2**i)
        return nes

    def attack(self,model,inputs,labels,clip_min=-1,clip_max=1):
        noise=torch.zeros_like(inputs,requires_grad=True)
        g=torch.zeros_like(inputs)
        a=self.esp/self.steps
        for i in range(0,self.steps):
            adv=noise.clone()+inputs
            adv=torch.autograd.Variable(adv,requires_grad=True)
            logits=model(adv)
            ce_loss=nn.CrossEntropyLoss()
            g_1=torch.zeros_like(inputs)
            for j in range(self.m):
                tnes=self.scale(adv,j)
                logits=model(tnes)
                loss=ce_loss(logits,labels)
                loss.backward()
                g_1+=adv.grad.clone()
                adv.grad.data.zero_()
                model.zero_grad()
            g_avg=g_1/self.m
            g=self.u*g+(g_avg/(torch.mean(torch.abs(g_avg),[1,2,3],keepdim=True)))

            adv=adv+a*torch.sign(g)
            adv=torch.clip(adv,clip_min,clip_max)
            diff=adv-inputs
            noise=torch.clip(diff,-self.esp,self.esp)
        return adv