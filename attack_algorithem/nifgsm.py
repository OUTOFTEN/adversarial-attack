import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from ReTransform import Re_transforms

class NIFGSM:
    def __init__(self,esp,steps,u,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        self.esp=esp
        self.u=u
        self.steps=steps
        self.device=device
    def attack(self,model,inputs,labels,clip_min,clip_max):
        noise=torch.zeros_like(inputs,requires_grad=True)
        g=torch.zeros_like(inputs)
        a=self.esp/self.steps
        transN = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        adv=inputs.clone()
        for i in range(0,self.steps):
            adv=noise.clone()+inputs
            adv=torch.autograd.Variable(adv,requires_grad=True)
            #计算获得nes变量
            nes=adv+a*self.u*g
            logits=model(nes)
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
        # adv_=Re_transforms(adv,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # adv=torch.clip(adv_,0,1)
        # adv=transN(adv)
        return adv