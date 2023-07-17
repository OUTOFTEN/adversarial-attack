import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from ReTransform import Re_transforms

class VNIFGSM:
    def __init__(self,esp,steps,u,a=3.2/255,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        self.esp=esp
        self.u=u
        self.steps=steps
        self.device=device
        self.a=a
    def attack(self,model,inputs,labels,clip_min,clip_max):
        noise=torch.zeros_like(inputs,requires_grad=True)
        g=torch.zeros_like(inputs)
        a=self.a
        transN = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        adv=inputs.clone()
        variance=torch.zeros_like(inputs)
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
            model.zero_grad()
            v_grad=torch.zeros_like(adv)
            for v in range(0,20):
                adv_neighbor = adv+torch.empty_like(adv,dtype=torch.float).uniform_(-self.esp*1.5,self.esp*1.5)
                adv_neighbor=torch.autograd.Variable(adv_neighbor,requires_grad=True)
                v_logits=model(adv_neighbor)
                v_loss=ce_loss(v_logits,labels)
                v_loss.backward()
                v_grad+=adv_neighbor.grad.clone()
                adv_neighbor.grad.data.zero_()
            current_grad=adv_grad+variance
            g=self.u*g+(current_grad/(torch.mean(torch.abs(current_grad),[1,2,3],keepdim=True)))
            variance = v_grad/20.0 - adv_grad

            adv=adv+a*torch.sign(g)
            if clip_min==None or clip_max==None:
                adv_=Re_transforms(adv,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                adv=torch.clip(adv_,0,1)
                adv=transN(adv)
            else:
                adv=torch.clip(adv,clip_min,clip_max)
            diff=adv-inputs
            noise=torch.clip(diff,-self.esp,self.esp)
        return adv,noise