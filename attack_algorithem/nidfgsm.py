from re import X, escape
import torch
import torch.nn as nn
import torch.nn.functional as F

class NIDFGSM:
    def __init__(self,esp,steps,u,kd,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        self.esp=esp
        self.u=u
        self.steps=steps
        self.device=device
        self.kd=kd
    def attack(self,model,inputs,labels,clip_min=-1,clip_max=1):
        noise_all=torch.zeros_like(inputs,requires_grad=True)
        d=torch.zeros_like(inputs,requires_grad=True)
        g=torch.zeros_like(inputs)
        a=self.esp/self.steps
        for i in range(0,self.steps):
            adv=noise_all.clone()+inputs
            adv=torch.autograd.Variable(adv,requires_grad=True)
            x_nes=adv+a*self.u*g
            x_b=adv-a*d
            x_b=torch.autograd.Variable(x_b,requires_grad=True)
            ce_loss=nn.CrossEntropyLoss()

            logits=model(x_nes)
            loss=ce_loss(logits,labels)
            loss.backward(retain_graph=True)
            adv_grad=adv.grad.clone()
            adv.grad.data.zero_()
            model.zero_grad()

            logits_1=model(x_b)
            loss_1=ce_loss(logits_1,labels)
            loss_1.backward()
            x_b_grad=x_b.grad.clone()
            x_b.grad.data.zero_()
            model.zero_grad()

            b_t=(adv_grad/torch.mean(torch.abs(adv_grad),[1,2,3],keepdim=True))-(x_b_grad/torch.mean(torch.abs(x_b_grad),[1,2,3],keepdim=True))
            d=d+self.kd*b_t
            g=self.u*g+(adv_grad/torch.mean(torch.abs(adv_grad),[1,2,3],keepdim=True))-d
            adv=adv+a*torch.sign(g)
            adv=torch.clip(adv,clip_min,clip_max)
            diff=adv-inputs
            noise_all=torch.clip(diff,-self.esp,self.esp)
        return adv