import torch
import torch.nn as nn
import torch.nn.functional as F

class MIFGSM:
    def __init__(self, esp,steps,u,device=torch.device('cpu')):
        self.esp=esp
        self.device=device
        self.steps=steps
        self.u=u

    def attack(self,model,inputs,labels,clip_min=-1,clip_max=1):
        noise=torch.zeros_like(inputs,requires_grad=True)
        a=self.esp/self.steps
        g = torch.zeros_like(inputs)
        for i in range(self.steps):
            adv = noise.clone() + inputs
            adv=torch.autograd.Variable(adv,requires_grad=True)
            logits=model(adv)
            ce_loss=nn.CrossEntropyLoss()
            loss=ce_loss(logits,labels)

            loss.backward()

            adv_grad = adv.grad.clone()
            adv.grad.data.zero_()
            g=self.u*g+(adv_grad/(torch.mean(torch.abs(adv_grad),[1,2,3],keepdim=True)))
            adv=adv+a*torch.sign(g)
            adv=torch.clip(adv,clip_min,clip_max)
            diff=adv-inputs
            noise=torch.clip(diff,-self.esp,self.esp)
            model.zero_grad()
        return adv
