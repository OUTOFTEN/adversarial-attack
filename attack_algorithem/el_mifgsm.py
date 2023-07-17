import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class EL_MIFGSM:
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
            logits = model(adv)
            logits_1 = model(self.scale(adv, 1))
            logits_2 = model(self.scale(adv, 2))
            # logits_3=model(self.u_scale(nes,1.25,0,1))
            # logits_4=model(self.u_scale(nes,1.5,0,1))
            logits_3 = model(self.scale(adv, 3))
            logits_4 = model(self.scale(adv, 4))
            f_logits = (logits + logits_1 + logits_2 + logits_3 + logits_4) / 5
            ce_loss = nn.CrossEntropyLoss()
            loss = ce_loss(f_logits, labels)
            loss.backward()

            adv_grad = adv.grad.clone()
            adv.grad.data.zero_()
            g = self.u * g + (adv_grad / (torch.mean(torch.abs(adv_grad), [1, 2, 3], keepdim=True)))

            adv = adv + a * torch.sign(g)
            adv = torch.clip(adv, clip_min, clip_max)
            diff = adv - inputs
            noise = torch.clip(diff, -self.esp, self.esp)
            model.zero_grad()
        return adv