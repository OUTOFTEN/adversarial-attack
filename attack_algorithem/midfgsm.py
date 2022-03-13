import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class MID_FGSM:
    def __init__(self,steps,esp,u,kd,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        self.steps=steps
        self.esp=esp
        self.u=u
        self.kd=kd
        self.device=device
    def attack(self,model,imgs,labels):
        a=self.esp/self.steps*2
        noise=torch.zeros_like(imgs,requires_grad=True)
        adv=imgs+noise.clone()
        g=0
        d=0
        for i in range(self.steps):
            # print(d)

            #获得b（中间变量）
            adv=torch.autograd.Variable(adv,requires_grad=True)
            logits_adv=model(adv)
            adv_loss=nn.CrossEntropyLoss()
            loss_adv=adv_loss(logits_adv,labels)
            model.zero_grad()
            loss_adv.backward()
            delta_adv=adv.grad

            if i==0:
                b=adv-a*delta_adv
            else:
                b=adv-a*d
            b=torch.autograd.Variable(b,requires_grad=True)

            #计算b的梯度
            logits_b=model(b)
            b_loss=nn.CrossEntropyLoss()
            loss=b_loss(logits_b,labels)
            model.zero_grad()
            loss.backward()
            delta_b=b.grad

            #更新d
            bt=delta_adv/torch.norm(delta_adv,p=1)-delta_b/torch.norm(delta_b,p=1)
            d=d+self.kd*bt
            #加d还是减d？
            g=self.u*g+delta_adv/torch.norm(delta_adv,p=1)-d
            adv=torch.clip(adv+a*torch.sign(g),-1,1)
        return adv