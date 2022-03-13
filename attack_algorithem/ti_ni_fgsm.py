import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
import torchvision.transforms as transforms
from ReTransform import Re_transforms


class TI_NIFGSM:
    def __init__(self, esp, steps, u, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        self.esp = esp
        self.u = u
        self.steps = steps
        self.device = device

    def get_kernel(self, kernel_len, nsig):
        x = np.linspace(-nsig, nsig, kernel_len)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def attack(self, model, inputs, labels, clip_min=-1, clip_max=1):
        noise = torch.zeros_like(inputs, requires_grad=True)
        g = torch.zeros_like(inputs)
        a = self.esp / self.steps
        kernel = self.get_kernel(7, 3)
        kernel = torch.from_numpy(kernel).expand(3, 1, 15, 15).type(torch.float32)
        # kernel=torch.from_numpy(kernel).expand(3,3,15,15).type(torch.float32)
        adv=inputs.clone()
        for i in range(0, self.steps):
            adv = noise.clone() + inputs
            adv = torch.autograd.Variable(adv, requires_grad=True)
            # 计算获得nes变量
            nes = adv + a * self.u * g
            logits = model(nes)
            ce_loss = nn.CrossEntropyLoss()
            loss = ce_loss(logits, torch.autograd.Variable(labels))
            loss.backward()

            adv_grad = adv.grad.clone()
            adv.grad.data.zero_()
            noise_=F.conv2d(input=adv_grad, weight=kernel.cuda(), groups=3, padding='same',stride=1)
            g = self.u * g + (noise_ / (torch.mean(torch.abs(noise_), [1, 2, 3], keepdim=True)))

            adv = adv + a * torch.sign(g)
            # adv=adv+a*torch.sign(F.conv2d(input=g,weight=kernel,padding=int((15-1)/2)))
            adv = torch.clip(adv, clip_min, clip_max)
            diff = adv - inputs
            noise = torch.clip(diff, -self.esp, self.esp)
            model.zero_grad()
        return adv