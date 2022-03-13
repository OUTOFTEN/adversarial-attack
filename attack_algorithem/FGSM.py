import torch
import torch.nn as nn
import torch.nn.functional as F

class FGSM:
    def __init__(self,esp) -> None:
        self.esp=esp
    def attack(self,model,inputs,lables):
        batch_size=inputs.shape[0]
        delta=torch.zeros_like(inputs,requires_grad=True)
        adv=inputs+delta

        logits=model(adv)
        pre=logits.argmax(1)
        loss=F.nll_loss(logits,lables)

        model.zero_grad()
        loss.backward()
        grad=delta.grad.sign()
        delta=self.esp*grad

        adv=adv+delta
        adv=torch.clip(adv,0,1)

        return adv
