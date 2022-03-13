import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class IFGSM:
    def __init__(self,
                steps,
                eps,
                device=torch.device('cpu')):
        self.steps = steps
        self.device = device
        self.eps = eps

    def attack(self, model, inputs, labels, targeted=False,clip_min=-1,clip_max=1):
        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        #adv = torch.ones_like(inputs, requires_grad=True)*inputs
        delta = torch.zeros_like(inputs, requires_grad=True)
        eps_iter=self.eps/self.steps

        for i in range(self.steps):
            adv = delta.clone() + inputs
            adv = torch.autograd.Variable(adv, requires_grad=True)
            logits = model(adv)
            pred_labels = logits.argmax(1)
            #loss = F.nll_loss(logits, labels)
            ce_loss = nn.CrossEntropyLoss()
            loss = ce_loss(logits,torch.autograd.Variable(labels))
            # loss = multiplier * ce_loss.item()
            #delta.retain_grad()
            #model.zero_grad()
            #loss.backward()
            loss.backward()

            delta = eps_iter * torch.sign(adv.grad)
            # print(self.eps_iter)
            adv.grad.data.zero_()

            adv = adv + delta
            adv = torch.clamp(adv,clip_min,clip_max)
            diff = adv - inputs
            delta = torch.clamp(diff, -self.eps, self.eps)
            model.zero_grad()
        return adv

