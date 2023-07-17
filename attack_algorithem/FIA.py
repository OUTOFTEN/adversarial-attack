import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from ReTransform import Re_transforms
import numpy as np

class FIA:
    def __init__(self, esp,steps,u,a=6.4/255,ens=30,drop_pb=0.9,device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.esp=esp
        self.device=device
        self.steps=steps
        self.u=u
        self.a=a
        self.ens=ens
        self.drop_pb=drop_pb
        self.feature_map=[]
        self.weight=[]

    # def get_weight(self,x,label,model):
    #     # baseline = np.zeros(x.shape)
    #     weight=[]
    #     logits=model(x)
    #     logits=nn.functional.softmax(logits,1)
    #     score=logits[:,label]
    #     loss=torch.sum(score)
    #     loss.backward()
    #     weight_grad=torch.mean(y_grad,0)
    #     return weight_grad

    def get_FIA_loss(self,x,models,weights):
        self.feature_map.clear()
        loss=0
        for idx,model in enumerate(models):
            logits=model(x)
            attribution=self.feature_map[idx]*weights[idx]
            loss+=torch.sum(attribution)/attribution.numel()
        return loss

    def forward_hook(self,model,input,output):
        self.feature_map.append(output)

    def backward_hook(self,model,grad_input,grad_output):
        self.weight.append(grad_output[0])

    def attack(self,models:list,layers:list[str],inputs:torch.tensor,labels:torch.tensor,clip_min=-1,clip_max=1):
        self.weight.clear()
        self.feature_map.clear()
        for model,layer in zip(models,layers):
            if hasattr(model,layer):
                getattr(model,layer).register_backward_hook(self.backward_hook)
                getattr(model,layer).register_forward_hook(self.forward_hook)
            else:
                raise Exception(f'can not find {layer} in {model.__class__.__name__}')
        noise=torch.zeros_like(inputs,requires_grad=True)
        a=self.a
        g = torch.zeros_like(inputs)
        transN = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        weights = []
        weight_tensor=0
        for i in range(self.steps):
            adv = noise.clone() + inputs
            adv=torch.autograd.Variable(adv,requires_grad=True)
            if i == 0:
                for model in models:
                    temp_weight=0
                    self.weight.clear()
                    for l in range(self.ens):
                        self.feature_map.clear()
                        mask = np.random.binomial(1, self.drop_pb, size=(inputs.shape[0],inputs.shape[1],inputs.shape[2],inputs.shape[3]))
                        mask=torch.from_numpy(mask).to(self.device)
                        image_tmp=inputs.clone()*mask
                        logits=model(image_tmp)
                        logits=nn.functional.softmax(logits,1)
                        score=logits[:,labels]
                        loss=torch.sum(torch.mean(score,1))
                        loss.backward()
                    for w in self.weight:
                        temp_weight+=w
                    weights.append(temp_weight)
                for idx,weight_tensor in enumerate(weights):
                    weight_tensor=weight_tensor.to(self.device)
                    square=torch.sum(torch.square(weight_tensor),[1,2,3],keepdim=True)
                    weight_tensor=-weight_tensor/torch.sqrt(square)
                    weights[idx]=weight_tensor

            loss=self.get_FIA_loss(adv,models,weights)

            loss.backward()

            adv_grad = adv.grad.clone()
            adv.grad.data.zero_()
            g=self.u*g+(adv_grad/(torch.mean(torch.abs(adv_grad),[1,2,3],keepdim=True)))
            adv=adv+a*torch.sign(g)
            if clip_min==None or clip_max==None:
                adv_=Re_transforms(adv,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                adv=torch.clip(adv_,0,1)
                adv=transN(adv)
            else:
                adv=torch.clip(adv,clip_min,clip_max)
            diff=adv-inputs
            noise=torch.clip(diff,-self.esp,self.esp)
            model.zero_grad()
            fin_adv=noise.clone()+inputs
        return fin_adv,noise