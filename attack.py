import torch
import torch.nn as nn
from torch.utils.data import dataloader
import torchvision
import torch.utils.data as data
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.models.resnet import resnet101, resnet152
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pretrainedmodels
import PSNR

from FGSM import FGSM
from ifgsm import IFGSM
from mifgsm import MIFGSM
from nifgsm import NIFGSM
from midfgsm import MID_FGSM
from nidfgsm import NIDFGSM
from attack_algorithem.di_ni_fgsm import DI_NIFGSM
from ti_ni_fgsm import TI_NIFGSM
from si_ni_fgsm import SI_NI_FGSM
from test_nifgsm import TEST_NIFGSM
from test6_nifgsm import TEST6_NIFGSM
from lin_img_nifgsm import LINEAR_NIFGSM


val_dir='./data/TinyImageNet'
transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,0.5,0.5),
        std=(0.5,0.5,0.5)
    ),
    # transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
    #                      std = [ 0.229, 0.224, 0.225 ]),
])
val=datasets.ImageFolder(val_dir,transform=transform)
val_loader=data.DataLoader(val,batch_size=1,shuffle=False,num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inc_v3= models.inception_v3(pretrained=True)
inc_v3.to(device)
inc_v3.eval()

res_152=models.resnet152(pretrained=True)
res_152.to(device)
res_152.eval()

res_18=models.resnet18(pretrained=True)
res_18.to(device)
res_18.eval()

# inc_v4=pretrainedmodels.models.inceptionv4()
# inc_v4.to(device)
# inc_v4.eval()

res_inc_v2=pretrainedmodels.models.inceptionresnetv2()
res_inc_v2.to(device)
res_inc_v2.eval()


def imshow(img):
    img = img/2+0.5     # unnormalize
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# data_it=iter(val_loader)
# imgs,labels=data_it.next()
# test_img=imgs[2]
# test_label=labels[2]
# or_output=inc_v3(test_img.unsqueeze(0))
# _,or_pre=torch.max(or_output,1)
# imshow(test_img)

MAX_EPS=16.0
MOMENTUM=1.0
STEP=10
KD=0.07

nid_attacker=NIDFGSM(esp=2*MAX_EPS/255,steps=STEP,u=MOMENTUM,kd=KD)
test_attacker=TEST_NIFGSM(esp=2*MAX_EPS/255,steps=STEP,u=MOMENTUM)
test6_attacker=TEST6_NIFGSM(esp=2*MAX_EPS/255,steps=STEP,u=MOMENTUM)
lin_attacker=LINEAR_NIFGSM(esp=2*MAX_EPS/255,steps=STEP,u=MOMENTUM)
di_ni_attacker=DI_NIFGSM(esp=2*MAX_EPS/255,steps=STEP,u=MOMENTUM)
ti_ni_attacker=TI_NIFGSM(esp=2*MAX_EPS/255,steps=STEP,u=MOMENTUM)
si_ni_attacker=SI_NI_FGSM(esp=2*MAX_EPS/255,steps=STEP,u=MOMENTUM)
mid_attacker=MID_FGSM(steps=STEP,kd=KD,esp=2*MAX_EPS/255,u=MOMENTUM)
mi_attacker=MIFGSM(steps=STEP,esp=2*MAX_EPS/255,u=MOMENTUM)
ni_attacker=NIFGSM(esp=2*MAX_EPS/255,steps=STEP,u=MOMENTUM)
i_attacker=IFGSM(steps=STEP,eps=2*MAX_EPS/255)
attacker=FGSM(esp=0.02)

# adv_img=mid_attacker.attack(inc_v3,test_img.unsqueeze(0),test_label.unsqueeze(0))
# adv_output=inc_v3(adv_img)
# _,adv_pre=torch.max(adv_output,1)
# imshow(adv_img.squeeze(dim=0))
# print(adv_pre==or_pre)


def delet_tensor(tensor,*index):
    b=tensor.cpu().detach().numpy()
    new_b=np.delete(b,index,axis=0)
    x=torch.Tensor(new_b).type_as(tensor)
    return x.to(device)

def save_img(tensor):
    pass

def attack_algorithem(*black_models,datasets=val_loader,attacker=mid_attacker,model=inc_v3,clip_min=-1,clip_max=1,save_adv=False,delet_f=True):
    #是否有黑盒攻击测试
    black_box_attack=False
    if len(black_models)>0:
        black_box_attack=True
    all=0
    white_seccess=0
    psnr=0.0
    black_seccess=[0.0 for i in range(len(black_models))]
    black_all=[0.0 for i in range(len(black_models))]
    r_black=[]
    for idx,(inputs,labels) in enumerate(tqdm(datasets)):
        inputs=inputs.to(device)
        labels=labels.to(device)
        #去除分类错误的图片数据
        or_output=model(inputs)
        _,o_pre=torch.max(or_output,1)
        if delet_f:
            delet_index=(o_pre!=labels).nonzero(as_tuple=True)[0]
            new_inputs=delet_tensor(inputs,delet_index.cpu().detach().numpy())
            new_labels=delet_tensor(labels,delet_index.cpu().detach().numpy()) 
        else:
            new_inputs=inputs
            new_labels=o_pre

        if int(new_labels.shape[0])==0:
            continue
        #生成对抗样本
        adv_imgs=attacker.attack(model,new_inputs,new_labels,clip_min=clip_min,clip_max=clip_max)
        psnr+=PSNR.psnr(adv_imgs.cpu().detach().numpy(),new_inputs.cpu().detach().numpy())
        adv_output=model(adv_imgs)
        _,adv_pre=torch.max(adv_output,1)
        all+=new_labels.shape[0]
        white_seccess+=(new_labels!=adv_pre).sum().item()
        # print(f'white_box seccess on inception_v3:{(white_seccess/all):3.2%},success num:{white_seccess},total:{all}')

        if save_adv:
            save_img(adv_imgs)

        if black_box_attack:
            for i,black_model in enumerate(black_models):
                black_output=black_model(new_inputs.to(device))
                _,black_pre=torch.max(black_output,1)
                if delet_f:
                    delet_index=(black_pre!=new_labels).nonzero(as_tuple=True)[0]
                    b_new_inputs=delet_tensor(new_inputs,delet_index.cpu().detach().numpy())
                    b_new_labels=delet_tensor(new_labels,delet_index.cpu().detach().numpy())
                    new_adv_imgs=delet_tensor(adv_imgs,delet_index.cpu().detach().numpy())
                else:
                    b_new_inputs=new_inputs
                    b_new_labels=new_labels
                    new_adv_imgs=adv_imgs

                if b_new_labels.shape[0]==0:
                    continue

                b_adv_output=black_model(new_adv_imgs.to(device))
                _,b_adv_pre=torch.max(b_adv_output,1)
                # print(new_labels!=b_adv_pre)
                black_all[i]+=b_new_labels.shape[0]
                black_seccess[i]+=(b_new_labels!=b_adv_pre).sum().item()
    print(f'white_box seccess on inception_v3:{(white_seccess/all):3.2%}')
    print(f'avg psnr: {(psnr/all):.2f}')
    if black_box_attack:
        for i in range(len(black_all)):
            print(f'black_box seccess on No.{i}model:{(black_seccess[i]/black_all[i]):3.2%}')
            r_black.append(black_seccess[i]/black_all[i])
    return psnr/all,white_seccess/all,r_black


if __name__=="__main__":
    attackers=[test_attacker]
    for attacker in attackers:
        _,_,_=attack_algorithem(model=inc_v3,datasets=val_loader,attacker=attacker)
        print('-'*80)
