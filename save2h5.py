import h5py
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm


from ifgsm import IFGSM
from nifgsm import NIFGSM
from nidfgsm import NIDFGSM
from ti_ni_fgsm import TI_NIFGSM
from si_ni_fgsm import SI_NI_FGSM
from test6_nifgsm import TEST6_NIFGSM

val_dir='./data/TinyImageNet'
transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize(
         mean=(0.5,0.5,0.5),
         std=(0.5,0.5,0.5)
     ),
])
val=datasets.ImageFolder(val_dir,transform=transform)
val_loader=data.DataLoader(val,batch_size=1,shuffle=False,num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

res_18=models.resnet18(pretrained=True)
res_18.to(device)
res_18.eval()

MAX_EPS=2.0
MOMENTUM=1.0
STEP=10
KD=0.07

i_attacker=IFGSM(steps=STEP,eps=MAX_EPS/255)
ni_attacker=NIFGSM(esp=MAX_EPS/255,steps=STEP,u=MOMENTUM)
nid_attacker=NIDFGSM(esp=MAX_EPS/255,steps=STEP,u=MOMENTUM,kd=KD)
ti_ni_attacker=TI_NIFGSM(esp=MAX_EPS/255,steps=STEP,u=MOMENTUM)
si_ni_attacker=SI_NI_FGSM(esp=MAX_EPS/255,steps=STEP,u=MOMENTUM)
test6_attacker=TEST6_NIFGSM(esp=MAX_EPS/255,steps=STEP,u=MOMENTUM)

def save_h5(dataset,dataset_name,group_name,out_path):
    with h5py.File(out_path,'a') as f:
        try:
            group=f.create_group(group_name)
            print(f'已创建组：{group_name}')
        except:
            print(f'组：{group_name}已存在！')
        try:
            group.create_dataset(dataset_name,data=dataset)
            print(f'已创建数据集：{dataset_name}')
        except:
            print(f'数据集：{dataset_name}已存在！')

def attack_save2h5(save_path,dataset_name,group_name,model=res_18,attacker=i_attacker,clip_min=0,clip_max=1):
    adv_imgs_list=[]
    ori_pic=[]
    label=[]
    for idx,(inputs,labels) in enumerate(tqdm(val_loader)):
        ori_pic.append(inputs.squeeze(0).cpu().detach().numpy())
        label.append(labels.squeeze(0).cpu().detach().numpy())
        inputs=inputs.to(device)
        labels=labels.to(device)
        adv_imgs=attacker.attack(model,inputs,labels,clip_min=clip_min,clip_max=clip_max)
        adv_imgs_list.append(adv_imgs.squeeze(0).cpu().detach().numpy())
    save_h5(adv_imgs_list,dataset_name,group_name,save_path)
    try:
        save_h5(ori_pic,'ori_pic','ori_pic_group',save_path)
    except:
        pass
    try:
        save_h5(label,'labels','labels',save_path)
    except:
        pass

if __name__=='__main__':
    attackers=[(test6_attacker,'TEST6'),(si_ni_attacker,'SINIFGSM')]
    for attacker in attackers:
        attack_save2h5(f'./resnet18_eps2_step10_{attacker[1]}.h5',f'adv_imgs','adv_imgs',attacker=attacker[0])