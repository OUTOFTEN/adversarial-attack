def Re_transforms(tensor,mean:list,std:list):
    #tensor.shape:(3,w.h)
    for idx,i in enumerate(std):
        tensor[:,idx,:,:]*=i
    for index,j in enumerate(mean):
        tensor[:,index,:,:]+=j
    return tensor
