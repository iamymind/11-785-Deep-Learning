import numpy as np # linear algebra
import torch
import pandas as pd
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch.utils.data.sampler as sampler

import utils

class WikiDataset(data.Dataset):
    """ wiki articles dataset:
    -  __getitem__
    - __len__
    """
    def __init__(self, X_data, X_transform = None):
        """
        X_data = array of articles
        X_transform - your augmentation function
        """
        self.X_data  = X_data
        self.X_transform = X_transform

    def __getitem__(self, index):
        """
        returns one Phoneme (1, 1, 40, #f
        """
            
        article_torch  = utils.to_tensor((X_data[index]))
       
        dict_ = {
                'article' : article_torch
                }
                 
        return dict_
def my_collate(batch):
    return batch
if __name__=='__main__':
   
    batch_size  = 8   
    train_data, val_data, vocabulary = (
                                        utils.to_tensor(np.concatenate(np.load('./dataset/wiki.train.npy'))),
                                        utils.to_tensor(np.concatenate(np.load('./dataset/wiki.valid.npy'))),
                                        np.load('./dataset/vocab.npy')
                                       )
    
    wiki_train_ds   =  WikiDataset(train_data)
    
    wiki_train_loader = data.DataLoader(
                                   wiki_train_ds, batch_size = batch_size,
                                   sampler = RandomSampler(wiki_train_ds),
                                   collate_fn  = my_collate
                                 ) 

    for batch_index, batch_dict in enumerate(val_loader):
        print(batch_dict)
        if batch_index ==0:break
