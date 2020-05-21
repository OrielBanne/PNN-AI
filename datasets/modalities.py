########################################################################################################
#                                                                                                      #
#                                  MODALITIES                                                          #
#                                                                                                      #
########################################################################################################


from datetime import datetime
from torch.utils import data
from typing import Dict, List
from sklearn.model_selection import train_test_split
import numpy as np

from . import LWIR, VIR577nm, VIR692nm, VIR732nm, VIR970nm, VIRPolar, VIRPolarA, VIRNoFilter, Color
from .labels import labels
from train.parameters import *  # importing all parameters
#from datasets.experiments import get_all_modalities

print('importing modalities')

mod_map = {
    'lwir': LWIR,
    '577nm': VIR577nm,
    '692nm': VIR692nm,
    '732nm': VIR732nm,
    '970nm': VIR970nm,
    'polar': VIRPolar,
    'polar_a': VIRPolarA,
    'noFilter': VIRNoFilter,
    'color': Color
}

#################################################################################
##           MODALITIES

class Modalities(data.Dataset):
    """
    A dataset class that lets the user decides which modalities to use.
    """
##  Modalities(used_modalities, experiment_path, experiment, split_cycle, start_date=curr_experiment.start_date, end_date=end_date)
    def __init__(self, root_dir: str, exp_name: str, split_cycle=7,start_date=start_date, end_date=end_date,transform=None, **k_mods: Dict):
        """
        :param root_dir: path to the experiment directory
        :param exp_name: the experiment we want to use
        :param mods: modalities to be in the dataset, initialized with default arguments
        :param split_cycle: amount of days the data will be split by
        :param transform: optional transform to be applied on a sample
        :param k_mods: modalities to be in the dataset, as dictionaries of initialization arguments
        """
        
        
        #mods = get_all_modalities()
        # print('DEBUG #####')
  #      print('mods :', mods)
        #print('k_mods :', k_mods)
        print('__________________________________________________________________________________')
    #    if len(mods) + len(k_mods) == 0:
        if len(k_mods) == 0:
            #print('mods + k_mods length was zero')
            print('k_mods length was zero')
            mods = mod_map.keys()

        print('creating self.modalities')
        self.modalities = dict()

        # print('for mod in mods:')
        # for mod in mods:
        #    self.modalities[mod] = mod_map[mod](root_dir, exp_name,split_cycle,start_date, end_date)
        # print('for mod in k_mods:')
        
        #print('DeBug')
        #print('______________________________________________________')
        #print('root_dir    : ',root_dir)
        #print('exp name is : ',exp_name)
        #print('split_cycle is :',split_cycle)
        #print('start_date is :',start_date)
        #print('end_date   is :',end_date)
        #print('======================================================')
        for mod in k_mods:
            # print('mod is         :',mod)
            # print('k_mods is :', k_mods)
            self.modalities[mod] = mod_map[mod](root_dir=root_dir, exp_name=exp_name, split_cycle=split_cycle,start_date=start_date, end_date=end_date, **(k_mods[mod]))
         
        self.transform = transform
        self.exp_name = exp_name
        self.split_cycle = split_cycle
        self.num_plants = len(labels[exp_name])
        print('self.num_plants is    :',self.num_plants)

    def __len__(self):
        dataset = next(iter(self.modalities.values()))
        return len(dataset)

    def __getitem__(self, idx):
        sample = {
            mod: dataset[idx]['image'] for mod, dataset in self.modalities.items()
        }

        plant = idx % self.num_plants

        sample['label'] = labels[self.exp_name][plant]
        sample['plant'] = plant

        if self.transform:        # check this part not sure it works
            sample = self.transform(sample)

        return sample
#################################################################################
##           MODALITIES SUBSET

class ModalitiesSubset(data.Dataset):
    def __init__(self, modalities: Modalities, plants: List[int]):
        self.data = modalities
        self.split_cycle = modalities.split_cycle
        self.plants = plants
        self.num_plants = len(plants)

    def __len__(self):
        return self.num_plants * self.split_cycle

    def __getitem__(self, idx):
        plant = self.plants[idx % self.num_plants]
        cycle = idx // self.num_plants

        data = self.data[self.data.num_plants * cycle + plant]
        data['plant'] = idx % self.num_plants

        return data

    @staticmethod
    def random_split(modalities: Modalities, train_ratio: float):
        indices = np.arange(modalities.num_plants)
        plant_labels = np.asarray(labels[modalities.exp_name])

        train_indices, test_indices = train_test_split(indices, train_size=train_ratio, stratify=plant_labels)

        return ModalitiesSubset(modalities, train_indices), ModalitiesSubset(modalities, test_indices)

    @staticmethod
    def leave_one_out(modalities: Modalities, plant_idx: int):
        rest_idx = list(range(modalities.num_plants))
        del rest_idx[plant_idx]

        one_out = ModalitiesSubset(modalities, [plant_idx])
        rest = ModalitiesSubset(modalities, rest_idx)

        return one_out, rest
