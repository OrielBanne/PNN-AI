########################################################################################################
#                                                                                                      #
#                                  MODALITIES                                                          #
#                                                                                                      #
########################################################################################################
from torch.utils import data
from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from datasets import LWIR, VIR577nm, VIR692nm, VIR732nm, VIR970nm, VIRPolar, VIRPolarA, VIRNoFilter, \
    Color, Depth_day_night
from datasets.labels import labels
from train import parameters
import numpy as np


# TODO: Add any additional mods here:
mod_map = {
    'lwir': LWIR,
    '577nm': VIR577nm,
    '692nm': VIR692nm,
    '732nm': VIR732nm,
    '970nm': VIR970nm,
    'polar': VIRPolar,
    'polar_a': VIRPolarA,
    'noFilter': VIRNoFilter,
    'color': Color,
    'depth': Depth_day_night,
}


class Modalities(data.Dataset):
    """
    A dataset class that lets the user decides which modalities to use.
    """

    def __init__(self, root_dir: str,
                 exp_name: str,
                 split_cycle=parameters.split_cycle,
                 start_date=parameters.start_date,
                 end_date=parameters.end_date,
                 transform=None, **k_mods: Dict):
        """
        :param root_dir: path to the experiment directory
        :param exp_name: the experiment we want to use
        :param mods: modalities to be in the dataset, initialized with default arguments
        :param split_cycle: amount of days the data will be split by
        :param transform: optional transform to be applied on a sample
        :param k_mods: modalities to be in the dataset, as dictionaries of initialization arguments
        """
        if len(k_mods) == 0:
            print('k_mods length was zero')
            mods = mod_map.keys()  # TODO - why mods? never used
        print('creating self.modalities')
        self.modalities = dict()
        for mod in k_mods:
            print('@Modalities, mod: ', mod, end='')
            self.modalities[mod] = mod_map[mod](
                root_dir=root_dir,
                exp_name=parameters.experiment,
                split_cycle=split_cycle,
                start_date=start_date,
                end_date=end_date,
                **(k_mods[mod]))
            print('root_dir is : ', root_dir)
        self.transform = transform
        self.exp_name = exp_name
        self.split_cycle = split_cycle
        self.num_plants = len(labels[exp_name])
        print('self.num_plants is    :', self.num_plants)

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
        if self.transform:  # check this part not sure it works
            sample = self.transform(sample)
        return sample


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
    def random_split(modalities: Modalities):
        indices = np.arange(modalities.num_plants)
        plant_labels = np.asarray(labels[modalities.exp_name])
        #  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        train_indices, test_indices = train_test_split(indices, train_size=parameters.train_ratio,
                                                       stratify=plant_labels)
        return ModalitiesSubset(modalities, train_indices), ModalitiesSubset(modalities, test_indices)

    @staticmethod  # A generator
    def cross_validation(modalities: Modalities):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
        n_plants = len(labels[parameters.experiment])
        n_classes = len(np.unique(labels[parameters.experiment]))
        n_splits = n_plants // n_classes
        X = tuple(i for i in range(n_plants))  # Plants by order
        y = np.array(labels[parameters.experiment])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        skf.get_n_splits(X, y)
        for train, test in skf.split(X, y):
            yield ModalitiesSubset(modalities, train), ModalitiesSubset(modalities, test)

    @staticmethod
    def leave_one_out(modalities: Modalities, plant_idx: int):
        rest_idx = list(range(modalities.num_plants))
        del rest_idx[plant_idx]
        one_out = ModalitiesSubset(modalities, [plant_idx])
        rest = ModalitiesSubset(modalities, rest_idx)
        return one_out, rest



