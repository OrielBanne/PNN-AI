########################################################################################################
#                                                                                                      #
#                           Inception Subset                                                           #
#                                                                                                      #
########################################################################################################

# TORCH
import torch
from torch.utils.data import Dataset, TensorDataset

# Python Packages
import numpy as np
# import glob
from datetime import datetime
from typing import List, Type, Tuple, Iterator
from collections import defaultdict  # Nested dictionaries in python

# SKLearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# PNN SPECIFIC
from datasets.labels import labels
from train import parameters


class Modalities(Dataset):
    """
    A dataset class that lets the user decides which modalities to use.
    """

    def __init__(self, root_dir: str,
                 exp_name: str,
                 start_date,
                 end_date,
                 **plant_files_dict):
        """
        :param root_dir: path to the experiment directory
        :param exp_name: the experiment we want to use
        :param mods: modalities to be in the dataset, as dictionaries of initialization arguments

        """
        self.experiment = exp_name
        self.num_plants = len(labels[exp_name])
        self.modalities = defaultdict(dict)
        self.plant_files_dict = plant_files_dict
        self.num_plants = len(labels[self.experiment])
        # todo: fix used plants to be more generic
        self.used_plants = [plant for plant in range(self.num_plants)]
        self.root_dir = root_dir
        self.mods = parameters.used_modalities
        self.files = []
        self.start_date = start_date.date()
        self.end_date = end_date.date()

    def __len__(self):  # the number of samples in the dataset
        return self.num_plants

    def __get_file_date(self, file, plant, mod):
        file_format = f"{self.root_dir}/{self.experiment}_{mod}_{plant}%Y-%m-%d_%H:%M:%S"
        return datetime.strptime(file, file_format).date()

    # TODO: this one did not work because getitem gets only one parameter (plant)
    # @staticmethod
    # def __plant_inception__(self, plant, mod):
    #     tensors = []
    #     data_list = self.plant_files_dict[mod][plant]
    #     if len(data_list) > 0:
    #         plant_name = f'plant_{plant}_'
    #         print(plant_name, mod)
    #         for file in data_list:
    #             date_is = self.__get_file_date__(file, plant_name)  # file date
    #             if self.start_date <= date_is <= self.end_date:
    #                 inception_embedding = torch.load(file)
    #                 tensors.append(inception_embedding)
    #     plant_inception = torch.stack([tensor for tensor in tensors])
    #
    #     return plant_inception

    def __plant_inception_one_mod(self, plant, mod):
        tensors = []
        data_list = self.plant_files_dict[mod][plant]
        # if len(data_list) > 0:
        if data_list is not None:
            plant_name = f'plant_{plant}_'
            for file in data_list:
                date_is = self.__get_file_date(file, plant_name, mod)  # file date
                if self.start_date <= date_is <= self.end_date:
                    inception_embedding = torch.load(file)
                    tensors.append(inception_embedding)
            plant_inception_mod = torch.stack([tensor for tensor in tensors])
            return plant_inception_mod
        else:
            print('at least one of the mods is empty, check experiment mods!!!')
            return None

    def __plant_inception_all_mods(self, plant):
        plant_inception_all_mods = {}
        for mod in parameters.used_modalities:
            plant_inception_all_mods[mod] = self.__plant_inception_one_mod(plant, mod)
        return plant_inception_all_mods

    def __getitem__(self, idx):
        plant = idx % self.num_plants
        sample = dict(
            inception=self.__plant_inception_all_mods(plant),
            label=labels[self.experiment][plant],
            plant=plant)

        return sample


class InceptionSubset(Dataset):
    def __init__(self, used_plants, experiment, root_dir, used_modalities, plants: List[int]):
        self.labels = labels
        self.experiment = experiment
        self.root_dir = root_dir
        self.modalities = used_modalities
        self.data = used_plants
        self.plants = plants
        self.num_plants = len(plants)

        print('self.plants  = ', self.plants)

    def __len__(self):   # needs to be corrected to the number of items per plant for each modality
        return self.num_plants

    def __getitem__(self, idx):
        plant = self.plants[idx % self.num_plants]
        cycle = idx // self.num_plants
        data = self.data[self.data.num_plants * cycle + plant]
        data['plant'] = idx % self.num_plants
        return data

    @staticmethod
    def random_split(modalities: Modalities):
        indices = np.arange(modalities.num_plants)
        plant_labels = np.asarray(labels[modalities.experiment])
        #  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        train_indices, test_indices = train_test_split(indices, train_size=parameters.train_ratio,
                                                       stratify=plant_labels)
        return InceptionSubset(modalities, train_indices, parameters.experiment, parameters.experiment_path,
                               parameters.used_modalities), InceptionSubset(modalities, test_indices,
                                                                            parameters.experiment,
                                                                            parameters.experiment_path,
                                                                            parameters.used_modalities)

    @classmethod  # A generator
    def cross_validation(cls: Type['InceptionSubset'], used_plants: Modalities) -> \
            Iterator[Tuple['InceptionSubset', 'InceptionSubset']]:
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
        n_plants = parameters.n_plants
        n_classes = parameters.n_classes
        n_splits = parameters.n_splits
        X = tuple(i for i in range(n_plants))  # Plants by order
        y = np.array(labels[parameters.experiment])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        skf.get_n_splits(X, y)
        for train, test in skf.split(X, y):
            train_ds = cls(used_plants, parameters.experiment, parameters.experiment_path,
                           parameters.used_modalities, train)
            test_ds = cls(used_plants, parameters.experiment, parameters.experiment_path,
                          parameters.used_modalities, test)
            yield train_ds, test_ds

    @staticmethod
    def leave_one_out(modalities: Modalities, plant_idx: int):
        rest_idx = list(range(modalities.num_plants))
        del rest_idx[plant_idx]
        one_out = InceptionSubset(modalities, [plant_idx], parameters.experiment, parameters.experiment_path,
                                  parameters.used_modalities)
        rest = InceptionSubset(modalities, rest_idx, parameters.experiment, parameters.experiment_path,
                               parameters.used_modalities)
        return one_out, rest
