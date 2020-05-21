########################################################################################################
#                                                                                                      #
#                                        DatasetCleaningApp                                            #
#                                                                                                      #
########################################################################################################


import torch
from torch import nn, optim
from torch.utils import data
import torch.nn.functional as F
from datetime import datetime, timedelta


from datasets import Modalities, ModalitiesSubset, classes
from datasets.transformations import *
from datasets.experiments import get_experiment_modalities_params, experiments_info, get_all_modalities
from model import PlantFeatureExtractor as FeatureExtractor    # this must be changed - creates confusion!!
from .utils import get_checkpoint_name, get_used_modalities, get_levels_kernel, get_training_name
from train.parameters import *  # importing all parameters

# imports for plotting 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def main():
    print('------------------------------------------------------------------------------------')
    print(' inside dataset cleaning  App  ')
    print('------------------------------------------------------------------------------------')
    checkpoint_name = get_checkpoint_name(experiment, excluded_modalities)
    
    # Read all parameters from parameters file and document in the logging 'training2.log' file
    fread = open("train/parameters.py", "r")
    parameters = fread.read()
    print('here is the parameters file as i have read it :')
    print('-------------------------------------------------------------------------------------')
    print(parameters) 
    print('-------------------------------------------------------------------------------------')
    print('end of parameters closing file ')
    fread.close()
    
    
    if torch.cuda.is_available():
        # Cleaning GPU Cache memory
        torch.cuda.empty_cache()
        # define active gpu by cdevice (in parameters file)
        device = torch.device(cdevice)
    else:
        device = 'cpu'

    curr_experiment = experiments_info[experiment]

    modalities = get_experiment_modalities_params(curr_experiment, lwir_skip, lwir_max_len, vir_max_len, color_max_len)
    print('-------------------------------------------------------------------------------------------------')
    print('Modalities     = ', modalities)
    used_modalities = get_used_modalities(modalities, excluded_modalities)
    print('-------------------------------------------------------------------------------------------------')
    print('Used Modalities = ', used_modalities)
    print('-------------------------------------------------------------------------------------------------')

    end_date = curr_experiment.end_date
    if num_days is not None:
        end_date = curr_experiment.start_date + timedelta(days=num_days-1)
        

    print()
    
    print(' ****  dataset preparations *****')

    dataset = Modalities(experiment_path,experiment,split_cycle=split_cycle,
                         start_date=curr_experiment.start_date,
                         end_date=end_date,
                         **used_modalities)

          
    print(' **** Finished dataset preparations *****')
    
    train_set, test_set = ModalitiesSubset.random_split(dataset, train_ratio)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)
    print('------------------------------------------------------------------------------------')
    print('len train_loader = ',len(train_loader))
    print('------------------------------------------------------------------------------------')
    
    print('used_modalities.keys()  = ',used_modalities.keys())
    
    print('feature extractor parameters to dict    ')
    feat_extractor_params = dict()
    for mod in used_modalities.keys():
        num_levels, kernel_size = get_levels_kernel(dataset.modalities[mod].max_len)
        feat_extractor_params[mod] = {
            'num_levels': num_levels,
            'kernel_size': kernel_size
        }
        print('Mod is :  ',mod, '\t TCN num_levels  = ',num_levels, '\t TCN kernel_size  = ',kernel_size)
        # these are the size and the depth of the TCN Kernel for this modality

    # feat_ext = FeatureExtractor(**feat_extractor_params).to(device)
    # label_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, len(classes[experiment]))).to(device)
    # plant_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, train_set.num_plants)).to(device)
    
    

if __name__ == '__main__':
    mods = get_all_modalities()
    print(' all modalilties are : ',mods)
    print('going to main ')
    main()
