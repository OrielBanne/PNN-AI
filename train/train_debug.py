import torch
from torch import nn, optim
from torch.utils import data
import torch.nn.functional as F
#import argparse
import datetime
from datetime import datetime, timedelta
import logging #log files package

from datasets import Modalities, ModalitiesSubset, classes
from datasets.transformations import *
from datasets.experiments import get_experiment_modalities_params, experiments_info, get_all_modalities
from model import PlantFeatureExtractor as FeatureExtractor
from .utils_alt import get_checkpoint_name, get_used_modalities, add_experiment_dataset_arguments, get_levels_kernel

"""  LOGGING SETUP   """
logging.basicConfig(filename = 'Training.log', level = logging.INFO, format = '%(asctime)s %(levelname)s:%(message)s')
print('ok1')

'''
TEST CONFIG SHOULD BE SPLIT TO TEST CONFIG AND TRAIN CONFIG? THINK
'''

# define test config
class TestConfig:
    def __init__(self, use_checkpoints, checkpoint_name, epochs, batch_size, domain_adapt_lr, device, dataset,
                 train_set, test_set, train_loader, feat_ext, label_cls, plant_cls, criterion, label_opt, plant_opt,
                 ext_opt, best_loss, loss_delta, return_epochs):
        self.use_checkpoints = use_checkpoints
        self.checkpoint_name = checkpoint_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.domain_adapt_lr = domain_adapt_lr
        self.device = device
        self.dataset = dataset
        self.train_set = train_set
        self.test_set = test_set
        self.train_loader = train_loader
        self.epochs = epochs
        self.feat_ext = feat_ext
        self.label_cls = label_cls
        self.plant_cls = plant_cls
        self.criterion = criterion
        self.label_opt = label_opt
        self.plant_opt = plant_opt
        self.ext_opt = ext_opt
        self.best_loss = best_loss
        self.loss_delta = loss_delta
        self.return_epochs = return_epochs
        self.epochs_without_improvement = 0
        
        
    # Create an itterabl to revuew the class
    def __iter__(self):
      #return  tadtdatdatda
      pass

def train_loop(test_config: TestConfig):
    loop_start_time = datetime.today()
    print('loop_start_time                : ',loop_start_time)
    print('datetime.now()                 : ',datetime.now())
    print(' list test_config.train_set ' , test_config.train_set)
    logging.info(' New Train Loop Start:')
    print(' **** Running        train loop                          *****  ', datetime.now())
    use_checkpoints = False
    print(' test_config.epochs   ', test_config.epochs , datetime.now())
    for epoch in range(test_config.epochs):
        print(f"epoch {epoch + 1}:", datetime.now())

        test_config.feat_ext.train()
        print('test_config.feat_ext.train()  done', datetime.now())
        
        test_config.label_cls.train()
        print('test_config.label_cls.train()  done', datetime.now())
        
        test_config.plant_cls.train()
        print('test_config.plant_cls.train()  done', datetime.now())

        tot_label_loss = 0.
        tot_plant_loss = 0.
        tot_accuracy = 0.
        
        ################################################################################
        ################################################################################
        print(' test_config.train_loader   ', test_config.train_loader , datetime.now())
        for i, batch in enumerate(test_config.train_loader, 1):
            print(f'batch {i}', datetime.now())
            labels = batch['label'].to(test_config.device)
            print(f'labels of batch {i} done', datetime.now())
            plants = batch['plant'].to(test_config.device)
            print(f'plants of batch {i} done', datetime.now())
            x = batch.copy()
            print(' x = batch copy done - should now print sooo far soooo good')
    
    
    
    print('So Far, So Good')


def restore_checkpoint(test_config: TestConfig):
    checkpoint = torch.load(f'checkpoints/{test_config.checkpoint_name}')
    test_config.feat_ext.load_state_dict(checkpoint['feat_ext_state_dict'])
    test_config.label_cls.load_state_dict(checkpoint['label_cls_state_dict'])
    test_config.plant_cls.load_state_dict(checkpoint['plant_cls_state_dict'])
    test_config.best_loss = checkpoint['loss']

    print(f"Restoring model to one with loss - {test_config.best_loss}")

    test_config.feat_ext = test_config.feat_ext.to(test_config.device)
    test_config.label_cls = test_config.label_cls.to(test_config.device)
    test_config.plant_cls = test_config.plant_cls.to(test_config.device)
    
    
def main(experiment, experiment_path, use_checkpoints, load_checkpoint, excluded_modalities, epochs, domain_adapt_lr, label_lr, plant_lr, extractor_lr, train_ratio, batch_size, loss_delta, return_epochs, modalities):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Runnig without argspars - text file will handle all the inputs
    checkpoint_name = get_checkpoint_name(experiment, excluded_modalities)
    curr_experiment = experiments_info[experiment]

    
    # change the name!! using modalities for the tupple and the following dictionary
    modalities = get_experiment_modalities_params(curr_experiment, lwir_skip, lwir_max_len, vir_max_len, color_max_len)
    #print('modalities : ',modalities)
    print('excluded_modalities : ',excluded_modalities)
    print('the parameters sent for get_experiments_modalities function: ')
    print('curr_experiment :', curr_experiment)
    print('lwir_skip  :', lwir_skip)
    print('lwir_max_len :',lwir_max_len)
    print('vir_max_len  :',vir_max_len)
    print('color_max_len:',color_max_len)
    print(' ***    THATS IT                                         *****  ', datetime.now())
    used_modalities = get_used_modalities(modalities, excluded_modalities)


    if experiment_path is None:
        experiment_path = experiment
    else:
        experiment_path = experiment_path
    end_date = curr_experiment.end_date
    if num_days is not None:
        end_date = curr_experiment.start_date + timedelta(days=num_days-1)


    dataset = Modalities(experiment_path, experiment, split_cycle=split_cycle, start_date=curr_experiment.start_date, end_date=end_date, k_mods = used_modalities)
    
    print(' **** Finished dataset preparations                      *****  ',datetime.now())
    
    
    train_set, test_set = ModalitiesSubset.random_split(dataset, train_ratio)
    print(' **** Finished train_set, test_set preparations          *****  ', datetime.now())
    
    train_loader = data.DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)
    print(' **** Finished train_loader preparations                 *****  ', datetime.now())
    print(' train_loader =', train_loader)
    print()
    print(' End of train_loader -----------------------------------------------------------------------')
    
    feat_extractor_params = dict()
    print(' **** Finished feat_extractor_params preparations        *****  ', datetime.now())
    
    
    for mod in used_modalities.keys():
        num_levels, kernel_size = get_levels_kernel(dataset.modalities[mod].max_len)
        feat_extractor_params[mod] = {
            'num_levels': num_levels,
            'kernel_size': kernel_size
        }
    print(' **** Finished for - loop  for mods                      *****  ', datetime.now())
        
        
    feat_ext = FeatureExtractor(**feat_extractor_params)
    print(' **** Finished FeatureExtractor creation                 *****  ', datetime.now())
    
    feat_ext = feat_ext.to(device)
    print(' **** Finished FeatureExtractor to device                *****  ', datetime.now())
    
    label_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, len(classes[experiment]))).to(device)
    print(' **** Finished label_cls preparations                    *****  ', datetime.now())
    
    plant_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, train_set.num_plants)).to(device)
    print(' **** Finished plant_cls preparations                    *****  ', datetime.now())

    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    print(' **** Finished defining cross entropy loss criterion     *****  ', datetime.now())
    
    

    label_opt = optim.SGD(label_cls.parameters(), lr=label_lr, weight_decay=1e-3)
    plant_opt = optim.SGD(plant_cls.parameters(), lr=plant_lr, weight_decay=1e-3)
    ext_opt = optim.SGD(feat_ext.parameters(), lr=extractor_lr, weight_decay=1e-3)
    print(' **** Finished weight decay preparations                 *****  ', datetime.now())


    best_loss = float('inf')

    test_config = TestConfig(use_checkpoints, checkpoint_name, epochs, batch_size, domain_adapt_lr, device,
                             dataset, train_set, test_set, train_loader, feat_ext, label_cls, plant_cls, criterion,
                             label_opt, plant_opt, ext_opt, best_loss, loss_delta, return_epochs)
    print(' **** Finished test config preparations                  *****  ', datetime.now())
    
    if load_checkpoint:# restore a checkpoint. start training from here
        restore_checkpoint(test_config)
        print(' **** Finished restore_checkpoint                       *****  ', datetime.now())
    
    print(' **** ready for train loop                               *****  ', datetime.now())
    print(' test_config : ', test_config)
    print(' now : train_loop(test_config)  ')
    train_loop(test_config)  # test_config

    if use_checkpoints:   #if during training it is found that the train did not bring us to the best checkpoint  - now will load it
        restore_checkpoint(test_config)


if __name__ == '__main__':
    mods = get_all_modalities()
    
    
    # PLACE THE FOLLOWING GLOBAL PARAMETERS IN A FILE
    
# Default Parameters
    use_checkpoints=False #  checkpoints disable for training
    load_checkpoint=False #  load previous training checkpoint 
    experiment = 'Exp0'   #  Experiment name
    load_features = True
    num_days = None
    experiment_path = experiment # Or None if theres any issue with experiment
    split_cycle = 7
    start_date = None
    PCA = 0                ######################################################################
    lwir_max_len = None
    lwir_skip = 10
    vir_max_len = None
    color_max_len = None
    num_days = None
    num_clusters = 0      ### is this needed here or only for clustering??
    load_checkpoint = False
    use_checkpoints = True

# training hyper-parameters
    excluded_modalities=[] # All of the modalities that you don't want to use
    train_ratio = 5/6     #  dataset train/test ratio
    batch_size=4          #  training batch size
    loss_delta=0.0        #  The minimum above the best loss that is allowed for the model to be  saved
    return_epochs=0       #  Epochs without improvement allowed (loss_delta included). Return to the best checkpoint otherwise. 0 to disable
    epochs=25             #  number of epochs during train
    label_lr=1e-2         #  phenotype classifier learning rate
    plant_lr=1e-2         #  plant classifier LR used for transfer learning
    extractor_lr=1e-2     #  feature extractor LR
    domain_adapt_lr=1e-2  #  domain adaptation learning rate

    
    main(experiment, experiment_path, use_checkpoints, load_checkpoint, excluded_modalities, epochs, domain_adapt_lr, label_lr, plant_lr, extractor_lr, train_ratio, batch_size, loss_delta, return_epochs, modalities = mods)