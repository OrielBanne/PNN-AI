########################################################################################################
#                                                                                                      #
#                                            Train  -  Loop                                            #
#                                                                                                      #
########################################################################################################


import torch
from torch import nn, optim
from torch.utils import data
import torch.nn.functional as F
from datetime import datetime, timedelta
import logging #log files package

from datasets import Modalities, ModalitiesSubset, classes
from datasets.transformations import *
from datasets.experiments import get_experiment_modalities_params, experiments_info, get_all_modalities
from model import PlantFeatureExtractor as FeatureExtractor    # this must be changed - creates confusion!!
from .utils import get_checkpoint_name, get_used_modalities, get_levels_kernel
from train.parameters import *  # importing all parameters

# imports for plotting 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""  LOGGING SETUP   """
logging.basicConfig(filename = 'TrainAdam.log', level = logging.INFO, format = '%(asctime)s :%(message)s')

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


def test_model(test_config: TestConfig):
    test_loader = data.DataLoader(test_config.test_set, batch_size=test_config.batch_size, num_workers=2, shuffle=True)

    test_config.feat_ext.eval()
    test_config.label_cls.eval()
    test_config.plant_cls.eval()

    tot_correct = 0.
    tot_label_loss = 0.
    with torch.no_grad():
        for batch in test_loader:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            labels = batch['label'].to(test_config.device)

            x = batch.copy()

            del x['label']
            del x['plant']

            for key in x:
                x[key] = x[key].to(test_config.device)

            features: torch.Tensor = test_config.feat_ext(**x)
            label_out = test_config.label_cls(features)
            label_loss = test_config.criterion(label_out, labels)

            equality = (labels.data == label_out.max(dim=1)[1])
            tot_correct += equality.float().sum().item()
            tot_label_loss += label_loss.item()

    accuracy = tot_correct / len(test_config.test_set)
    loss = tot_label_loss / len(test_config.test_set)
    print(f"Model Test: \t Test label accuracy %8.3f   Test label loss  %8.3f" % ((accuracy),(loss)))
    logging.info(f"\t Test label accuracy %8.3f  Test label loss  %8.3f" % ((accuracy),(loss)))

    if test_config.use_checkpoints and loss < test_config.best_loss + test_config.loss_delta:
        test_config.best_loss = min(loss, test_config.best_loss)

        print(f'\t\tsaving model with new best loss {loss}')
        logging.info(f'\t\tsaving model with new best loss {loss}')
        torch.save({
            'feat_ext_state_dict': test_config.feat_ext.state_dict(),
            'label_cls_state_dict': test_config.label_cls.state_dict(),
            'plant_cls_state_dict': test_config.plant_cls.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }, f'checkpoints/{test_config.checkpoint_name}')
        test_config.epochs_without_improvement = 0
    elif test_config.return_epochs > 0:
        test_config.epochs_without_improvement += 1
        if test_config.epochs_without_improvement == test_config.return_epochs:
            restore_checkpoint(test_config)
            test_config.epochs_without_improvement = 0

    return accuracy, loss


def train_loop(test_config: TestConfig):
    logging.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logging.info('\tNew Train Loop Start:')
    loop_start_time = datetime.today()
    print(' **** Running        train loop                          *****  ', datetime.now())
    # parameters for graph
    Train_label_losses = []
    Train_plant_losses = []
    Train_accuracy_prog = []
    TestAccuracy = []
    TestLosses   = []
    for epoch in range(test_config.epochs):
        print(f"epoch {epoch + 1}:", datetime.now())
        epoch_start_time = datetime.today()
        test_config.feat_ext.train()
        test_config.label_cls.train()
        test_config.plant_cls.train()

        tot_label_loss = 0.
        tot_plant_loss = 0.
        tot_accuracy = 0.
        
        for i, batch in enumerate(test_config.train_loader, 1):
            # print(f'batch {i}', datetime.now())
            labels = batch['label'].to(test_config.device)
            # print('labels done ' )
            plants = batch['plant'].to(test_config.device)

            x = batch.copy()


            # Copied up to here
            del x['label']
            del x['plant']

            for key in x:
                x[key] = x[key].to(test_config.device)

            test_config.label_opt.zero_grad()
            test_config.plant_opt.zero_grad()
            test_config.ext_opt.zero_grad()

            features: torch.Tensor = test_config.feat_ext(**x)
            features_plants = features.clone()
            #if i == 25:
            #  test_config.domain_adapt_lr = test_config.domain_adapt_lr/10.
            features_plants.register_hook(lambda grad: -test_config.domain_adapt_lr * grad)

            label_out = test_config.label_cls(features)
            label_loss = test_config.criterion(label_out, labels)

            plant_out = test_config.plant_cls(features_plants)
            plant_loss = test_config.criterion(plant_out, plants)
            (label_loss + plant_loss).backward()

            equality = (labels.data == label_out.max(dim=1)[1])
            tot_accuracy += equality.float().mean()

            test_config.label_opt.step()
            test_config.ext_opt.step()
            test_config.plant_opt.step()

            tot_label_loss += label_loss.item()
            tot_plant_loss += plant_loss.item()

            num_print = 24
            if i % num_print == 0 or i * test_config.batch_size == len(test_config.train_set):
                num_since_last = num_print if i % num_print == 0 else i % num_print
                print(f"\tbatch {i}. label loss: %8.3f plant loss: %8.3f accuracy: %8.3f" % 
                         ((tot_label_loss / num_since_last),(tot_plant_loss / num_since_last),(tot_accuracy   / num_since_last)))

            #logging.info(f"\tepoch {epoch + 1}: label loss: %8.3f plant loss: %8.3f accuracy: %8.3f" % ((tot_label_loss / len(test_config.train_set)),(tot_plant_loss / len(test_config.train_set)),(tot_accuracy / len(test_config.train_set))))
            batch_end_time = datetime.today()
            ## End of Batch
        trainSize = len(test_config.train_set)
        logging.info(f"\tepoch {epoch + 1}: {i}. label loss: %8.3f plant loss: %8.3f accuracy: %8.3f" % 
                         ((tot_label_loss / trainSize),(tot_plant_loss / trainSize),(tot_accuracy / trainSize)))
        print(f"\tepoch {epoch + 1}: {i}. label loss: %8.3f plant loss: %8.3f accuracy: %8.3f" % 
                         ((tot_label_loss / len(test_config.train_set)),(tot_plant_loss / len(test_config.train_set)),(tot_accuracy / len(test_config.train_set))))
        Train_label_losses.append(tot_label_loss/trainSize)
        Train_plant_losses.append(tot_plant_loss/trainSize)
        Train_accuracy_prog.append(tot_accuracy/trainSize) # meaning  = train accuracy progress
        tot_label_loss = 0.
        tot_plant_loss = 0.
        tot_accuracy = 0.
        epoch_end_time = datetime.today()
        print(f'Epoch Time [hh:mm:sec] = {epoch_end_time-epoch_start_time}')
        ## End of Epoch

        test_acc, test_loss = test_model(test_config)
        TestAccuracy.append(test_acc)
        TestLosses.append(test_loss)
    print('Train_label_losses   :',Train_label_losses)
    print('Train_plant_losses   :',Train_plant_losses)
    print('Train_accuracy_prog  :',Train_accuracy_prog)
    print('Test acc  = ',test_acc)
    print('Test Loss = ',test_loss)
    
    f = open("results.txt", "w+")
    f.write('Train_label_losses   :')
    f.write(" ".join(map(str,Train_label_losses)))
    f.write('Train_plant_losses   :')
    f.write(" ".join(map(str,Train_plant_losses)))
    f.write('Train_accuracy_prog  :')
    f.write(" ".join(map(str,Train_accuracy_prog.item())))
    
    f.close()
    
    fig = plt.figure(1)
    plt.plot(Train_label_losses,'or')
    fig.savefig('Train_label_losses_Adam.png')
    
    fig = plt.figure(2)
    plt.plot(TestAccuracy,'ob')
    fig.savefig('TestAccuracy_Chart_Adam.png')

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


def main():
    checkpoint_name = get_checkpoint_name(experiment, excluded_modalities)

    # creating a results file, exactly one per run
    f = open("res.txt", "w+")
    fread = open("train/parameters.py", "r")
    parameters = fread.read()
    fread.close()
    f.write(parameters)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    curr_experiment = experiments_info[experiment]

    modalities = get_experiment_modalities_params(curr_experiment, lwir_skip, lwir_max_len, vir_max_len, color_max_len)
    used_modalities = get_used_modalities(modalities, excluded_modalities)

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

    
    print('used_modalities.keys()  = ',used_modalities.keys())
    
  
    feat_extractor_params = dict()
    for mod in used_modalities.keys():
        num_levels, kernel_size = get_levels_kernel(dataset.modalities[mod].max_len)
        feat_extractor_params[mod] = {
            'num_levels': num_levels,
            'kernel_size': kernel_size
        }

    feat_ext = FeatureExtractor(**feat_extractor_params).to(device)
    label_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, len(classes[experiment]))).to(device)
    plant_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, train_set.num_plants)).to(device)

    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    
    # Changing the learning rates for Adam:
    label_lr=1e-4         #  phenotype classifier learning rate
    plant_lr=1e-4         #  plant classifier LR used for transfer learning
    extractor_lr=1e-4     #  feature extractor LR

    label_opt = optim.Adam(label_cls.parameters(), lr=label_lr,     betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
    plant_opt = optim.Adam(plant_cls.parameters(), lr=plant_lr,     betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
    ext_opt   = optim.Adam( feat_ext.parameters(), lr=extractor_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)

    best_loss = float('inf')

    test_config = TestConfig(use_checkpoints, checkpoint_name, epochs, batch_size, domain_adapt_lr, device,
                             dataset, train_set, test_set, train_loader, feat_ext, label_cls, plant_cls, criterion,
                             label_opt, plant_opt, ext_opt, best_loss, loss_delta, return_epochs)

    if load_checkpoint:
        restore_checkpoint(test_config)

    train_loop(test_config)

    if use_checkpoints:
        restore_checkpoint(test_config)

    f.close()

if __name__ == '__main__':
    mods = get_all_modalities()
    print(mods)
    main()
