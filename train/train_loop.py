########################################################################################################
#                                                                                                      #
#                                            Train  -  Loop                                            #
#                                                                                                      #
########################################################################################################


import torch
from torch import nn, optim
# from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime, timedelta
import logging  # log files package
import yaml

from datasets import Modalities, ModalitiesSubset, classes
from datasets.transformations import *
from datasets.experiments import get_experiment_modalities_params, experiments_info, get_all_modalities
from model import PlantFeatureExtractor as FeatureExtractor  # this must be changed - creates confusion!!
from utils import get_checkpoint_name, get_used_modalities, get_levels_kernel, get_training_name
from train.parameters import *  # importing all parameters

# imports for plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""  LOGGING SETUP   """
logging.basicConfig(filename='Training2.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


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
    test_loader = DataLoader(test_config.test_set, batch_size=test_config.batch_size, num_workers=0, pin_memory=True,
                             shuffle=True)

    '''
    model.train() --> the model knows it has to learn the layers
    model.eval()  --> indicates that nothing new is to be learnt and the model is used for testing.
    '''

    # Because this is the test part - Model Eval is used
    test_config.feat_ext.eval()
    test_config.label_cls.eval()
    test_config.plant_cls.eval()

    tot_correct = 0.  # what is correct
    tot_label_loss = 0.  # label loss
    with torch.no_grad():  # test stage - no gradients are calculated
        for batch in test_loader:  # batch calculations are made
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            labels = batch['label'].to(test_config.device)

            x = batch.copy()

            del x['label']
            del x['plant']

            for key in x:
                x[key] = x[key].to(test_config.device)

            features: torch.Tensor = test_config.feat_ext(
                **x)  # x forward through plant feature extractor - applying the feature extractor
            label_out = test_config.label_cls(features)  # here test dataset labels are infered from the net
            label_loss = test_config.criterion(label_out,
                                               labels)  # the infered labels and original labels are sent to the loss criterion

            # 'there must be a better way! ' - check scikit-learn
            equality = (labels.data == label_out.max(dim=1)[1])  # finds out which labels are equal
            tot_correct += equality.float().sum().item()  # sums up batch correct inferences
            tot_label_loss += label_loss.item()  # sums up batch label loss criterions to determine general loss

    accuracy = tot_correct / len(test_config.test_set)  # Total Test Accuracy
    loss = tot_label_loss / len(test_config.test_set)  # Total Test Loss calculated
    print(f"Model Test: \t Test label accuracy %8.3f   Test label loss  %8.3f" % ((accuracy), (loss)))
    logging.info(f"\t Test label accuracy %8.3f  Test label loss  %8.3f" % ((accuracy), (loss)))

    # if test_config.use_checkpoints and loss < test_config.best_loss + test_config.loss_delta:
    #    test_config.best_loss = min(loss, test_config.best_loss)

 #   if test_config.use_checkpoints and loss < test_config.best_loss - test_config.loss_delta:
 #       test_config.best_loss = loss
#
  #      print(f'\t\tsaving model with new best loss {loss}')
 #       logging.info(f'\t\tsaving model with new best loss {loss}')
 #       torch.save({
 #           'feat_ext_state_dict': test_config.feat_ext.state_dict(),
 #           'label_cls_state_dict': test_config.label_cls.state_dict(),
 #           'plant_cls_state_dict': test_config.plant_cls.state_dict(),
 #           'loss': loss,
#            'accuracy': accuracy
#        }, f'checkpoints/{test_config.checkpoint_name}')
#        test_config.epochs_without_improvement = 0
#    elif test_config.return_epochs > 0:
#        test_config.epochs_without_improvement += 1

  #      if test_config.epochs_without_improvement == test_config.return_epochs:  # check the rational of this statement
  #          restore_checkpoint(test_config)
  #          test_config.epochs_without_improvement = 0

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
    TestLosses = []

    for epoch in range(test_config.epochs):  # run over a defined number of epochs
        epoch_start_time = datetime.today()
        print(f"epoch {epoch + 1}:", epoch_start_time)

        # Setting the neural networks to train mode
        test_config.feat_ext.train()
        test_config.label_cls.train()
        test_config.plant_cls.train()


        tot_label_loss = 0.
        tot_plant_loss = 0.
        tot_accuracy = 0.

        # in the following line train invokes the iterator over the train loader
        # this generates a single batch for every call
        # if train_loader is empty it receives nothing. this should be checked here with an assert
        # the call is to modalities __getitem__(self, idx) method

        # print('going to batch enumerate test_config.train_loader')
        for i, batch in enumerate(test_config.train_loader,
                                  1):  # the number 1 signifies that the itterable starts from 1
            # print(f'batch {i}', datetime.now())
            labels = batch['label'].to(test_config.device)
            # print('labels done ' )
            plants = batch['plant'].to(test_config.device)

            x = batch.copy()
            # print(' checking x none')
            if x == None:
                print(' x is None in line 174 x = batch.copy() in train loop  ')
            # else:
            #     print(' x is not none, x dictionary length is ', len(x))

            # Coppied everything to x and now earasing the label and plant
            del x['label']
            del x['plant']

            for key in x:
                # print('key =', key, end='  ')
                # print('x[key] image tensor to cuda device')
                x[key] = x[key].to(test_config.device)

            # zeroing all gradients so that they are recalculated per batch
            test_config.label_opt.zero_grad()
            test_config.plant_opt.zero_grad()
            test_config.ext_opt.zero_grad()

            # print('**x in features: torch.Tensor = test_config.feat_ext(**x) is:\n')
            # print(yaml.dump(x))
            # print('\n\n', "===============================================================\n\n")
            features: torch.Tensor = test_config.feat_ext(**x)  # applying the feature extractor for the batch
            features_plants = features.clone()
            '''tensor.clone()creates a copy of tensor that imitates the original tensor's requires_grad field.'''

            # read domain adaptation as refference for the following part
            # the following makes the gradients of the plant feature extractor get multiplied by lambda with a minus sign
            # setting this to negative makes plant optimization and label optimization work in opposing directions
            features_plants.register_hook(lambda grad: -test_config.domain_adapt_lr * grad)

            # label_out = forward through
            label_out = test_config.label_cls(features)  # training dataset labels are infered from the net
            label_loss = test_config.criterion(label_out,labels)  # infered labels and original labels are sent to the loss criterion

            plant_out = test_config.plant_cls(
                features_plants)  # training dataset plant classification is infered from the net
            plant_loss = test_config.criterion(plant_out,
                                               plants)  # infered plant classification and original plant names are sent to the loss criterion
            (label_loss + plant_loss).backward()  # backward propagation for both label classification and plant classification (opposit directions)

            equality = (labels.data == label_out.max(dim=1)[1])  # finds out which labels are equal
            tot_accuracy += equality.float().mean()  # sums up batch label correct inferences

            # Optimizers implement a step() method, that updates the parameters.
            test_config.label_opt.step()  # updating label parameters
            test_config.ext_opt.step()
            test_config.plant_opt.step()

            tot_label_loss += label_loss.item()  # batch label loss sum
            tot_plant_loss += plant_loss.item()  # batch plant loss sum

            num_print = 24
            if i % num_print == 0 or i * test_config.batch_size == len(test_config.train_set):
                num_since_last = num_print if i % num_print == 0 else i % num_print
                print(f"\tbatch {i}. label loss: %8.3f plant loss: %8.3f accuracy: %8.3f" %
                      ((tot_label_loss / num_since_last), (tot_plant_loss / num_since_last),
                       (tot_accuracy / num_since_last)))
            batch_end_time = datetime.today()
            ## End of Batch

        f = open(f"results_{experiment}.txt", "a+")
        trainSize = len(test_config.train_set)
        a, b, c = tot_label_loss / trainSize, tot_plant_loss / trainSize, tot_accuracy / trainSize
        logging.info(f"\tepoch {epoch + 1}: label loss: %8.3f plant loss: %8.3f accuracy: %8.3f" % ((a), (b), (c)))
        f.write(f"\tepoch {epoch + 1}: label loss: %8.3f plant loss: %8.3f accuracy: %8.3f \n" % ((a), (b), (c)))
        f.close()
        print(f"\tepoch {epoch + 1}: {i}. label loss: %8.3f plant loss: %8.3f accuracy: %8.3f" % ((a), (b), (c)))

        Train_label_losses.append(tot_label_loss / trainSize)
        Train_plant_losses.append(tot_plant_loss / trainSize)
        Train_accuracy_prog.append(tot_accuracy / trainSize)  # meaning  = train accuracy progress

        epoch_end_time = datetime.today()
        print(f'Epoch Time [hh:mm:sec] = {epoch_end_time - epoch_start_time}')
        ## End of Epoch

        test_acc, test_loss = test_model(test_config)
        TestAccuracy.append(test_acc)
        TestLosses.append(test_loss)
    print('Train_label_losses   :', Train_label_losses)
    print('Train_plant_losses   :', Train_plant_losses)
    print('Train_accuracy_prog  :', Train_accuracy_prog)
    print('Test acc  = ', test_acc)
    print('Test Loss = ', test_loss)

    f = open(f"results_{experiment}.txt", "a+")
    f.write(f'Train_label_losses   :')
    f.write(" ".join(map(str, Train_label_losses)))
    f.write('Train_plant_losses   :')
    f.write(" ".join(map(str, Train_plant_losses)))
    f.write('Train_accuracy_prog  :')
    f.write(" ".join(map(str, Train_accuracy_prog)))

    f.close()

    train_name: str = get_training_name(experiment, excluded_modalities)
    fig = plt.figure(1)
    plt.plot(Train_label_losses, 'or')
    fig.savefig(f'/home/pnn/PNN-AI/PNN-AI/train_results/{train_name}_label_losses2{datetime.date()}.png')

    fig = plt.figure(2)
    plt.plot(TestAccuracy, 'ob')
    fig.savefig(f'/home/pnn/PNN-AI/PNN-AI/train_results/{train_name}_test_accuracy2{datetime.date()}.png')


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
    print(' ----------------- inside main ------------------------------    ')
    checkpoint_name = get_checkpoint_name(experiment, excluded_modalities)

    # creating a results file, exactly one per run
    f = open(f"results_{experiment}.txt", "w+")
    # datetime of results file
    now = datetime.now()
    f.write('training start time :    ')
    f.write(str(now))
    f.write('\n')
    # Read all parameters from parameters file and document in the logging 'training2.log' file
    fread = open("parameters.py", "r")
    parameters = fread.read()
    fread.close()
    f.write(parameters)
    f.write('--------------------------------------------------------------------------------------------\n')
    f.close()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Cleaning GPU Cache memory
    #if torch.cuda.is_available():
    #    torch.cuda.empty_cache()
    #    # defining which GPU will run the process by cdevice from parameters file:
    #    device = torch.device(cdevice)
    #else:
    #    device = 'cpu'

    curr_experiment = experiments_info[experiment]

    modalities = get_experiment_modalities_params(curr_experiment, lwir_skip, lwir_max_len, vir_max_len, color_max_len)
    used_modalities = get_used_modalities(modalities, excluded_modalities)

    end_date = curr_experiment.end_date
    if num_days is not None:
        end_date = curr_experiment.start_date + timedelta(days=num_days - 1)

    # print(' used_modalities    =  ',used_modalities)
    print(' ****  dataset preparations *****')

    dataset = Modalities(experiment_path, experiment, split_cycle=split_cycle,
                         start_date=curr_experiment.start_date,
                         end_date=end_date,
                         **used_modalities)

    print(' **** Finished dataset preparations *****')
    print('dataset[1] printed using the __get_item__ method ')
    # print(dataset[1])
    for key in dataset[1]:
        if torch.is_tensor(dataset[1][key]):
            print(key, dataset[1][key].size())
        else:
            print(key, dataset[1][key])

    print('--------------------------------------------')

    train_set, test_set = ModalitiesSubset.random_split(dataset, train_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
    print('------------------------------------------------------------------------------------')
    print('len train_loader = ', len(train_loader))
    print('------------------------------------------------------------------------------------')
    
    print('used_modalities.keys()  = ', used_modalities.keys())

    # creating feat_extractor_params
    feat_extractor_params = dict()
    for mod in used_modalities.keys():
        num_levels, kernel_size = get_levels_kernel(dataset.modalities[mod].max_len)
        print('For mod =  ', mod, ' max len =', dataset.modalities[mod].max_len, '   num_levels  =  ', num_levels,
              '  kernel_size  =  ', kernel_size)
        feat_extractor_params[mod] = {
            'num_levels': num_levels,
            'kernel_size': kernel_size
        }
    print()
    print('feat_extractor_params dictionary :')
    print(yaml.dump(feat_extractor_params))

    feat_ext = FeatureExtractor(**feat_extractor_params).to(device)
    label_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, len(classes[experiment]))).to(device)
    plant_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, train_set.num_plants)).to(device)

    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

    label_opt = optim.SGD(label_cls.parameters(), lr=label_lr, momentum=0.7, weight_decay=1e-3)
    plant_opt = optim.SGD(plant_cls.parameters(), lr=plant_lr, momentum=0.7, weight_decay=1e-3)
    ext_opt = optim.SGD(feat_ext.parameters(), lr=extractor_lr, momentum=0.7, weight_decay=1e-3)

    best_loss = float('inf')

    test_config = TestConfig(use_checkpoints, checkpoint_name, epochs, batch_size, domain_adapt_lr, device,
                             dataset, train_set, test_set, train_loader, feat_ext, label_cls, plant_cls, criterion,
                             label_opt, plant_opt, ext_opt, best_loss, loss_delta, return_epochs)

    if load_checkpoint:
        restore_checkpoint(test_config)

    train_loop(test_config)

    if use_checkpoints:
        restore_checkpoint(test_config)


if __name__ == '__main__':
    main()
