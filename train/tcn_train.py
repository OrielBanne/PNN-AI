import glob
import math

from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import timedelta, datetime

from datasets.labels import classes, labels
from datasets.inception_subset import Modalities, InceptionSubset
from datasets.experiments import experiments_info
from datasets.transformations import *
from model.PlantFeatExtNew import PlantFeatureExtractor

from train.utils import get_checkpoint_name, get_training_name
from train import parameters  # importing all parameters/home/pnn/PNN/train

import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import classification_report
import numpy as np
import seaborn
from collections import defaultdict
import yaml


class TestConfig:
    def __init__(self, checkpoint_name, device, dataset, train_set, test_set, train_loader, feat_ext, label_cls,
                 plant_cls, criterion, label_opt, plant_opt, ext_opt, best_loss):
        self.use_checkpoints = parameters.use_checkpoints
        self.checkpoint_name = checkpoint_name
        self.epochs = parameters.epochs
        self.batch_size = parameters.batch_size
        self.domain_adapt_lr = parameters.domain_adapt_lr
        self.device = device
        self.dataset = dataset
        self.train_set = train_set
        self.test_set = test_set
        self.train_loader = train_loader
        self.epochs = parameters.epochs
        self.feat_ext = feat_ext
        self.label_cls = label_cls
        self.plant_cls = plant_cls
        self.criterion = criterion
        self.label_opt = label_opt
        self.plant_opt = plant_opt
        self.ext_opt = ext_opt
        self.best_loss = best_loss
        self.loss_delta = parameters.loss_delta
        self.return_epochs = parameters.return_epochs
        self.epochs_without_improvement = 0


def unique(list1):
    x = np.array(list1)
    return list(np.unique(x))


def confusion_matrix(test_config: TestConfig, run):
    test_loader = DataLoader(test_config.test_set, batch_size=test_config.batch_size, num_workers=0, shuffle=True)

    print('----------------------------------------------------')
    print('experiment = ', parameters.experiment, 'classes = ', classes[parameters.experiment])
    print('----------------------------------------------------')
    num_of_classes = len(classes[parameters.experiment])
    conf_matrix = np.zeros([num_of_classes, num_of_classes])

    test_config.feat_ext.eval()
    test_config.label_cls.eval()
    test_config.plant_cls.eval()

    with torch.no_grad():
        for batch in test_loader:
            label = batch['label'].to(test_config.device)

            x = batch['inception'].copy()

            for key in x:
                x[key] = x[key].to(test_config.device)

            features: torch.Tensor = test_config.feat_ext(**x)
            label_out = test_config.label_cls(features)
            #  label_out.shape  torch.Size([6, 6])
            print('label_out  = ', label_out)
            y = label.data.tolist()
            y_hat = label_out.max(dim=1)[1].tolist()

            for u in y:
                y_tot.append(u)
            for u in y_hat:
                y_hat_tot.append(u)

            for yi, y_hati in zip(y, y_hat):
                conf_matrix[yi, y_hati] += 1

    print('\n confusion matrix (not normalized)')
    print('\n', conf_matrix)

    '''
    Plot confusion matrix using heatmap.

        Args:
            data (list of list): List of lists with confusion matrix data.
            labels (list): Labels which will be plotted across x and y axis.
            output_filename (str): Path to output file.
    '''
    seaborn.set(color_codes=True)
    # plt.figure(4+run, figsize=(9, 6))
    plt.figure(figsize=(9, 6))

    plt.title(f"{parameters.experiment} Confusion Matrix")

    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(conf_matrix, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})

    ax.set_xticklabels(classes[parameters.experiment])
    ax.set_yticklabels(classes[parameters.experiment])

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    train_name: str = get_training_name(parameters.experiment, parameters.excluded_modalities)
    plt.savefig(f'{parameters.results_path}{train_name}_confusion_{datetime.date(datetime.now())}_run_{run}.png',
                bbox_inches='tight',
                dpi=300)

    plt.close()

    print(' for classification report -   y_tot  and y hat_tot are: ')
    print('y_tot = ', y_tot)
    print('y hat tot = ', y_hat_tot)
    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    print(classification_report(y_tot, y_hat_tot, target_names=list((classes[parameters.experiment])), digits=3))
    # https://towardsdatascience.com/multi-class-metrics-made-simple-the-kappa-score-aka-cohens-kappa-coefficient-bdea137af09c
    print('cohen_kappa_score  = ', cohen_kappa_score(y_tot, y_hat_tot))
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
    print('mathews correlation coefficient  = ', matthews_corrcoef(y_tot, y_hat_tot))
    class_report = 0

    return conf_matrix, class_report


def test_model(test_config: TestConfig):
    test_loader = DataLoader(test_config.test_set, batch_size=test_config.batch_size, num_workers=0, shuffle=True)

    test_config.feat_ext.eval()
    test_config.label_cls.eval()
    test_config.plant_cls.eval()

    tot_correct = 0.
    tot_label_loss = 0.
    with torch.no_grad():
        for batch in test_loader:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            lbls = batch['label'].to(test_config.device)

            x = batch['inception'].copy()

            # del x['label']
            # del x['plant']

            for key in x:
                x[key] = x[key].to(test_config.device)

            features: torch.Tensor = test_config.feat_ext(**x)
            label_out = test_config.label_cls(features)
            label_loss = test_config.criterion(label_out, lbls)

            equality = (lbls.data == label_out.max(dim=1)[1])
            tot_correct += equality.float().sum().item()
            tot_label_loss += label_loss.item()

    accuracy = tot_correct / len(test_config.test_set)  # Total Test Accuracy
    loss = tot_label_loss / len(test_config.test_set)  # Total Test Loss calculated
    print(f'\t       %8.3f  \t   %8.3f' % (loss, accuracy))

    if test_config.use_checkpoints and loss < test_config.best_loss + test_config.loss_delta:
        test_config.best_loss = min(loss, test_config.best_loss)

        print(f'\t\tsaving model with new best loss {loss}')
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


def train_loop(test_config: TestConfig,
               train_label_losses,
               train_plant_losses,
               train_accuracy_prog,
               test_accuracy,
               test_losses,
               run):
    print('  Train:  label loss:       plant loss:      accuracy:    Test: Label loss     accuracy  ')
    for epoch in range(test_config.epochs):
        print(f'epoch {epoch + 1}:', end=' ')

        test_config.feat_ext.train()
        test_config.label_cls.train()
        test_config.plant_cls.train()

        tot_label_loss = 0.
        tot_plant_loss = 0.
        tot_accuracy = 0.
        for i, batch in enumerate(test_config.train_loader, 1):
            labels: str = batch['label'].to(test_config.device)
            plants = batch['plant'].to(test_config.device)

            x = batch['inception'].copy()

            # del x['label']
            # del x['plant']

            for key in x:
                x[key] = x[key].to(test_config.device)

            test_config.label_opt.zero_grad()
            test_config.plant_opt.zero_grad()
            test_config.ext_opt.zero_grad()

            features: torch.Tensor = test_config.feat_ext(**x)
            features_plants = features.clone()
            features_plants.register_hook(lambda grad: -test_config.domain_adapt_lr * grad)

            label_out = test_config.label_cls(features)
            label_loss = test_config.criterion(label_out, labels)

            plant_out = test_config.plant_cls(features_plants)
            plant_loss = test_config.criterion(plant_out, plants)
            (label_loss + plant_loss).backward()

            equality = (labels.data == label_out.max(dim=1)[1])
            tot_accuracy += equality.float().sum().item()

            test_config.label_opt.step()
            test_config.ext_opt.step()
            test_config.plant_opt.step()

            tot_label_loss += label_loss.item()
            tot_plant_loss += plant_loss.item()

        train_size = len(test_config.train_set)
        a, b, c = tot_label_loss / train_size, tot_plant_loss / train_size, tot_accuracy / train_size
        print(f'   %8.3f        %8.3f         %8.3f' % (a, b, c), end=' ')

        train_label_losses.append(a)
        train_plant_losses.append(b)
        train_accuracy_prog.append(c)  # meaning  = train accuracy progress

        test_acc, test_loss = test_model(test_config)
        test_accuracy.append(test_acc)
        test_losses.append(test_loss)

    train_name: str = get_training_name(parameters.experiment, parameters.excluded_modalities)

    # plot accuracies
    # plt.figure(2+run)
    plt.figure()
    plt.plot(train_accuracy_prog, 'or', label='train accuracy')
    plt.plot(test_accuracy, 'ob', label='test accuracy')
    plt.legend(title='Parameters')
    plt.title(f'{parameters.experiment} Train and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f'{parameters.results_path}{train_name}_accuracy_{datetime.date(datetime.now())}_run_{run}.png')
    plt.close()

    # plot losses
    # plt.figure(3+run)
    plt.figure()
    plt.plot(train_label_losses, 'or', label='train label loss')
    plt.plot(train_plant_losses, 'ob', label='train plant loss')
    plt.plot(test_losses, 'og', label='test label loss')
    plt.title(f'{parameters.experiment} Train and Test Losses')
    plt.legend(title='Parameters')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{parameters.results_path}{train_name}_loss_{datetime.date(datetime.now())}_run_{run}.png')
    plt.close()

    conf_matrix, class_report = confusion_matrix(test_config, run)

    return train_label_losses, train_plant_losses, train_accuracy_prog, \
                test_accuracy, test_losses, conf_matrix, class_report


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
    checkpoint_name = get_checkpoint_name(parameters.experiment, parameters.excluded_modalities)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    used_modalities = parameters.used_modalities
    inception_embeddings_path = parameters.inception_embeddings_path

    start_date = experiments_info[parameters.experiment].start_date
    end_date = experiments_info[parameters.experiment].end_date
    if parameters.num_days is not None:
        end_date = experiments_info[parameters.experiment].start_date + timedelta(days=parameters.num_days - 1)

    # creating plants file dictionary # TODO: could be uploaded
    plant_files_dict = defaultdict(dict)
    mod_receptive_field = {}
    num_plants = len(labels[parameters.experiment])
    used_plants = [plant for plant in range(num_plants)]
    root_dir = '/home/pnn/inception_features_exp0'  # TODO change to generic root dir definition

    global y_tot, y_hat_tot
    y_tot = []
    y_hat_tot = []

    for mod in parameters.used_modalities:
        for plant in used_plants:
            plant_name = f'plant_{plant}_'  # example plant
            files = sorted(glob.glob(f'{root_dir}/*{mod}*{plant_name}*'))
            if plant is 0:
                mod_receptive_field[mod] = len(files)
            if files is not None:
                plant_files_dict[mod][plant] = files
    print('mod_receptive_field  ', mod_receptive_field)

    # print(' CREATING AND PRINTING THE DATASET   ')
    dataset = Modalities(inception_embeddings_path,
                         parameters.experiment,
                         start_date=start_date,
                         end_date=end_date,
                         **plant_files_dict)

    targets = [dataset[i]['label'] for i in range(len(dataset))]
    print('targets  = ', targets)
    class_numbers = unique(targets)  # TODO: a general way to enumerate classes of different experiments
    print('class_numbers   = ', class_numbers)

    t_label_losses = []
    t_plant_losses = []
    t_accuracy_prog = []
    val_accuracy = []
    val_losses = []
    # train_set, test_set = InceptionSubset.random_split(dataset)
    run = 0
    conf = []
    for train_set, test_set in InceptionSubset.cross_validation(dataset):

        train_labels = [train_set[i]['label'] for i in range(len(train_set))]
        test_labels = [test_set[i]['label'] for i in range(len(test_set))]

        print('train_labels : ', train_labels)
        print('train_set : ', train_set.plants)
        print('test_labels : ', test_labels)
        print('test_set : ', test_set.plants)
        print('classes {}  = '.format(parameters.experiment), classes[parameters.experiment])

        # TODO: if all counts for all keys in each group are equal - just print ok, no need for the charts,
        #  and keep running
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(5, 15))
        fig.suptitle('amounts of genotypes per set')
        keys1, counts1 = np.unique(targets, return_counts=True)
        ax1.bar(keys1, len(counts1))
        keys2, counts2 = np.unique(train_labels, return_counts=True)
        ax2.bar(keys2, len(counts2))
        keys3, counts3 = np.unique(test_labels, return_counts=True)
        ax3.bar(keys3, len(counts3))
        ax1.set_title('dataset')
        ax2.set_title('train set')
        ax3.set_title('test set')
        ax1.set_ylabel('number of elements')
        ax2.set_ylabel('number of elements')
        ax3.set_ylabel('number of elements')
        ax1.set_xlabel('classes')
        ax2.set_xlabel('classes')
        ax3.set_xlabel('classes')
        plt.draw()

        train_name: str = get_training_name(parameters.experiment, parameters.excluded_modalities)
        plt.savefig(f'{parameters.results_path}{train_name}_balanced_{datetime.date(datetime.now())}_run_{run}.png',
                    dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

        train_loader = DataLoader(train_set, batch_size=parameters.batch_size, num_workers=0, shuffle=True)

        print(len(train_loader))

        # print(yaml.dump(dataset))
        feat_extractor_params = dict()
        kernel_size = parameters.tcn_kernel_size
        # TODO - pad receptive field start with zeroes till start of data
        for mod in used_modalities:
            if mod_receptive_field[mod] > 0:
                num_levels = math.log2(mod_receptive_field[mod] / kernel_size) / math.log2(kernel_size)
                int_num_levels = int(num_levels)
                round_num_levels = round(num_levels)
                if round_num_levels == int_num_levels:
                    num_levels = int_num_levels + 1
                else:
                    num_levels = round_num_levels

                feat_extractor_params[mod] = {
                    'num_levels': int(num_levels),
                    'kernel_size': int(kernel_size)
                }
                print(mod, 'receptive field = ', mod_receptive_field[mod], ' number of levels = ', num_levels)

        feat_ext = PlantFeatureExtractor(**feat_extractor_params).to(device)

        # because the criterion is cross entropy which in pytorch includes softmax,
        # there's no need for softmax layer as last layer in the net below (nn.Softmax(dim=1))
        print('len parameters = ', parameters.net_features_dim, '# of classes =', len(classes[parameters.experiment]))
        print()
        print('linear layer ')
        print('input ', parameters.net_features_dim, 'output ', parameters.net_features_dim // 2)

        label_cls = nn.Sequential(
            nn.ReLU(),
            nn.Linear(parameters.net_features_dim, len(classes[parameters.experiment])),
            nn.Dropout(p=0.2, inplace=False),
        ).to(device)

        plant_cls = nn.Sequential(
            nn.ReLU(),
            nn.Linear(parameters.net_features_dim, train_set.num_plants),
            nn.Dropout(p=0.35, inplace=False)
        ).to(device)

        criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

        # TODO - for fast label learn use Adam. For slow Plant learn - use SGD
        if parameters.optimizer == 'SGD':
            label_opt = optim.SGD(label_cls.parameters(), lr=parameters.label_lr, weight_decay=1e-3)  # , momentum=0.7
            plant_opt = optim.SGD(plant_cls.parameters(), lr=parameters.plant_lr, weight_decay=1e-3)  # momentum=0.9,
            ext_opt = optim.SGD(feat_ext.parameters(), lr=parameters.extractor_lr, weight_decay=1e-3)  # momentum=0.7,
        elif parameters.optimizer == 'Adam':
            label_opt = optim.Adam(label_cls.parameters(), lr=parameters.label_lr_Adam, betas=(0.9, 0.999),
                                   eps=1e-08, weight_decay=1e-3, amsgrad=False)
            plant_opt = optim.Adam(plant_cls.parameters(), lr=parameters.plant_lr_Adam, betas=(0.9, 0.999),
                                   eps=1e-08, weight_decay=1e-3, amsgrad=False)
            ext_opt = optim.Adam(feat_ext.parameters(), lr=parameters.extractor_lr_Adam, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=1e-3, amsgrad=False)
        else:
            label_opt = None
            plant_opt = None
            ext_opt = None

        best_loss = float('inf')

        test_config = TestConfig(checkpoint_name, device, dataset, train_set, test_set, train_loader, feat_ext,
                                 label_cls, plant_cls, criterion, label_opt, plant_opt, ext_opt, best_loss)

        if parameters.load_checkpoint:
            restore_checkpoint(test_config)

        train_label_losses = []
        train_plant_losses = []
        train_accuracy_prog = []
        test_accuracy = []
        test_losses = []
        # TRAIN LOOP CALL
        train_label_losses, train_plant_losses, train_accuracy_prog, \
            test_accuracy, test_losses, conf_matrix, \
            class_report = train_loop(
                test_config,
                train_label_losses,
                train_plant_losses,
                train_accuracy_prog,
                test_accuracy,
                test_losses,
                run)

        t_label_losses.append(train_label_losses)
        t_plant_losses.append(train_plant_losses)
        t_accuracy_prog.append(train_accuracy_prog)
        val_accuracy.append(test_accuracy)
        val_losses.append(test_losses)
        conf.append(conf_matrix)

        if parameters.use_checkpoints:
            restore_checkpoint(test_config)

        run += 1

    print('t_label_losses  = ', t_label_losses)
    print('t_plant_losses = ', t_plant_losses)
    print('t_accuracy_prog = ', t_accuracy_prog)
    print('val_accuracy = ', val_accuracy)
    print('val_losses = ', val_losses)
    print('confusion matrix sum :')
    conf_sum = sum(c for c in conf)
    conf_sum = conf_sum / run
    print(conf_sum)

    val_losses_avg = np.mean(val_losses, axis=0)
    val_accuracy_avg = np.mean(val_accuracy, axis=0)
    t_accuracy_avg = np.mean(t_accuracy_prog, axis=0)
    t_plant_loss_avg = np.mean(t_plant_losses, axis=0)
    t_label_losses_avg = np.mean(t_label_losses, axis=0)

    train_name: str = get_training_name(parameters.experiment, parameters.excluded_modalities)

    # plot accuracies
    plt.figure()
    plt.plot(t_accuracy_avg, 'or', label='avg train accuracy')
    plt.plot(val_accuracy_avg, 'ob', label='avg validation accuracy')
    plt.legend(title='Parameters')
    plt.title(f'{parameters.experiment} Train and Validation Avg Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f'{parameters.results_path}{train_name}_AVG_accuracy_{datetime.date(datetime.now())}_run_{run}.png')
    plt.close()

    # plot losses
    plt.figure()
    plt.plot(t_label_losses_avg, 'or', label='avg train label loss')
    plt.plot(t_plant_loss_avg, 'ob', label='avg train plant loss')
    plt.plot(val_losses_avg, 'og', label='avg validation label loss')
    plt.title(f'{parameters.experiment} Train and Validation Avg Losses')
    plt.legend(title='Parameters')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{parameters.results_path}{train_name}_AVG_loss_{datetime.date(datetime.now())}_run_{run}.png')
    plt.close()

    # plot confusion matrix summary
    seaborn.set(color_codes=True)
    plt.figure(figsize=(9, 6))
    plt.title(f'{parameters.experiment} Confusion Matrix Summary')
    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(conf_sum, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
    ax.set_xticklabels(classes[parameters.experiment])
    ax.set_yticklabels(classes[parameters.experiment])
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    train_name: str = get_training_name(parameters.experiment, parameters.excluded_modalities)
    plt.savefig(f'{parameters.results_path}{train_name}_confusion_summary_{datetime.date(datetime.now())}_run_{run}.png',
                bbox_inches='tight',
                dpi=300)
    plt.close()

    print('Class Report = ', class_report)


if __name__ == '__main__':
    main()
