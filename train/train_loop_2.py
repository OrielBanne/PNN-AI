from torch import nn, optim
from torch.utils import data
from datetime import timedelta, datetime

from datasets import Modalities, ModalitiesSubset, classes, labels
from datasets.transformations import *
from datasets.experiments import get_experiment_modalities_params, experiments_info, get_all_modalities
from model import PlantFeatureExtractor as FeatureExtractor

from train.utils import get_checkpoint_name, get_used_modalities, get_levels_kernel, get_training_name
from train import parameters  # importing all parameters/home/pnn/PNN/train

# imports for plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, show

from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn


class TestConfig:
    def __init__(self, use_checkpoints, checkpoint_name, epochs, batch_size, domain_adapt_lr, device,
                 dataset, train_set, test_set, train_loader, feat_ext, label_cls, plant_cls, criterion,
                 label_opt, plant_opt, ext_opt, best_loss, loss_delta, return_epochs):
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
    test_loader = data.DataLoader(test_config.test_set, batch_size=test_config.batch_size, num_workers=2, shuffle=True)

    confusion = 0  # stam for the return

    classesExp = classes[parameters.experiment]
    print('----------------------------------------------------')
    print('experiment = ', parameters.experiment, 'classes = ', classesExp)
    print('----------------------------------------------------')
    temp = len(classesExp)
    confMatrix = np.zeros([temp, temp])

    test_config.feat_ext.eval()
    test_config.label_cls.eval()
    test_config.plant_cls.eval()

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
            y = labels.data.tolist()
            y_hat = label_out.max(dim=1)[1].tolist()

            for yi, y_hati in zip(y, y_hat):
                confMatrix[yi, y_hati] += 1

    print()
    print('Simple not normalized confusion matrix')
    print()
    print(confMatrix)

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

    plt.title("Confusion Matrix")

    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(confMatrix, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})

    ax.set_xticklabels(classes[parameters.experiment])
    ax.set_yticklabels(classes[parameters.experiment])

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    train_name: str = get_training_name(parameters.experiment, parameters.excluded_modalities)
    plt.savefig(f'/home/pnn/PNN/Results/{train_name}_confusion_{datetime.date(datetime.now())}_run_{run}.png',
                bbox_inches='tight',
                dpi=300)

    plt.close()

    # TODO: Normalized Confusion Matrix

    return confusion


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

    accuracy = tot_correct / len(test_config.test_set)  # Total Test Accuracy
    loss = tot_label_loss / len(test_config.test_set)  # Total Test Loss calculated
    print(f'\t       %8.3f  \t        %8.3f' % (loss, accuracy))

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

    print('  Train:  label loss:         plant loss:         accuracy:    Test: Label loss    accuracy  ')
    for epoch in range(test_config.epochs):
        print(f'epoch {epoch + 1}:', end=' ')

        test_config.feat_ext.train()
        test_config.label_cls.train()
        test_config.plant_cls.train()

        tot_label_loss = 0.
        tot_plant_loss = 0.
        tot_accuracy = 0.
        for i, batch in enumerate(test_config.train_loader, 1):
            labels = batch['label'].to(test_config.device)
            plants = batch['plant'].to(test_config.device)

            x = batch.copy()

            del x['label']
            del x['plant']

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
        print(f'   %8.3f        %8.3f         %8.3f' % ((a), (b), (c)), end=' ')

        train_label_losses.append(a)
        train_plant_losses.append(b)
        train_accuracy_prog.append(c)  # meaning  = train accuracy progress

        test_acc, test_loss = test_model(test_config)
        test_accuracy.append(test_acc)
        test_losses.append(test_loss)

    train_name: str = get_training_name(parameters.experiment, parameters.excluded_modalities)
    # plt.figure(2+run)
    plt.figure()
    plt.plot(train_accuracy_prog, 'or', label='train accuracy')
    plt.plot(test_accuracy, 'ob', label='test accuracy')
    plt.legend(title='Parameters')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f'/home/pnn/PNN/Results/{train_name}_accuracy_{datetime.date(datetime.now())}_run_{run}.png')
    plt.close()

    # plt.figure(3+run)
    plt.figure()
    plt.plot(train_label_losses, 'or', label='train label loss')
    plt.plot(train_plant_losses, 'ob', label='train plant loss')
    plt.plot(test_losses, 'og', label='test label loss')
    plt.title('Train and Test Losses')
    plt.legend(title='Parameters')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'/home/pnn/PNN/Results/{train_name}_loss_{datetime.date(datetime.now())}_run_{run}.png')
    plt.close()

    confusion_matrix(test_config, run)

    return train_label_losses, train_plant_losses, train_accuracy_prog, test_accuracy, test_losses


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

    curr_experiment = experiments_info[parameters.experiment]
    modalities = get_experiment_modalities_params(curr_experiment, parameters.lwir_skip, parameters.lwir_max_len,
                                                  parameters.vir_max_len, parameters.color_max_len)
    used_modalities = get_used_modalities(modalities, parameters.excluded_modalities)
    print(parameters.experiment_path)

    if parameters.experiment_path is None:
        experiment_path = parameters.experiment
    else:
        experiment_path = parameters.experiment_path

    end_date = curr_experiment.end_date
    if parameters.num_days is not None:
        end_date = curr_experiment.start_date + timedelta(days=parameters.num_days - 1)

    print(' CREATING AND PRINTING THE DATASET   ')
    dataset = Modalities(experiment_path, parameters.experiment, split_cycle=parameters.split_cycle,
                         start_date=curr_experiment.start_date, end_date=end_date, **used_modalities)

    # TODO: automate a change in the class labels to enumerate from 0 to len(classes)
    targets = [dataset[i]['label'] for i in range(len(dataset))]
    class_numbers = unique(targets)
    print('class_numbers   = ', class_numbers)

    t_label_losses = []
    t_plant_losses = []
    t_accuracy_prog = []
    val_accuracy = []
    val_losses = []
    # train_set, test_set = ModalitiesSubset.random_split(dataset)
    run = 0
    for train_set, test_set in ModalitiesSubset.cross_validation(dataset):

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
        plt.savefig(f'/home/pnn/PNN/Results/{train_name}_balanced_{datetime.date(datetime.now())}_run_{run}.png',
                    dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)
        # plt.show()  show# TODO: tried to use plt.show(block=False) but it did not show the chart

        train_loader = data.DataLoader(train_set, batch_size=parameters.batch_size, num_workers=2, shuffle=True)

        feat_extractor_params = dict()
        for mod in used_modalities.keys():
            num_levels, kernel_size = get_levels_kernel(dataset.modalities[mod].max_len)
            feat_extractor_params[mod] = {
                'num_levels': num_levels,
                'kernel_size': kernel_size
            }

        feat_ext = FeatureExtractor(**feat_extractor_params).to(device)
        label_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, len(classes[parameters.experiment]))).to(device)
        plant_cls = nn.Sequential(nn.ReLU(), nn.Linear(512, train_set.num_plants)).to(device)

        criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

        label_opt = optim.SGD(label_cls.parameters(), lr=parameters.label_lr, weight_decay=1e-3)  # , momentum=0.7
        plant_opt = optim.SGD(plant_cls.parameters(), lr=parameters.plant_lr, weight_decay=1e-3)  # momentum=0.9,
        ext_opt = optim.SGD(feat_ext.parameters(), lr=parameters.extractor_lr, weight_decay=1e-3)  # momentum=0.7,

        best_loss = float('inf')

        test_config = TestConfig(parameters.use_checkpoints, checkpoint_name, parameters.epochs, parameters.batch_size,
                                 parameters.domain_adapt_lr, device, dataset, train_set, test_set, train_loader,
                                 feat_ext,
                                 label_cls, plant_cls, criterion, label_opt, plant_opt, ext_opt, best_loss,
                                 parameters.loss_delta, parameters.return_epochs)

        if parameters.load_checkpoint:
            restore_checkpoint(test_config)

        train_label_losses = []
        train_plant_losses = []
        train_accuracy_prog = []
        test_accuracy = []
        test_losses = []
        train_label_losses, train_plant_losses, train_accuracy_prog, test_accuracy, test_losses = train_loop(
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

        if parameters.use_checkpoints:
            restore_checkpoint(test_config)

        run += 1

    print('t_label_losses  = ', end='')
    print(t_label_losses)

    print('t_plant_losses = ', end='')
    print(t_plant_losses)

    print('t_accuracy_prog = ', end='')
    print(t_accuracy_prog)

    print('val_accuracy = ', end='')
    print(val_accuracy)

    print('val_losses = ', end='')
    print(val_losses)


if __name__ == '__main__':
    main()
