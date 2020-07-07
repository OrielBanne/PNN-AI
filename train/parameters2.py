########################################################################################################
#                                                                                                      #
#                                  PARAMETERS AND HYPER PARAMETERS                                     #
#                                                                                                      #
########################################################################################################

from datetime import datetime

print('~~~~~~~~~~~~~IMPORT PARAMETERS2 ~~~~~~~~~~~~~')

global experiment

def init():

    experiment = "Exp0"  # Experiment name
    print('Experiment = ', experiment)
    global experiment_path
    experiment_path = f'/home/pnn/experiments/{experiment}/'
    global num_days
    num_days = None  ### should be None
    global split_cycle
    split_cycle = 1
    global lwir_max_len
    lwir_max_len = 64  # prefer 64
    global lwir_skip
    lwir_skip = 6
    global vir_max_len
    vir_max_len = 16  # need to understand how many of these are the max that exist
    global color_max_len
    color_max_len = 16  # "

    global load_checkpoint
    load_checkpoint = False
    global use_checkpoints
    use_checkpoints = True

    # define start and end dates of the experiment here, this should be a dictionary according to the experiment
    # Exp0
    global start_date
    start_date = datetime(2019, 6, 4)
    global end_date
    end_date = datetime(2019, 7, 7)


    # Choose Device
    global cdevice
    cdevice = "cuda:0"

    # training hyper-parameters       #     added again , '577nm'
    global excluded_modalities
    excluded_modalities = ['noFilter', 'polar', 'polar_a']  # All of the modalities that you don't want to use
    global train_ratio
    train_ratio = 5 / 6  # dataset train/test ratio
    global batch_size
    batch_size = 4  # training batch size
    global loss_delta
    loss_delta = 0.0  # The minimum above the best loss that is allowed for the model to be  saved
    global return_epochs
    return_epochs = 0  # Epochs without improvement allowed (loss_delta included). Return to the best checkpoint otherwise. 0 to disable
    global epochs
    epochs = 50  # number of epochs during train
    global lamda
    lamda = 0.5e-1  # generic lerning rate to start all learners from
    global label_lr
    label_lr = lamda  # phenotype classifier learning rate
    global plant_lr
    plant_lr = lamda  # plant classifier LR used for transfer learning
    global extractor_lr
    extractor_lr = lamda  # feature extractor LR
    global domain_adapt_lr
    domain_adapt_lr = lamda  # domain adaptation learning rate

    #  Clustering Parameters
    global compare_clusters
    compare_clusters = True  # Compare cluster evaluation results for used modalities
    global num_clusters
    num_clusters = 6  # The number of clusters used in the evaluations, default is Exp number of phenotypes
    global load_features
    load_features = False  # action='store_true', default=False
    global cluster_tsne_results
    cluster_tsne_results = '/home/orielban/PNN-AI/PNN-AI/tsne_results/'
    # load features -- when true Loads the features from a file, else - computed from the extractor and saved in a csv file.

    # Clustering - TSNE
    global plot_TSNE
    plot_TSNE = True  # Save a TSNE plot for used modalities
    global PCA
    PCA = 0  # default = 0, no PCA usage . If PCA > 0 -- features will be transformed by PCA with that number of components.
