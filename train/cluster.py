########################################################################################################
#                                                                                                      #
#                                            CLUSTERING                                                #
#                                                                                                      #
########################################################################################################

import torch
from torch.utils import data

import pandas as pd

import seaborn as sns

from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import mixture
from sklearn import metrics

from datetime import timedelta
from typing import List

from datasets.labels import classes
from datasets import Modalities
from datasets.experiments import experiments_info, get_experiment_modalities_params, get_all_modalities
from model.feature_extraction import PlantFeatureExtractor as FeatureExtractor
from .utils import *
from train.parameters import *  # importing all parameters


# imports for plotting 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""  LOGGING SETUP   """
from datetime import datetime, timedelta
import logging #log files package
logging.basicConfig(filename = 'Clustering.log', level = logging.INFO, format = '%(asctime)s :%(message)s')

#torch.cuda.set_device(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def load_extractor(experiment_name: str, feat_extractor_params, excluded_modalities: List[str] = []):
    checkpoint_name = get_checkpoint_name(experiment_name, excluded_modalities)

    feat_extractor = FeatureExtractor(**feat_extractor_params)#.to(device)# check is .to(device) needed twice (here and at return) - remove here:

    checkpoint = torch.load(f'checkpoints/{checkpoint_name}')
    feat_extractor.load_state_dict(checkpoint['feat_ext_state_dict'])

    return feat_extractor.to(device)


def extract_features(modalities, split_cycle: int, start_date, end_date, experiment_name: str,
                     experiment_root_dir: str, excluded_modalities: List[str] = []):

    used_modalities = get_used_modalities(modalities, excluded_modalities)
    dataset = Modalities(experiment_root_dir, experiment_name, split_cycle=split_cycle, start_date=start_date,
                         end_date=end_date, **used_modalities)
                         
    print('feature extractor params = dict ')
    feat_extractor_params = dict()
    for mod in used_modalities.keys():
        num_levels, kernel_size = get_levels_kernel(dataset.modalities[mod].max_len)
        feat_extractor_params[mod] = {
            'num_levels': num_levels,
            'kernel_size': kernel_size
        }

    print('now feature extractor = load extractor')
    print() 
    feat_extractor = load_extractor(experiment_name, feat_extractor_params, excluded_modalities).eval()
    print('dataloader ')
    print('dataset = the tensor was printed here')
    dataloader = data.DataLoader(dataset, batch_size=4, num_workers=0) # for testset - data.DataLoader(test_set, batch size = , num workers =)

    print('dataloader size')     # ,dataloader,list(dataloader))
    print(len(dataloader.dataset))
    print()

    print(' out of dataloader, start creating a dataframe')
    df = pd.DataFrame()
    print('dataframe created starting batches ')
    #print('for batch in dataloader:')
    # print('DeBug  ---------------------------------------------------')
    # debug =1
    # print('debug      = ',debug)
    #####################################################################
    # Get a batch of training data
    # batch = next(iter(dataloader))
    # print('Batch shape:',batch['lwir'].shape)
    # print('DeBug End    ')
    
    #############################################################################################################
    for batch in dataloader:
    #while debug:
        # print('batch :' )
        # print('for key in batch, key is :')
        for key in batch:
            # print('key    = ',key)
            batch[key] = batch[key].to(device)

        # print('labels and plants' )
        labels = batch['label'].cpu().numpy()
        labels = list(map(lambda i: classes[experiment_name][i], labels))
        plants = batch['plant'].cpu().numpy()

        x = batch.copy()

        del x['label']
        del x['plant']

        features = feat_extractor(**x).cpu().detach().numpy()

        batch_df = pd.DataFrame(data=features)
        batch_df.loc[:, 'label'] = labels
        batch_df.loc[:, 'plant'] = plants

        df = df.append(batch_df, ignore_index=True)
        debug = 0

    df.to_csv(f"/home/orielban/PNN-AI/PNN-AI/saved_features/{get_feature_file_name(experiment_name, excluded_modalities)}", index=False)
    print('Finished extraction.')
    print('return data frame ')
    return df


def pca_features(df: pd.DataFrame, n_components=50):
    pca = PCA(n_components=n_components)

    labels = df['label']
    plants = df['plant']
    df.drop('label', axis=1, inplace=True)
    df.drop('plant', axis=1, inplace=True)

    pca_results = pca.fit_transform(df.values)

    df = pd.DataFrame(pca_results)
    df.loc[:, 'label'] = labels
    df.loc[:, 'plant'] = plants

    return df


def plot_tsne(df: pd.DataFrame, experiment_name: str, excluded_modalities=[], pca=0):
    if pca > 0:
        df = pca_features(df, pca)
        print('Finished PCA.')


    # Pandas preparation of data for seaborn
    print('experiment_name   =  :',experiment_name)
    print('tsne = TSNE(n_components=2, verbose=True)')
    tsne = TSNE(n_components=2, verbose=True)

    labels = df['label']
    plants = df['plant']
    df.drop('label', axis=1, inplace=True)
    df.drop('plant', axis=1, inplace=True)

    print('tsne_results')
    tsne_results = tsne.fit_transform(df.values)
    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]
    df['label'] = labels
    df['plant'] = plants

    print('tsne_df')
    tsne_df = pd.DataFrame(data=tsne_results)
    tsne_df.loc[:, 'label'] = labels

    print('tsne - creating the figures')
    fig = plt.figure(figsize=(25.6, 19.2))
    ax = sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="label",
        palette=sns.color_palette("hls", labels.nunique()),
        data=df,
        legend="full",
        alpha=0.3,
        s=500,
    )

    for x, y, plant in zip(df['tsne-one'], df['tsne-two'], plants):
        ax.annotate(str(plant), (x, y), fontsize='large', ha="center")

    tsne_name: str = get_tsne_name(experiment, excluded_modalities, pca)

    plt.title(f"{tsne_name} clusters".replace('_', ' '))

    fig.savefig(f'tsne_results/{tsne_name}_clusters', bbox_inches="tight")
    tsne_df.to_csv(f'tsne_results/{tsne_name}2d.csv', index=False)

def eval_cluster(labels_true, labels_pred, plants):
    ARI = metrics.adjusted_rand_score(labels_true, labels_pred)
    AMI = metrics.adjusted_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')
    print(f"cluster evaluation: \t ARI: %8.3f   AMI  %8.3f" % ((ARI),(AMI)))
    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    print(f"                    \t Homogeneity: %8.3f   Completeness:  %8.3f" % ((homogeneity),(completeness)))
    plants_completeness = metrics.completeness_score(plants, labels_pred)
    print(f"                    \t V-measure: %8.3f   Plants Completeness:  %8.3f" % ((v_measure),(plants_completeness)))


def cluster_comp(df: pd.DataFrame, num_clusters=6):
    labels = df['label']
    plants = df['plant']
    df.drop('label', axis=1, inplace=True)
    df.drop('plant', axis=1, inplace=True)

    print("KMeans:")
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=100).fit(df.values)
    eval_cluster(labels, kmeans.labels_, plants)

    print("Spectral:")
    spectrals = cluster.SpectralClustering(n_clusters=num_clusters, assign_labels='discretize').fit(df.values)
    eval_cluster(labels, spectrals.labels_, plants)

    print("GMM:")
    gmms = mixture.GaussianMixture(n_components=num_clusters).fit_predict(df.values)
    eval_cluster(labels, gmms, plants)


def get_data_features(modalities):
    if load_features:
        print('Loading Features ')
        return pd.read_csv(f"saved_features/{get_feature_file_name(experiment, excluded_modalities)}")
    else:
        print('curr_experiment = experiment_info    ')
        curr_experiment = experiments_info[experiment]

        print('setting up end date  ')
        end_date = curr_experiment.end_date
        if num_days is not None:
            end_date = curr_experiment.start_date + timedelta(days=num_days-1)

        root_dir = experiment_path
        print('Extract features')
        return extract_features(modalities, split_cycle, curr_experiment.start_date, end_date, experiment,root_dir, excluded_modalities)

def main():
    print('Getting the modalities  ')
    print('________________________')
    modalities = get_experiment_modalities_params(experiments_info[experiment], lwir_skip,lwir_max_len, vir_max_len)
    
    # Plotting TSNE:
    print('Plotting TSNE:  ')
    print('________________________')
    
    plot_tsne(get_data_features(modalities),experiment_path,excluded_modalities,PCA)


    print('cluster_methods_metrics_comparison')
    print('________________________')
    cluster_comp(
        get_data_features(modalities),
        num_clusters if num_clusters > 0 else len(classes[experiment])
    )

if __name__ == '__main__':
    #mods = get_all_modalities()
    print('going to main')
    main()

