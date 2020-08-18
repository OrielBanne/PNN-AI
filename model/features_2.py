import torch
from torch import nn
from torchvision.models import inception_v3
from model.TCN import TemporalConvNet
from typing import Dict, Union


def greyscale_to_rgb(image: torch.Tensor):
    image = image.unsqueeze(-3)
    dims = [-1] * len(image.shape)
    dims[-3] = 3
    return image.expand(*dims)


class ImageFeatureExtractor(nn.Module):  # INCEPTION V3
    def __init__(self):
        super().__init__()
        self.inception = inception_v3(pretrained=True, aux_logits=True)
        # pretrained      - if true, returns a model pre-trained on ImageNet.

        self.inception.fc = nn.Identity()  # adding a last layer FC layer identity
        self.inception.eval()  # evaluation mode

        for p in self.inception.parameters():
            p.requires_grad = False  # here we freeze the pretrained inception model parameters

    # make sure that the inception model stays on eval
    def train(self, mode=True):
        return self

    def forward(self, x: torch.Tensor):
        """
        :param x: a batch of image sequences of either size (NxTxCxHxW) or the squeezed size (NxTxHxW)
        :return: a batch of feature vectors for each image of size (NxTx2048)
        N is batch size N
        T is Time, each image is 1 date_time point
        C is channel
        H - Height
        W Width
        """
        # if the image is greyscale convert it to RGB
        if (len(x.shape) < 5) or (len(x.shape) >= 5 and x.shape[-3] == 1):
            x = greyscale_to_rgb(x)

        # if we got a batch of sequences we have to calculate each sequence separately
        # TODO: understand the reshape below
        n, t = x.shape[:2]
        return self.inception(x.view(-1, *x.shape[2:])).view(n, t, -1)


class ModalityFeatureExtractor(nn.Module):  # Temporal Convolutional Net
    def __init__(self, num_levels: int = 3, num_hidden: int = 64, embedding_size: int = 128, kernel_size=2,
                 dropout=0.2):
        """
        :param num_levels: the number of TCN layers
        :param num_hidden: number of channels used in the hidden layers
        :param embedding_size: size of final feature vector
        :param kernel_size: kernel size, make sure that it matches the feature vector size
        :param dropout: dropout probability
        :return: a TemporalConvNet matching the inputted params
        """
        super().__init__()
        num_channels = [num_hidden] * (num_levels - 1) + [embedding_size]
        self.tcn = TemporalConvNet(2048, num_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor):
        """
        :param x: input tensor of size (NxTx2048),
                where N is the batch size, T is the sequence length and 2048 is the input embedding dim
        :return: tensor of size (N x T x embedding_size) where out[:,t,:] is the output given all values up to time t
        """
        # transpose each sequence so that we get the correct size for the TemporalConvNet
        x = torch.stack([m.t() for m in x])

        # Forward through TCN:
        out = self.tcn(x)

        # undo the previous transpose
        return torch.stack([m.t() for m in out])


class PlantFeatureExtractor(nn.Module):   # TODO make embedding size a parameter at parameters
    def __init__(self, *default_mods: str, embedding_size: int = 512, **param_mods: Dict[str, Union[int, float]]):
        super().__init__()

        # all modalities
        self.mods = list(default_mods) + list(param_mods.keys())

        # create a feature extractor for images - Inception
        self.image_feat_ext = ImageFeatureExtractor()

        # a dictionary for the feature extractors for each modality
        self.mod_extractors: Dict[str, nn.Module] = dict()

        # create a feature extractor with the inputted params for each param modality - TCN
        for mod in param_mods.keys():
            self.mod_extractors[mod] = ModalityFeatureExtractor(**param_mods[mod])
            self.add_module(f'TCN_{mod}_feat_extractor', self.mod_extractors[mod])

        # final feature extractor - linear FC
        d_in = 128 * len(self.mods)  # Exp0: 128* 5 mods = 640//Exp2: 128*6 mods = 768
        d_out = embedding_size  # 512
        h = int((d_in + d_out) / 2)  # 640

        print('D_in  = ', d_in, 'D_out  = ', d_out, ' H = ', h)
        self.final_feat_extractor = nn.Linear(d_in, d_out)  # FC Linear last layer

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def forward(self, **x: torch.Tensor):
        """
        :param x: input of type mod=x_mod for each modality where each x_mod is of shape
                    (NxT_modxC_modxH_modxW_mod),
                    where the batch size N is the same over all mods and others are mod-specific
        :return: a feature vector of shape (N x embedding_size)
        """
        # extract features from each image
        img_feats = {}
        for mod in self.mods:
            img_feats[mod] = self.image_feat_ext(x[mod])

        # extract the features for each mod using the corresponding feature extractor
        mod_feats = {}
        for mod in self.mods:
            mod_feats[mod] = self.mod_extractors[mod](img_feats[mod])  #

        # take the final feature vector from each sequence
        combined_features = torch.cat([mod_feats[mod][:, -1, :] for mod in self.mods], dim=1)

        return self.final_feat_extractor(combined_features)
