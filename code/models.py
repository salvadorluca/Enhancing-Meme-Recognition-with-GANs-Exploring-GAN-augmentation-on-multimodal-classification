# Combiner architectures based on papers:
# @inproceedings{kumar-nandakumar-2022-hate,
    # title = "Hate-{CLIP}per: Multimodal Hateful Meme Classification based on Cross-modal Interaction of {CLIP} Features",
    # author = "Kumar, Gokul Karthik  and
    #   Nandakumar, Karthik",
    # booktitle = "Proceedings of the Second Workshop on NLP for Positive Impact (NLP4PI)",
    # month = dec,
    # year = "2022",
    # address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    # publisher = "Association for Computational Linguistics",
    # url = "https://aclanthology.org/2022.nlp4pi-1.20",
    # pages = "171--183",
    # abstract = "Hateful memes are a growing menace on social media. While the image and its corresponding text in a meme are related, they do not necessarily convey the same meaning when viewed individually. Hence, detecting hateful memes requires careful consideration of both visual and textual information. Multimodal pre-training can be beneficial for this task because it effectively captures the relationship between the image and the text by representing them in a similar feature space. Furthermore, it is essential to model the interactions between the image and text features through intermediate fusion. Most existing methods either employ multimodal pre-training or intermediate fusion, but not both. In this work, we propose the Hate-CLIPper architecture, which explicitly models the cross-modal interactions between the image and text representations obtained using Contrastive Language-Image Pre-training (CLIP) encoders via a feature interaction matrix (FIM). A simple classifier based on the FIM representation is able to achieve state-of-the-art performance on the Hateful Memes Challenge (HMC) dataset with an AUROC of 85.8, which even surpasses the human performance of 82.65. Experiments on other meme datasets such as Propaganda Memes and TamilMemes also demonstrate the generalizability of the proposed approach. Finally, we analyze the interpretability of the FIM representation and show that cross-modal interactions can indeed facilitate the learning of meaningful concepts. The code for this work is available at https://github.com/gokulkarthik/hateclipper",

# Mapping Memes to Words for Multimodal Hateful Meme Classification

############################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math

class Generator(nn.Module):
    # Add cross layers in future work
    def __init__(self,
                 img_embedding_size: int,
                 text_embedding_size: int,
                 noise_dim: int,
                 hidden_dims: List[int] = [],
                 dropout_prob: float = 0.2,
                 bn: bool = False,
                 act: str = 'relu',
                 normalize_features: bool = False):
        
        super(Generator, self).__init__()
        self.img_embedding_size = img_embedding_size
        self.text_embedding_size = text_embedding_size
        self.noise_dim = noise_dim
        self.hidden_dims = hidden_dims
        self.bn = bn
        self.dropout_prob = dropout_prob
        self.normalize_features = normalize_features

        self.MLP_input_dim = noise_dim
        self.MLP_output_dim = img_embedding_size + text_embedding_size
        self.MLP = MLP(
                       input_dim=self.MLP_input_dim,
                       output_dim=self.MLP_output_dim,
                       hidden_channels=hidden_dims,
                       num_hidden_lyr=len(hidden_dims),
                       dropout_prob=dropout_prob,
                       bn=bn,
                       act=act
                       )

    def forward(self, noise):
        out =  self.MLP(noise)
        img_embedding, text_embedding = torch.split(out, [self.img_embedding_size, self.text_embedding_size], dim=-1)
        if self.normalize_features:
            img_embedding = F.normalize(img_embedding, p=2, dim=-1)
            text_embedding = F.normalize(text_embedding, p=2, dim=-1)
        return img_embedding, text_embedding
              
class Classifier(nn.Module):
    def __init__(self,
                 input_dim,
                 comb_convex_tensor: bool = False,
                 comb_proj: bool = False, 
                 comb_fusion: str = 'concat', 
                 comb_dropout_prob: float = 0.5,
                 classifier_hidden_dims: List[int] = [],
                 act: str = "relu",
                 bn: bool = False,
                 classifiers_dropout_prob: float = 0.2,
                 normalize_features: bool = False
                 ):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.comb_convex_tensor = comb_convex_tensor
        self.comb_proj = comb_proj
        self.comb_fusion = comb_fusion
        self.comb_dropout_prob = comb_dropout_prob
        self.classifier_hidden_dims = classifier_hidden_dims
        self.act = act
        self.bn = bn
        self.classifiers_dropout_prob = classifiers_dropout_prob
        self.normalize_features = normalize_features
        self.combiner_output_dim = input_dim

        # Define combiner
        self.combiner = Combiner(
            convex_tensor=self.comb_convex_tensor, 
            input_dim=input_dim, 
            comb_proj=comb_proj, 
            comb_fusion=comb_fusion, 
            comb_dropout_prob=comb_dropout_prob
        )

        self.classifier =  MLP(
                       input_dim=self.combiner_output_dim,
                       output_dim=1,
                       hidden_channels=classifier_hidden_dims,
                       num_hidden_lyr=len(classifier_hidden_dims),
                       dropout_prob=classifiers_dropout_prob,
                       bn=bn,
                       act=act
                       )
        
    def forward(self, img_embedding, text_embedding):
        # Normalization
        if self.normalize_features:
            img_embedding = F.normalize(img_embedding, p=2, dim=-1)
            text_embedding = F.normalize(text_embedding, p=2, dim=-1)

        # Forward pass
        x = self.combiner(img_embedding, text_embedding)

        if self.normalize_features:
            x = F.normalize(x, p=2, dim=-1)

        prediction = self.classifier(x).squeeze(dim=-1)
        return prediction


class Combiner(nn.Module):
    def __init__(self, convex_tensor: bool, input_dim: int, comb_proj: bool, comb_fusion: str, comb_dropout_prob: float = 0.5):
        super(Combiner, self).__init__()
        self.map_dim = input_dim
        self.comb_proj = comb_proj
        self.comb_fusion = comb_fusion
        self.convex_tensor = convex_tensor          # Sigmoid applied to every feature dimension
        self.comb_dropout_prob = comb_dropout_prob

        if self.convex_tensor:
            branch_out_dim = self.map_dim
        else:
            branch_out_dim = 1

        comb_in_dim = self.map_dim
        comb_concat_out_dim = self.map_dim

        if self.comb_proj:
            self.comb_image_proj = nn.Sequential(
                nn.Linear(comb_in_dim, 2 * comb_in_dim),
                nn.ReLU(),
                nn.Dropout(self.comb_dropout_prob)
            )

            self.comb_text_proj = nn.Sequential(
                nn.Linear(comb_in_dim, 2 * comb_in_dim),
                nn.ReLU(),
                nn.Dropout(self.comb_dropout_prob)
            )

            comb_in_dim = 2 * comb_in_dim

        if self.comb_fusion == 'concat':
            branch_in_dim = 2 * comb_in_dim
        elif self.comb_fusion == 'align':
            branch_in_dim = comb_in_dim
        else:
            ValueError()

        self.comb_shared_branch = nn.Sequential(
            nn.Linear(branch_in_dim, 2 * branch_in_dim),
            nn.ReLU(),
            nn.Dropout(self.comb_dropout_prob),
            nn.Linear(2 * branch_in_dim, branch_out_dim),
            nn.Sigmoid()
        )

        self.comb_concat_branch = nn.Sequential(
            nn.Linear(branch_in_dim, 2 * branch_in_dim),
            nn.ReLU(),
            nn.Dropout(self.comb_dropout_prob),
            nn.Linear(2 * branch_in_dim, comb_concat_out_dim),
        )

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def forward(self, img_features, text_features):
        if self.comb_proj:
            proj_img_fea = self.comb_image_proj(img_features)
            proj_txt_fea = self.comb_text_proj(text_features)
        else:
            proj_img_fea = img_features
            proj_txt_fea = text_features

        if self.comb_fusion == 'concat':
            comb_features = torch.cat([proj_img_fea, proj_txt_fea], dim=1)
        elif self.comb_fusion == 'align':
            comb_features = torch.mul(proj_img_fea, proj_txt_fea)
        else:
            raise ValueError()

        side_branch = self.comb_shared_branch(comb_features)
        central_branch = self.comb_concat_branch(comb_features)

        features = central_branch + ((1 - side_branch) * img_features + side_branch * text_features)

        return features
    
# Code taken from ...
class MLP(nn.Module):
    """mlp can specify number of hidden layers and hidden layer channels"""

    def __init__(
        self,
        input_dim,
        output_dim,
        act="relu",
        num_hidden_lyr=2,
        dropout_prob=0.5,
        return_layer_outs=False,
        hidden_channels=None,
        bn=False,
    ):
        super().__init__()
        self.out_dim = output_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.return_layer_outs = return_layer_outs
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels"
            )
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.act_name = act
        self.activation = create_act(act)
        self.layers = nn.ModuleList(
            list(
                map(
                    self.weight_init,
                    [
                        nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                        for i in range(len(self.layer_channels) - 2)
                    ],
                )
            )
        )
        final_layer = nn.Linear(self.layer_channels[-2], self.layer_channels[-1])
        self.weight_init(final_layer, activation="linear")
        self.layers.append(final_layer)

        self.bn = bn
        if self.bn:
            self.bn = nn.ModuleList(
                [torch.nn.BatchNorm1d(dim) for dim in self.layer_channels[1:-1]]
            )

    def weight_init(self, m, activation=None):
        if activation is None:
            activation = self.act_name
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(activation))
        return m

    def forward(self, x):
        """
        :param x: the input features
        :return: tuple containing output of MLP,
                and list of inputs and outputs at every layer
        """
        layer_inputs = [x]
        for i, layer in enumerate(self.layers):
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                if self.bn:
                    output = self.activation(self.bn[i](layer(input)))
                else:
                    output = self.activation(layer(input))
                layer_inputs.append(self.dropout(output))

        # model.store_layer_output(self, layer_inputs[-1])
        if self.return_layer_outs:
            return layer_inputs[-1], layer_inputs
        else:
            return layer_inputs[-1]


def calc_mlp_dims(input_dim, division=2, output_dim=1):
    dim = input_dim
    dims = []
    while dim > output_dim:
        dim = dim // division
        dims.append(int(dim))
    print("dims: ", dims)
    dims = dims[:-1]
    return dims


def create_act(act, num_parameters=None):
    if act == "relu":
        return nn.ReLU()
    elif act == "prelu":
        return nn.PReLU(num_parameters)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":

        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    else:
        raise ValueError("Unknown activation function {}".format(act))


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)