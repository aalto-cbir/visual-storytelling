import os
import sys
from collections import namedtuple

import gensim
import torch
import torch.nn as nn
import torchvision.models as models
from gensim.scripts.glove2word2vec import glove2word2vec

Features = namedtuple('Features', 'external, internal')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelParams:
    def __init__(self, d):
        self.embed_size = self._get_param(d, 'embed_size', 256)
        self.hidden_size = self._get_param(d, 'hidden_size', 512)
        self.semantics_size = self._get_param(d, 'semantics_size', 1470)
        self.num_layers = self._get_param(d, 'num_layers', 1)
        self.batch_size = self._get_param(d, 'batch_size', 128)
        self.dropout = self._get_param(d, 'dropout', 0)
        self.learning_rate = self._get_param(d, 'learning_rate', 0.001)
        self.features = self._get_features(d, 'features', 'vgg16')
        self.extractor = self._get_param(d, 'extractor', 'resnet152')
        self.persist_features = self._get_features(d, 'persist_features', '')
        self.input_size = self._get_param(d, 'input_size', 4096)
        self.encoder_dropout = self._get_param(d, 'encoder_dropout', 0)

    @classmethod
    def fromargs(cls, args):
        return cls(vars(args))

    @staticmethod
    def _get_param(d, param, default):
        if param not in d or d[param] is None:
            print('WARNING: {} not set, using default value {}'.
                  format(param, default))
            return default
        return d[param]

    def _get_features(self, d, param, default):
        p = self._get_param(d, param, default)

        # If it's already of type Features, just return it
        if hasattr(p, 'internal'):
            return p

        features = p.split(',') if p else []

        ext_feat = []
        int_feat = []
        for fn in features:
            (tmp, ext) = os.path.splitext(fn)
            if ext:
                ext_feat.append(fn)
            else:
                int_feat.append(fn)

        return Features(ext_feat, int_feat)

    def has_persist_features(self):
        return self.persist_features.internal or self.persist_features.external

    def has_internal_features(self):
        return self.features.internal or self.persist_features.internal

    def has_external_features(self):
        return self.features.external or self.persist_features.external

    def update_ext_features(self, ef):
        self.features = self._update_ext_features(ef, self.features)

    def update_ext_persist_features(self, ef):
        self.persist_features = self._update_ext_features(ef,
                                                          self.persist_features)

    @staticmethod
    def _update_ext_features(ef, features):
        return features._replace(external=ef.split(','))

    def __str__(self):
        return '\n'.join(['[ModelParams] {}={}'.format(k, v) for k, v in
                          self.__dict__.items()])


class EmbedSentence(nn.Module):
    def __init__(self, embedding_type, path_to_weights, vocab_size=None, embed_size=None):
        super(EmbedSentence, self).__init__()

        if 'word2vec' in embedding_type:
            if path_to_weights is None:
                print('ERROR: specify path to weight vectors!')
                sys.exit(1)
            model = gensim.models.KeyedVectors.load_word2vec_format(path_to_weights, binary=True)
        elif 'glove' in embedding_type:
            if path_to_weights is None:
                print('ERROR: specify path to weight vectors!')
                sys.exit(1)
            glove2word2vec(path_to_weights, './processed_glove.txt')
            model = gensim.models.KeyedVectors.load_word2vec_format('./processed_glove.txt', binary=False)
            os.remove('./processed_glove.txt')
        elif None not in (vocab_size, embed_size):
            print('using PyTorch text embedding layer')
            self.embed = nn.Embedding(vocab_size, embed_size)
            return
        else:
            print('ERROR: creating embedding layer!'
                  'provide vocab_size, embed_size values or use embedding_type = word2vec|glove')
            sys.exit(1)

        weights = torch.FloatTensor(model.syn0)
        print('shape of text embedding weights: ', weights.shape)
        self.embed = nn.Embedding.from_pretrained(weights)

    def forward(self, sentence):
        return self.embed(sentence)


class FeatureExtractor(nn.Module):
    def __init__(self, model_name, debug=False):
        """Load the pretrained model and replace top fc layer.
        Inception assumes input image size to be 299x299.
        Other models assume input image of size 224x224
        More info: https://pytorch.org/docs/stable/torchvision/models.html """
        super(FeatureExtractor, self).__init__()

        if model_name == 'alexnet':
            if debug:
                print('Using AlexNet, features shape 256 x 6 x 6')
            model = models.alexnet(pretrained=True)
            self.output_dim = 256 * 6 * 6
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
        elif model_name == 'densenet201':
            if debug:
                print('Using DenseNet 201, features shape 1920 x 7 x 7')
            model = models.densenet201(pretrained=True)
            self.output_dim = 1920 * 7 * 7
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
        elif model_name == 'resnet152':
            if debug:
                print('Using resnet 152, features shape 2048')
            self.cnn = models.resnet152(pretrained=True)
            modules = list(self.cnn.children())[:-1]
            self.extractor = nn.Sequential(*modules)
            for param in self.extractor.parameters():
                param.requires_grad = False
        elif model_name == 'vgg16':
            if debug:
                print('Using vgg 16, features shape 4096')
            model = models.vgg16(pretrained=True)
            self.output_dim = 4096
            num_features = model.classifier[6].in_features
            features = list(model.classifier.children())[:-1]
            features.extend([nn.Linear(num_features, self.output_dim)])
            model.classifier = nn.Sequential(*features)
            self.extractor = model
        elif model_name == 'inceptionv3':
            if debug:
                print('Using Inception V3, features shape 1000')
                print('WARNING: Inception requires input images to be 299x299')
            model = models.inception_v3(pretrained=True)
            model.aux_logits = False
            self.extractor = model
        else:
            raise ValueError('Unknown model name: {}'.format(model_name))

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.extractor(images)
        return features.shape

    @classmethod
    def list(cls, internal_features):
        el = nn.ModuleList()
        total_dim = 0
        for fn in internal_features:
            e = cls(fn)
            el.append(e)
            total_dim += e.output_dim
        return el, total_dim
