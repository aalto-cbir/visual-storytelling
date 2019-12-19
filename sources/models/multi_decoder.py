import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from models.common import FeatureExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, p):
        """
        initialize lstm cell for reading image sequences
        :param p: model parameters
        """
        super(Encoder, self).__init__()

        self.feature_extractor = FeatureExtractor(p.extractor).to(device) if p.extractor is not None else None
        self.rnn = nn.GRU(input_size=p.input_size,
                          hidden_size=p.hidden_size,
                          num_layers=p.num_layers,
                          batch_first=True,
                          dropout=p.dropout)
        self.dropout = nn.Dropout(p.dropout)

    def forward(self, images):
        """
        :param images: VIST images in batches
        :return: last cell hidden state as sequence-context-vector
        """
        sequence_features = self.get_features(images)
        _, h_n = self.rnn(sequence_features)
        return h_n, sequence_features

    def get_features(self, images):
        """
        :param images: VIST iamges
        :return: image features from a pretrained feature extractor CNN
        """
        if len(images.shape) != 5:
            return images
        features = images.view((-1, images.shape[2], images.shape[3], images.shape[4]))
        features = self.feature_extractor.extractor(features)
        features = torch.autograd.Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        return torch.reshape(features, (images.shape[0], -1, features.shape[1]))


class Decoder(nn.Module):
    def __init__(self, p, vocab_size, embedder, max_seq_length=15):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        self.embed = embedder
        self.rnn = nn.GRU(input_size=p.embed_size,
                          hidden_size=p.hidden_size,
                          num_layers=p.num_layers,
                          batch_first=True,
                          dropout=p.dropout)

        self.out = nn.Linear(p.hidden_size, vocab_size)
        self.feature_transformer = nn.Linear(p.input_size, p.embed_size)
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(p.dropout)

    def forward(self, context_vector, image_features, sentence_stories, lengths):
        """learn context vector and training story through teacher forcing.
        :param context_vector: images sequence context vectors (sequence encoder final hidden)
        :param image_features: image features extracted from pretrained CNN
        :param sentence_stories: single image descriptions from the sequence
        :param lengths: number of words in the descriptions of the batch
        :return: output sentence stories of forward pass, which is then back-propagated
        """
        sentence_stories = sentence_stories[:, :-1]
        embeddings = self.embed(sentence_stories)
        image_features = self.feature_transformer(torch.reshape(image_features, [image_features.shape[0], 1, -1]))
        embeddings = torch.cat((image_features, embeddings), 1)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        output, _ = self.rnn(packed, context_vector)
        output = self.out(output[0])
        return output

    def generate(self, context_vector, image_features):
        sampled_ids = []
        states = context_vector
        inputs = self.feature_transformer(torch.reshape(image_features, [image_features.shape[0], 1, -1]))
        for i in range(self.max_seq_length):
            hiddens, states = self.rnn(inputs, states)
            outputs = self.out(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids
