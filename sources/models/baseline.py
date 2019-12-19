import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from sources.models.common import FeatureExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, p):
        """
        initialize lstm cell for reading image sequences
        :param p: model parameters
        """
        super(Encoder, self).__init__()

        self.feature_extractor = FeatureExtractor(p.extractor).to(device)
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
        sequence_features = self.get_features(images.view((-1, images.shape[2], images.shape[3], images.shape[4])))
        sequence_features = torch.reshape(sequence_features, (images.shape[0], -1, sequence_features.shape[1]))

        _, h_n = self.rnn(sequence_features)
        return h_n, sequence_features

    def get_features(self, images):
        """
        :param images: VIST iamges
        :return: image features from a pretrained feature extractor CNN
        """
        features = self.feature_extractor.extractor(images)
        features = torch.autograd.Variable(features.data)
        features = features.view(features.size(0), -1)
        return self.dropout(features)


class Decoder(nn.Module):
    def __init__(self, p, vocab_size, max_seq_length=40):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, p.embed_size)
        self.rnn = nn.GRU(input_size=p.embed_size,
                          hidden_size=p.hidden_size,
                          num_layers=p.num_layers,
                          batch_first=True,
                          dropout=p.dropout)

        self.out = nn.Linear(p.hidden_size, vocab_size)
        self.feature_transformer = nn.Linear(p.input_size * 5, p.embed_size)
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(p.dropout)

    def forward(self, context_vector, sequence_features, stories, lengths):
        """learn context vector and training story through teacher forcing.
        :param context_vector: images sequence context vectors (sequence encoder final hidden)
        :param sequence_features: individual image features extracted from pretrained CNN
        :param stories: concat`ed sequence image descriptions
        :param lengths: number of words in the stories of the batch
        :return: output stories of forward pass, which is then back-propagated
        """
        stories = stories[:, :-1]
        embeddings = self.embed(stories)
        sequence_features = self.feature_transformer(
            torch.reshape(sequence_features, [sequence_features.shape[0], 1, -1]))
        embeddings = torch.cat((sequence_features, embeddings), 1)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        output, _ = self.rnn(packed, context_vector)
        output = self.out(output[0])
        return output

    def generate(self, context_vector, sequence_features):
        """Generate story for a given image features using greedy search.
        :param context_vector: images sequence context vector (sequence encoder final hidden)
        :param sequence_features: individual image features extracted from pretrained CNN
        :return: generated story from the trained model
        """
        sampled_ids = []
        states = context_vector
        inputs = self.feature_transformer(torch.reshape(sequence_features, [sequence_features.shape[0], 1, -1]))
        for i in range(self.max_seq_length):
            hiddens, states = self.rnn(inputs, states)
            outputs = self.out(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


class SemanticDecoder(nn.Module):
    def __init__(self, p, vocab_size, max_seq_length=40):
        """Set the hyper-parameters and build the layers."""
        super(SemanticDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, p.embed_size)
        self.rnn = nn.GRU(input_size=p.embed_size * 2,
                          hidden_size=p.hidden_size,
                          num_layers=p.num_layers,
                          batch_first=True,
                          dropout=p.dropout)

        self.out = nn.Linear(p.hidden_size, vocab_size)
        self.feature_transformer = nn.Linear(p.input_size * 5, p.embed_size)
        self.semantic_transformer = nn.Linear(p.semantics_size, p.embed_size)
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(p.dropout)

    def forward(self, context_vector, sequence_features, semantics, stories, lengths):
        """learn context vector and training story through teacher forcing.
        :param context_vector: images sequence context vectors (sequence encoder final hidden)
        :param sequence_features: individual image features extracted from pretrained CNN
        :param semantics: semantics feature vector of characters
        :param stories: concat`ed sequence image descriptions
        :param lengths: number of words in the stories of the batch
        :return: output stories of forward pass, which is then back-propagated
        """
        stories = stories[:, :-1]
        embeddings = self.embed(stories)
        sequence_features = self.feature_transformer(
            torch.reshape(sequence_features, [sequence_features.shape[0], 1, -1]))
        embeddings = torch.cat((sequence_features, embeddings), 1)

        semantics = self.semantic_transformer(semantics.float()).unsqueeze(1)
        # embeddings is [bs, timesteps, 300], semantics is [bs, 1, 300]
        # expand semantics as [bs, timesteps, 300], so .expand(-1, embeddings.shape[1], -1)
        # concat over dim = -1
        semantics = semantics.expand((-1, embeddings.shape[1], -1))
        embeddings = torch.cat((embeddings, semantics), dim=-1)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        output, _ = self.rnn(packed, context_vector)
        output = self.out(output[0])
        return output

    def generate(self, context_vector, sequence_features, semantics):
        """Generate story for a given image features using greedy search.
        :param context_vector: images sequence context vector (sequence encoder final hidden)
        :param sequence_features: individual image features extracted from pretrained CNN
        :param semantics: semantics feature vector of characters
        :return: generated story from the trained model
        """
        sampled_ids = []
        states = context_vector
        inputs = self.feature_transformer(torch.reshape(sequence_features, [sequence_features.shape[0], 1, -1]))
        semantics = self.semantic_transformer(semantics.float()).unsqueeze(1)
        inputs = torch.cat([inputs, semantics], dim=-1)
        for i in range(self.max_seq_length):
            hiddens, states = self.rnn(inputs, states)
            outputs = self.out(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
            inputs = torch.cat([inputs, semantics], dim=-1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids
