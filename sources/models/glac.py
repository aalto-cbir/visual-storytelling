from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.models as models
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderCNN(nn.Module):
    def __init__(self, target_size):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(resnet.fc.in_features, target_size)
        self.bn = nn.BatchNorm1d(target_size, momentum=0.01)
        self.init_weights()

    def get_params(self):
        return list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.dropout(features)
        features = self.linear(features)
        features = self.bn(features)
        return features


class EncoderStory(nn.Module):
    def __init__(self, img_feature_size, hidden_size, n_layers):
        super(EncoderStory, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn = EncoderCNN(img_feature_size)
        self.lstm = nn.LSTM(img_feature_size, hidden_size, n_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size * 2 + img_feature_size, hidden_size * 2)
        self.bn = nn.BatchNorm1d(hidden_size * 2, momentum=0.01)

    def get_params(self):
        return self.cnn.get_params() + list(self.lstm.parameters()) + list(self.linear.parameters()) + list(
            self.bn.parameters())

    def forward(self, story_images):
        data_size = story_images.size()
        cnn_features = self.cnn(story_images.view(-1, data_size[2], data_size[3], data_size[4]))
        output, hidden = self.lstm(cnn_features.view(data_size[0], data_size[1], -1))
        output = torch.cat((cnn_features.view(data_size[0], data_size[1], -1), output), 2)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.bn(output.contiguous().view(-1, self.hidden_size * self.n_layers)).view(data_size[0],
                                                                                              data_size[1], -1)
        return output, hidden


class DecoderStory(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab):
        super(DecoderStory, self).__init__()

        self.embed_size = embed_size
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.rnn = DecoderRNN(embed_size, hidden_size, 2, vocab)
        self.init_weights()

    def get_params(self):
        return list(self.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, story_feature, captions, lengths):
        story_feature = self.dropout(story_feature)
        story_feature = self.linear(story_feature)
        story_feature = nnf.relu(story_feature)
        result = self.rnn(story_feature, captions, lengths)
        return result

    def inference(self, story_feature):
        story_feature = self.dropout(story_feature)
        story_feature = self.linear(story_feature)
        story_feature = nnf.relu(story_feature)
        result = self.rnn.inference(story_feature)
        return result

    def inference_for_sc(self, story_feature):
        story_feature = self.dropout(story_feature)
        story_feature = self.linear(story_feature)
        story_feature = nnf.relu(story_feature)
        result = self.rnn.inference_for_sc(story_feature)
        return result


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, vocab):
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        vocab_size = len(vocab)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(0)
        self.log_softmax = nn.LogSoftmax(0)

        self.brobs = []

        self.init_input = torch.zeros([5, 1, embed_size], dtype=torch.float32)

        if torch.cuda.is_available():
            self.init_input = self.init_input.to(device)

        self.start_vec = torch.zeros([1, vocab_size], dtype=torch.float32)
        self.start_vec[0][1] = 10000
        if torch.cuda.is_available():
            self.start_vec = self.start_vec.to(device)

        self.init_weights()

    def get_params(self):
        return list(self.parameters())

    def init_hidden(self):
        hidden = (torch.zeros(self.n_layers, 1, self.hidden_size).to(device),
                  torch.zeros(self.n_layers, 1, self.hidden_size).to(device))
        return hidden

    def init_weights(self):
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = self.dropout1(embeddings)
        features = features.unsqueeze(1).expand(-1, np.amax(lengths), -1)
        embeddings = torch.cat((features, embeddings), 2)

        outputs = []
        hidden = self.init_hidden()
        for i, length in enumerate(lengths):
            lstm_input = embeddings[i][0:length - 1]
            output, hidden = self.lstm(lstm_input.unsqueeze(0), hidden)
            output = self.dropout2(output)
            output = self.linear(output[0])
            output = torch.cat((self.start_vec, output), 0)
            outputs.append(output)

        return outputs

    def inference(self, features):
        results = []
        hidden = self.init_hidden()
        vocab = self.vocab
        end_vocab = vocab('<end>')
        forbidden_list = [vocab('<pad>'), vocab('<start>'), vocab('<unk>')]
        termination_list = [vocab('.'), vocab('?'), vocab('!')]
        function_list = [vocab('<end>'), vocab('.'), vocab('?'), vocab('!'), vocab('the'), vocab('a'), vocab('an'),
                         vocab('of'), vocab('am'),
                         vocab('is'), vocab('was'), vocab('are'), vocab('were'), vocab('do'), vocab('does'),
                         vocab('did'), vocab('have'),
                         vocab('has'), vocab('had'), vocab('and'), vocab('or'), vocab('they'), vocab('he'),
                         vocab('she'), vocab('them'), vocab('him'),
                         vocab('her'), vocab('not')]

        cumulated_word = []
        for feature in features:

            feature = feature.unsqueeze(0).unsqueeze(0)
            predicted = torch.tensor([1], dtype=torch.long).to(device)
            pred_emb = self.embed(predicted).unsqueeze(0)
            lstm_input = torch.cat((feature, pred_emb), 2)
            sampled_ids = [predicted, ]

            for _ in range(50):
                outputs, hidden = self.lstm(lstm_input, hidden)
                outputs = self.linear(outputs.squeeze(1))

                if predicted not in termination_list:
                    outputs[0][end_vocab] = -100.0

                for forbidden in forbidden_list:
                    outputs[0][forbidden] = -100.0

                cumulated_counter = Counter()
                cumulated_counter.update(cumulated_word)

                prob_res = outputs[0]
                prob_res = self.softmax(prob_res)

                for word, cnt in cumulated_counter.items():
                    if cnt > 0 and word not in function_list:
                        prob_res[word] = prob_res[word] / (1.0 + cnt * 5.0)
                prob_res = prob_res * (1.0 / prob_res.sum())

                candidate = []
                for _ in range(100):
                    index = np.random.choice(prob_res.size()[0], 1, p=prob_res.cpu().detach().numpy())[0]
                    candidate.append(index)

                counter = Counter()
                counter.update(candidate)

                predicted, _ = counter.most_common(1)[0]
                cumulated_word.append(predicted)

                predicted = torch.from_numpy(np.array([predicted])).to(device)
                sampled_ids.append(predicted)

                if predicted == 2:
                    break

                lstm_input = torch.cat((feature, self.embed(predicted).unsqueeze(1)), 2)

            results.append(sampled_ids)

        return results

    def inference_for_sc(self, features):
        results = []
        hidden = self.init_hidden()
        vocab = self.vocab
        end_vocab = vocab('<end>')
        forbidden_list = [vocab('<pad>'), vocab('<start>'), vocab('<unk>')]
        termination_list = [vocab('.'), vocab('?'), vocab('!')]
        function_list = [vocab('<end>'), vocab('.'), vocab('?'), vocab('!'), vocab('the'), vocab('a'), vocab('an'),
                         vocab('of'), vocab('am'),
                         vocab('is'), vocab('was'), vocab('are'), vocab('were'), vocab('do'), vocab('does'),
                         vocab('did'), vocab('have'),
                         vocab('has'), vocab('had'), vocab('and'), vocab('or'), vocab('they'), vocab('he'),
                         vocab('she'), vocab('them'), vocab('him'),
                         vocab('her'), vocab('not')]

        cumulated_word = []
        for feature in features:

            feature = feature.unsqueeze(0).unsqueeze(0)
            predicted = torch.tensor(1, dtype=torch.long).to(device)
            pred_emb = self.embed(predicted).unsqueeze(0)
            lstm_input = torch.cat((feature, pred_emb.unsqueeze(1)), 2)
            sampled_ids = [predicted, ]
            start_prob = torch.zeros(len(self.vocab))
            start_prob[1] = 1.0
            start_log_prob = torch.log(start_prob)
            log_probs = [start_log_prob.to(device), ]

            for _ in range(15):
                outputs, hidden = self.lstm(lstm_input, hidden)
                outputs = self.linear(outputs.squeeze(1))

                if predicted not in termination_list:
                    outputs[0][end_vocab] = -100.0

                for forbidden in forbidden_list:
                    outputs[0][forbidden] = -100.0

                cumulated_counter = Counter()
                cumulated_counter.update(cumulated_word)

                logit = outputs[0]
                prob_res = self.softmax(logit)
                log_prob_res = self.log_softmax(logit)

                for word, cnt in cumulated_counter.items():
                    if cnt > 0 and word not in function_list:
                        prob_res[word] = prob_res[word] / (1.0 + cnt * 5.0)
                prob_res = prob_res * (1.0 / prob_res.sum())

                candidate = []
                for _ in range(100):
                    index = np.random.choice(prob_res.size()[0], 1, p=prob_res.cpu().detach().numpy())[0]
                    candidate.append(index)

                counter = Counter()
                counter.update(candidate)

                predicted, _ = counter.most_common(1)[0]
                cumulated_word.append(predicted)

                predicted = torch.from_numpy(np.array([predicted])).to(device)
                sampled_ids.append(predicted)
                log_probs.append(log_prob_res)

                # if predicted == 2:
                #    break

                lstm_input = torch.cat((feature, self.embed(predicted).unsqueeze(1)), 2)

            results.append((sampled_ids, log_probs))

        return results
