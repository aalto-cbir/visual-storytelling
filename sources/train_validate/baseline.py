import argparse
import os
import sys

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from sources.general.data_loader import DatasetParams, get_loader, collate_fn_baseline
# noinspection PyUnresolvedReferences
from sources.general.vocabulary import Vocabulary, get_vocab
from sources.models.baseline import Encoder, Decoder
from sources.models.common import ModelParams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Baseline:
    def __init__(self, args):
        print(f'model: {self.__class__.__name__}, mode: train, validate, save, device: {device.type}')
        self.transform = transforms.Compose([
            transforms.Resize(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        self.sequence_length = 5
        self.start_epoch = 0

        print('\n============ LOADING DATA ============\n')
        self._load_data(args)

        print('\n============ BUILDING MODEL ============\n')
        self.params = ModelParams.fromargs(args)
        self._build_model(args)

        if args.load_model:
            print('\n============ RESUMING PREV MODEL ============\n')
            self._reload_model(torch.load(args.load_model), args.load_model)

        if args.force_epoch:
            self.start_epoch = args.force_epoch - 1

    def _load_data(self, args):
        dataset_configs = DatasetParams(args.dataset_config_file)
        dataset_params, vocab_path = dataset_configs.get_params(args.dataset)
        self.vocab = get_vocab(vocab_path)

        self.data_loader_train, _ = get_loader(dataset_configs=dataset_params, vocab=self.vocab,
                                               transform=self.transform,
                                               batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                               ext_feature_sets=args.features, skip_images=False,
                                               multi_decoder=False, _collate_fn=collate_fn_baseline)

        if args.validate is not None:
            dataset_params, _ = dataset_configs.get_params(args.validate)
            self.data_loader_val, _ = get_loader(dataset_configs=dataset_params, vocab=self.vocab,
                                                 transform=self.transform,
                                                 batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                                 ext_feature_sets=args.features, skip_images=False,
                                                 multi_decoder=False, _collate_fn=collate_fn_baseline)

    def _build_model(self, args):
        self.encoder = Encoder(self.params).to(device)
        self.decoder = Decoder(self.params, len(self.vocab)).to(device)

        opt_params = (list(self.decoder.parameters()) + list(self.encoder.parameters()))
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, opt_params), lr=args.learning_rate)

        self.criterion = torch.nn.CrossEntropyLoss()

    def _reload_model(self, state, model_name):
        self.params = ModelParams(state)
        self.start_epoch = state['epoch']
        self.encoder.load_state_dict(state['encoder'])
        self.decoder.load_state_dict(state['decoder'])
        self.optimizer.load_state_dict(state['optimizer'])
        print(f'Loading model {model_name} at epoch {self.start_epoch}.')

    def adjust_learning_rate(self, epoch, init_lr):
        lr = init_lr * (0.1 ** (epoch // 2))
        print(f'learning at rate: {lr}')
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_validate_save(self, args):
        training_losses = []
        validation_losses = []
        for epoch in range(self.start_epoch, args.num_epochs):
            # self.adjust_learning_rate(epoch, args.learning_rate)

            print('\n============ TRAINING ============\n')
            training_losses.append(self.train(epoch, args.num_epochs, args.log_step))

            print('\n============ VALIDATING ============\n')
            validation_losses.append(self.validate(epoch, args.num_epochs, args.log_step))

            print('\n============ SAVING ============\n')
            self.save_model(args, epoch)
            print('training losses: ', training_losses)
            print('validation losses: ', validation_losses)

    def train(self, epoch, epochs, log_step):
        epoch_loss = 0
        batches = 0
        for idx, (sequences, target_stories, target_lengths, story_ids, feature_sets) in enumerate(
                self.data_loader_train):
            batches += 1
            self.decoder.zero_grad()
            self.encoder.zero_grad()

            sequence_data = Baseline.torchify_sequence(sequences)
            context_vector, sequence_features = self.encoder(sequence_data)

            target_stories = target_stories.to(device)
            targets = pack_padded_sequence(target_stories, target_lengths, batch_first=True)[0]
            outputs = self.decoder(context_vector.squeeze(0), sequence_features, target_stories, target_lengths)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            if (idx + 1) % log_step == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{idx + 1}/{len(self.data_loader_train)}], '
                      f'Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):5.4f}')
                sys.stdout.flush()

        return epoch_loss / batches

    def validate(self, epoch, epochs, log_step):
        epoch_loss = 0
        batches = 0
        for idx, (sequences, target_stories, target_lengths, story_ids, feature_sets) in enumerate(
                self.data_loader_val):
            batches += 1
            sequence_data = Baseline.torchify_sequence(sequences)
            context_vector, sequence_features = self.encoder(sequence_data)

            target_stories = target_stories.to(device)
            targets = pack_padded_sequence(target_stories, target_lengths,
                                           batch_first=True)[0]
            outputs = self.decoder(context_vector.squeeze(0), sequence_features, target_stories, target_lengths)

            loss = self.criterion(outputs, targets)
            epoch_loss += loss.item()
            if (idx + 1) % log_step == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{idx + 1}/{len(self.data_loader_val)}], '
                      f'Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):5.4f}')
                sys.stdout.flush()

        return epoch_loss / batches

    def save_model(self, args, epoch):
        bn = args.model_basename
        file_name = f'{bn}-ep{epoch + 1}.pth'

        state = {
            'epoch': epoch + 1,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'embed_size': self.params.embed_size,
            'hidden_size': self.params.hidden_size,
            'input_size': self.params.input_size,
            'batch_size': self.params.batch_size,
            'num_layers': self.params.num_layers,
            'dropout': self.params.dropout,
            'learning_rate': self.params.learning_rate
        }

        torch.save(state, os.path.join(args.model_path, file_name))
        print(f'model successfully saved to {file_name}')

    @classmethod
    def torchify_sequence(cls, batch):
        batch_tensor = torch.Tensor([])
        for batch_idx in range(len(batch)):
            sequence_tensor = torch.Tensor([])
            image_sequence = batch[batch_idx]
            image_sequence.reverse()
            for image in image_sequence:
                image = image.unsqueeze(0)
                sequence_tensor = torch.cat([sequence_tensor, image])

            sequence_tensor = sequence_tensor.unsqueeze(0)
            batch_tensor = torch.cat([batch_tensor, sequence_tensor])

        return batch_tensor.to(device)


def main(args):
    _model = Baseline(args)
    _model.train_validate_save(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # prelim parameters (6)
    parser.add_argument('--dataset', type=str, default='vist:train',
                        help='dataset to use')
    parser.add_argument('--dataset_config_file', type=str,
                        default='./resources/configs/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--load_model', type=str,
                        help='existing model, for continuing training')
    parser.add_argument('--model_basename', type=str, default='baseline',
                        help='base name for model snapshot filenames')
    parser.add_argument('--model_path', type=str, default='./resources/models/',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--log_step', type=int, default=1,
                        help='step size for prining log info')

    # validation parameters (2)
    parser.add_argument('--validate', type=str,
                        help='validation dataset', default='vist:val')
    # parser.add_argument('--validation_step', type=int, default=1,
    #                     help='epochs in between validation')

    # model parameters (6)
    parser.add_argument('--features', type=str, default=None,
                        help='features to use as the initialload input for the '
                             'caption generator, given as comma separated list, '
                             'multiple features are concatenated, '
                             'features ending with .npy are assumed to be '
                             'precalculated features read from the named npy file, '
                             'example: "resnet152,c_in14_gr_pool5_d_ca3.lmdb"')
    parser.add_argument('--embed_size', type=int, default=250,
                        help='dimension of text embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=1000,
                        help='dimension of RNN hidden states')
    parser.add_argument('--input_size', type=int, default=2048,
                        help='dimension of the input feature vector')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of RNN')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout for the RNN layers')

    # training parameters (7)
    parser.add_argument('--force_epoch', type=int, default=0,
                        help='force start epoch (for broken model files...)')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--embed_type', type=str, default='default')
    parser.add_argument('--path_to_weights', type=str, default='./resources/GoogleNews-vectors-negative300.bin')

    main(parser.parse_args())
