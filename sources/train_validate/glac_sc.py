import argparse
import os
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from sources.general.data_loader import DatasetParams, get_loader, collate_fn_glac
from sources.general.loss import SelfCriticalLoss
from sources.general.vocabulary import get_vocab
from sources.models.common import ModelParams
from sources.models.glac import EncoderStory, DecoderStory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class GlacSC:
    def __init__(self, args):
        print(f'model: {self.__class__.__name__}, mode: train, test, save, device: {device.type}')
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        self.test_transform = transforms.Compose([
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
            self._reload_model(torch.load(args.load_model, map_location=lambda storage, loc: storage), args.load_model)

        if args.force_epoch:
            self.start_epoch = args.force_epoch - 1

    def _load_data(self, args):
        dataset_configs = DatasetParams(args.dataset_config_file)
        dataset_params, vocab_path = dataset_configs.get_params(args.dataset)
        self.vocab = get_vocab(vocab_path)

        self.data_loader_train, _ = get_loader(dataset_configs=dataset_params, vocab=self.vocab,
                                               transform=self.train_transform,
                                               batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                               ext_feature_sets=args.features, skip_images=False,
                                               multi_decoder=False, glac=True, _collate_fn=collate_fn_glac)

        if args.test is not None:
            dataset_params, _ = dataset_configs.get_params(args.test)
            self.data_loader_test, _ = get_loader(dataset_configs=dataset_params, vocab=self.vocab,
                                                  transform=self.test_transform, batch_size=1, shuffle=False,
                                                  num_workers=args.num_workers, ext_feature_sets=args.features,
                                                  skip_images=False, multi_decoder=False,
                                                  glac=True, _collate_fn=collate_fn_glac)

    def _build_model(self, args):
        self.encoder = EncoderStory(args.input_size, args.hidden_size, args.num_layers)
        self.decoder = DecoderStory(args.embed_size, args.hidden_size, self.vocab)

        if torch.cuda.device_count() > 1:
            print(f'\nUsing {torch.cuda.device_count()} GPUs!...\n')
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.decoder = torch.nn.DataParallel(self.decoder)

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        params = self.decoder.get_params() + self.encoder.get_params()
        self.optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

        self.rl_criterion = SelfCriticalLoss(self.vocab)
        self.xe_criterion = torch.nn.CrossEntropyLoss()

    def _reload_model(self, state, model_name):
        self.params = ModelParams(state)
        self.start_epoch = state['epoch']
        self.encoder.load_state_dict(state['encoder'])
        self.decoder.load_state_dict(state['decoder'])
        self.optimizer.load_state_dict(state['optimizer'])
        print(f'Loading model {model_name} at epoch {self.start_epoch}')

    def train_test_save(self, args):
        training_losses = []
        for epoch in range(self.start_epoch, args.num_epochs):
            # self.adjust_learning_rate(epoch, args.learning_rate)

            print('\n============ TRAINING ============\n')
            training_losses.append(self.train(args, epoch))

            print('\n============ TESTING AND STORING RESULTS ============\n')
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                self.test(args, epoch)

            print('\n============ SAVING MODEL ============\n')
            self.save_model(args, epoch)
            print('training losses: ', training_losses)

    def get_greedy_sampled_baseline(self, images):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            features, _ = self.encoder(images)
            greedy_baseline = self.decoder.inference(features.squeeze(0))

        self.encoder.train()
        self.decoder.train()

        greedy_baseline = torch.cat([torch.cat(greedy_baseline[_]) for _ in range(len(greedy_baseline))])

        return [greedy_baseline]

    def train(self, args, epoch):
        self.encoder.train()
        self.decoder.train()
        avg_loss = 0.0

        for bi, (_, image_stories, targets_set, lengths_set, _, _) in enumerate(self.data_loader_train):
            self.decoder.zero_grad()
            self.encoder.zero_grad()
            xe_loss = torch.scalar_tensor(0.0).to(device)
            images = torch.stack(image_stories).to(device)

            greedy_sequences = self.get_greedy_sampled_baseline(images[0].unsqueeze(0)) * images.shape[0]

            sequences = []
            log_probs = []
            ground_truths = []

            features, _ = self.encoder(images)

            for si, data in enumerate(zip(features, targets_set, lengths_set)):
                feature = data[0]
                captions = data[1].to(device)
                lengths = data[2]

                outputs = self.decoder(feature, captions, lengths)
                sample_log_prob = []
                sample_story = []
                ground_truth = []
                for sj, result in enumerate(zip(outputs, captions, lengths)):
                    xe_loss += self.xe_criterion(result[0], result[1][0:result[2]])
                    logits = result[0]
                    log_prob = torch.log_softmax(logits, dim=1)
                    sample_log_prob.append(log_prob)
                    _, sentence = torch.max(torch.softmax(logits, dim=1), dim=1)
                    sample_story.append(sentence)
                    ground_truth.append(result[1][0:result[2]])

                sequences.append(torch.cat(sample_story))
                log_probs.append(torch.cat(sample_log_prob))
                ground_truths.append(torch.cat(ground_truth))

            rl_loss, advantage = self.rl_criterion(sequences, log_probs, greedy_sequences, ground_truths, lengths_set,
                                                return_advantage=True)
            avg_loss += xe_loss.item()
            avg_loss += (5 * rl_loss)

            xe_loss /= (args.batch_size * 5)
            loss = xe_loss + rl_loss

            loss.backward()
            self.optimizer.step()

            if bi % args.log_step == 0:
                print(f'Epoch [{epoch + 1}/{args.num_epochs}], '
                      f'Train Step [{bi + 1}/{len(self.data_loader_train)}], '
                      f'Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):5.4f}, Advantage: {advantage.item()}')

        avg_loss /= (args.batch_size * len(self.data_loader_train) * 5)
        return avg_loss

    def convert_to_sentences(self, idx_form):
        word_form = []
        for sentence in idx_form:
            words = []
            for word_id in sentence:
                word = self.vocab.idx2word[word_id.cpu().item()]
                words.append(word)
                if word == '<end>':
                    break

            words.remove('<start>')
            try:
                words.remove('<end>')
            except Exception:
                pass

            word_form.append(' '.join(words))

        return word_form

    def test(self, args, epoch):
        outputs = []
        for idx, (image_stories_display, image_stories, targets_set, lengths_set, album_ids_set, image_ids_set) \
                in enumerate(self.data_loader_test):
            images = torch.stack(image_stories).to(device)

            features, _ = self.encoder(images)
            inference_results = self.decoder.inference(features.squeeze(0))

            inference_sentences = self.convert_to_sentences(inference_results)
            # target_sentences = self.convert_to_sentences(targets_set[0])

            result = {"duplicated": False, "album_id": album_ids_set[0][0], "photo_sequence": image_ids_set[0],
                      "story_text_normalized": ' '.join(inference_sentences)}

            outputs.append(result)

        GlacSC.save_results(outputs, epoch, args)

    @staticmethod
    def save_results(outputs, epoch, args):

        for i in reversed(range(len(outputs))):
            if not outputs[i]["duplicated"]:
                for j in range(i):
                    if np.array_equal(outputs[i]["photo_sequence"], outputs[j]["photo_sequence"]):
                        outputs[j]["duplicated"] = True

        filtered_res = []
        for result in outputs:
            if not result["duplicated"]:
                del result["duplicated"]
                filtered_res.append(result)

        print(f'{len(filtered_res)} results available, storning to disk!')

        evaluation_info = {"version": "initial version"}
        _output = {"team_name": "Aalto CBIR", "evaluation_info": evaluation_info, "output_stories": filtered_res}

        filename = args.results_path + 'ep' + str(epoch + 1) + '_' + args.results_file
        with open(filename, 'w') as json_file:
            json_file.write(json.dumps(_output))
        json_file.close()

        print(f'saved results to {filename}. back to training.')

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


def main(args):
    _model = GlacSC(args)
    _model.train_test_save(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # prelim parameters (6)
    parser.add_argument('--dataset', type=str, default='vist:train',
                        help='dataset to use')
    parser.add_argument('--dataset_config_file', type=str,
                        default='../../resources/configs/datasets.local.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--load_model', type=str,
                        help='existing model, for continuing training', default='../../resources/models/glac-ep100.pth')
    parser.add_argument('--model_basename', type=str, default='glac_sc',
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

    # test parameters (3)
    parser.add_argument('--test', type=str, default='vist:test',
                        help='dataset to use for testing')
    parser.add_argument('--results_path', type=str, default='./resources/results/',
                        help='path for storing results')
    parser.add_argument('--results_file', type=str, default='results.json',
                        help='results file name')

    # model parameters (5)
    parser.add_argument('--features', type=str, default=None,
                        help='features to use as the initialload input for the '
                             'caption generator, given as comma separated list, '
                             'multiple features are concatenated, '
                             'features ending with .npy are assumed to be '
                             'precalculated features read from the named npy file, '
                             'example: "resnet152,c_in14_gr_pool5_d_ca3.lmdb"')
    parser.add_argument('--input_size', type=int, default=1024,
                        help='dimension of image feature')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=1024,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in lstm')

    # training parameters (6)
    parser.add_argument('--force_epoch', type=int, default=0,
                        help='force start epoch (for broken model files...)')
    parser.add_argument('--num_epochs', type=int, default=101)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    main(parser.parse_args())
