import argparse
import json
import os

import torch
from torchvision import transforms

from sources.general.data_loader import DatasetParams, get_loader, collate_fn_baseline
# noinspection PyUnresolvedReferences
from sources.general.vocabulary import Vocabulary, get_vocab
from sources.models.baseline import Encoder, Decoder
from sources.models.common import ModelParams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Baseline:
    def __init__(self, args):
        self.outputs = []
        print('model: {}, mode: {}, device: {}'.format(self.__class__.__name__,
                                                       'infer & save',
                                                       device.type))
        self.transform = transforms.Compose([
            transforms.Resize(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        self.sequence_length = 5

        print('\n============ LOADING DATA ============\n')
        self._load_data(args)

        print('\n============ LOADING MODEL ============\n')
        self._reload_model(torch.load(args.model_path + args.model_name, map_location=lambda storage, loc: storage))

    def _load_data(self, args):
        dataset_configs = DatasetParams(args.dataset_config_file)
        dataset_params, vocab_path = dataset_configs.get_params(args.dataset)
        self.vocab = get_vocab(vocab_path)

        self.data_loader_test, _ = get_loader(dataset_configs=dataset_params, vocab=self.vocab,
                                              transform=self.transform, batch_size=1, shuffle=False,
                                              num_workers=args.num_workers, ext_feature_sets=args.features,
                                              skip_images=False, multi_decoder=False,
                                              _collate_fn=collate_fn_baseline)

    def _reload_model(self, state):
        self.params = ModelParams(state)
        self.encoder = Encoder(self.params).to(device)
        self.decoder = Decoder(self.params, len(self.vocab)).to(device)
        self.encoder.load_state_dict(state['encoder'])
        self.decoder.load_state_dict(state['decoder'])
        print('model loaded for inference')

    def infer_save(self, args):
        print('\n============ INFERRING ============\n')
        self.infer()

        print('\n============ SAVING ============\n')
        self.save_results(args)

    def infer(self):
        self.encoder.eval()
        self.decoder.eval()
        for idx, (sequences, _, _, story_ids, _) in enumerate(self.data_loader_test):
            sequence_data = Baseline.torchify_sequence(sequences)
            context_vector, sequence_features = self.encoder(sequence_data)
            sampled_ids = self.decoder.generate(context_vector.squeeze(0), sequence_features)
            sampled_ids = sampled_ids[0].cpu().numpy()

            sampled_caption = []
            for word_id in sampled_ids:
                word = self.vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            story = ' '.join(sampled_caption)
            self.outputs.append({'story': story, 'story_id': story_ids})

    def save_results(self, args):
        filename = os.path.join(args.results_path, args.results_file)
        json.dump(self.outputs, open(filename, 'w'))

        if args.print_results:
            for d in self.outputs:
                print('{}: {}'.format(d['story_id'], d['story']))
        print(f'saved results to {filename}')

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
    _model.infer_save(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # prelim parameters
    parser.add_argument('--dataset', type=str, default='non_vist:test',
                        help='dataset to use')
    parser.add_argument('--dataset_config_file', type=str,
                        default='./resources/configs/datasets.local.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--model_path', type=str, default='./resources/models/',
                        help='path for looking for models')
    parser.add_argument('--model_name', type=str, default='baseline-ep100.pth',
                        help='model to load')
    parser.add_argument('--results_path', type=str, default='./resources/results/',
                        help='path for storing results')
    parser.add_argument('--results_file', type=str, default='results_baseline_memad.json',
                        help='results file name')
    parser.add_argument('--print_results', type=bool, default=True,
                        help='option to print inference results')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--tmp_dir_prefix', type=str, default='vist_tmp',
                        help='where in /tmp folder to store project data')
    parser.add_argument('--log_step', type=int, default=1,
                        help='step size for prining log info')

    # model parameters
    parser.add_argument('--extractor', type=str, default='resnet152',
                        help='pretrained feature extractor CNN')
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
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--embed_type', type=str, default='default')
    parser.add_argument('--path_to_weights', type=str, default='./resources/GoogleNews-vectors-negative300.bin')

    main(parser.parse_args())
