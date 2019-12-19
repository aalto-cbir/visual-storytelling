import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from sources.general.data_loader import DatasetParams, get_loader, collate_fn_glac
# noinspection PyUnresolvedReferences
from sources.general.vocabulary import Vocabulary, get_vocab
from sources.models.glac import EncoderStory, DecoderStory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Glac:
    def __init__(self, args):
        self.outputs = []
        print('model: {}, mode: {}, device: {}'.format(self.__class__.__name__,
                                                       'infer & save',
                                                       device.type))

        self.inverse_norm = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])

        self.transform = transforms.Compose([
            transforms.Resize(args.crop_size, interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        self.sequence_length = 5

        print('\n============ LOADING DATA ============\n')
        self._load_data(args)

        print('\n============ LOADING MODEL ============\n')
        self._reload_model(torch.load(args.model_path + args.model_name,
                                      map_location=lambda storage, loc: storage),
                           args)

    def _load_data(self, args):
        dataset_configs = DatasetParams(args.dataset_config_file)
        dataset_params, vocab_path = dataset_configs.get_params(args.dataset)
        self.vocab = get_vocab(vocab_path)

        self.data_loader_test, _ = get_loader(dataset_configs=dataset_params, vocab=self.vocab,
                                              transform=self.transform, batch_size=1, shuffle=False,
                                              num_workers=args.num_workers, ext_feature_sets=args.features,
                                              skip_images=False, multi_decoder=False,
                                              glac=True, _collate_fn=collate_fn_glac)

    def _reload_model(self, state, args):
        self.encoder = EncoderStory(args.input_size, args.hidden_size, args.num_layers).to(device)
        self.decoder = DecoderStory(args.embed_size, args.hidden_size, self.vocab).to(device)

        self.encoder.load_state_dict(state['encoder'])
        self.decoder.load_state_dict(state['decoder'])
        print('model loaded for inference')

    def infer_save(self, args):
        print('\n============ INFERRING ============\n')
        self.infer(args)

        print('\n============ SAVING ============\n')
        self.save_results(args)

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

    @staticmethod
    def display_images(images, save_as='infer.jpg', titles=None):
        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        for idx in range(5):
            axes[idx].imshow(images[idx])
            # axes[idx].set_title(titles[idx], fontsize=11, loc='center')
            axes[idx].axis('off')

        fig.suptitle(' '.join(titles), fontsize='xx-large', va='bottom', fontweight='black')
        fig.subplots_adjust(hspace=0.1)
        fig.savefig(save_as, bbox_inches='tight')

    @staticmethod
    def print_samples(images, actual, expected, idx):
        imgs_show, text_show = [], []
        for combo in zip(images, actual):
            imgs_show.append(combo[0])
            text_show.append(combo[1])

        file_name = 'sample_' + str(idx) + '.jpg'
        Glac.display_images(imgs_show, save_as=file_name, titles=text_show)
        print(f'{file_name}:\n {expected}\n')

    def infer(self, args):
        self.encoder.eval()
        self.decoder.eval()
        for idx, (image_stories_display, image_stories, targets_set, lengths_set, album_ids_set, image_ids_set) \
                in enumerate(self.data_loader_test):
            images = torch.stack(image_stories).to(device)

            features, _ = self.encoder(images)
            inference_results = self.decoder.inference(features.squeeze(0))

            inference_sentences = self.convert_to_sentences(inference_results)
            target_sentences = self.convert_to_sentences(targets_set[0])

            result = {"duplicated": False, "album_id": album_ids_set[0][0], "photo_sequence": image_ids_set,
                      "story_text_normalized": ' '.join(inference_sentences)}

            self.outputs.append(result)

            if idx < 5 and args.print_results:
                Glac.print_samples(image_stories_display[0], inference_sentences, target_sentences, idx)

        print('inference complete.\n')

    def save_results(self, args):

        for i in reversed(range(len(self.outputs))):
            if not self.outputs[i]["duplicated"]:
                for j in range(i):
                    if np.array_equal(self.outputs[i]["photo_sequence"], self.outputs[j]["photo_sequence"]):
                        self.outputs[j]["duplicated"] = True

        filtered_res = []
        for result in self.outputs:
            if not result["duplicated"]:
                del result["duplicated"]
                filtered_res.append(result)

        print(f'{len(filtered_res)} results available, storning to disk!')

        evaluation_info = {"version": "initial version"}
        _output = {"team_name": "Aalto CBIR", "evaluation_info": evaluation_info, "output_stories": filtered_res}

        filename = args.results_path + args.results_file
        with open(filename, 'w') as json_file:
            json_file.write(json.dumps(_output))
        json_file.close()

        print(f'saved results to {args.results_path + args.results_file}')


def main(args):
    _model = Glac(args)
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
    parser.add_argument('--model_name', type=str, default='glac_sc-ep100.pth',
                        help='model to load')
    parser.add_argument('--results_path', type=str, default='./resources/results/',
                        help='path for storing results')
    parser.add_argument('--results_file', type=str, default='results_glac_sc_memad.json',
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
    parser.add_argument('--num_workers', type=int, default=0)

    main(parser.parse_args())
