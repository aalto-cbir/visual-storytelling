import configparser
import glob
import json
import os
import sys
import zipfile
from collections import namedtuple

import nltk
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

# noinspection PyUnresolvedReferences
from sources.general.vocabulary import Vocabulary  # (Needed to handle Vocabulary pickle)

DatasetConfig = namedtuple('DatasetConfig',
                           'name, child_split, dataset_class, image_dir, caption_path, '
                           'vocab_path, features_path, subset')


def _print_info(configs):
    """
    print out details about datasets being configured
    :param configs: datasets.conf file content
    """
    for ds in configs:
        print('[Dataset]', ds.name)
        for name, value in ds._asdict().items():
            if name != 'name' and value is not None:
                print('    {}: {}'.format(name, value))


def _get_config_path(dataset_config_file):
    """
    :param dataset_config_file: config file location
    :return: validated config file path
    """
    # List of places to look for dataset configuration - in this order:
    # In current working directory
    conf_in_working_dir = dataset_config_file
    # In user configuration directory:
    conf_in_user_dir = os.path.expanduser(os.path.join("~/.config/image_captioning",
                                                       dataset_config_file))
    # Inside code folder:
    file_path = os.path.realpath(__file__)
    conf_in_code_dir = os.path.join(os.path.dirname(file_path), dataset_config_file)

    search_paths = [conf_in_working_dir, conf_in_user_dir, conf_in_code_dir]

    config_path = None
    for i, path in enumerate(search_paths):
        if os.path.isfile(path):
            config_path = path
            break

    return config_path


def _cfg_path(cfg, s):
    path = cfg.get(s)
    if path is None or os.path.isabs(path):
        return path
    else:
        root_dir = cfg.get('root_dir', '')
        return os.path.join(root_dir, path)


def _get_param(d, param, default):
    if not d or param not in d or not d[param]:
        return default
    return d[param]


def _lmdb_to_numpy(value):
    return np.frombuffer(value, dtype=np.float32)


class DatasetParams:
    def __init__(self, dataset_config_file=None):
        """
        initialize dataset configuration object, by default loading data from
        configs/datasets.conf file or creating a generic configuration if datasets.conf
        file was not specified or found
        """

        self.config = configparser.ConfigParser()

        config_path = _get_config_path(dataset_config_file)

        # Generic, "fall back" dataset
        self.config['generic'] = {'dataset_class': 'GenericDataset'}

        # If the configuration file is not found, we can still use
        # 'generic' dataset with sensible defaults when infering.
        if not config_path:
            print('Config file not found. Loading default settings for generic dataset.')
            print('Hint: you can use configs/datasets.conf.default as a starting point.')

        # Otherwise all is good, and we are using the config file as
        else:
            print("Loading dataset configuration from {}...".format(config_path))
            self.config.read(config_path)

    def get_params(self, args_dataset, image_dir=None, image_files=None, vocab_path=None):

        datasets = args_dataset.split('+')
        configs = []
        for dataset in datasets:
            dataset = dataset.lower()

            if dataset not in self.config:
                print('No such dataset configured:', dataset)
                sys.exit(1)

            if self.config[dataset]:
                # If dataset is of the form "parent_dataset:child_split",
                # if child doesn't have a parameter specified use fallback values from parent
                cfg, child_split = self._combine_cfg(dataset)

                dataset_name = dataset
                dataset_class = cfg['dataset_class']

                if dataset == 'generic' and (image_files or image_dir):
                    image_list = []
                    if image_files:
                        image_list += image_files
                    if image_dir:
                        image_list += glob.glob(image_dir + '/*.jpg')
                        image_list += glob.glob(image_dir + '/*.jpeg')
                        image_list += glob.glob(image_dir + '/*.png')
                    root = image_list
                else:
                    root = _cfg_path(cfg, 'image_dir')

                caption_path = _cfg_path(cfg, 'caption_path')
                cfg_vocab_path = vocab_path if vocab_path else _cfg_path(cfg, 'vocab_path')
                features_path = _cfg_path(cfg, 'features_path')
                subset = cfg.get('subset')

                dataset_config = DatasetConfig(dataset_name,
                                               child_split,
                                               dataset_class,
                                               root,
                                               caption_path,
                                               cfg_vocab_path,
                                               features_path,
                                               subset)

                configs.append(dataset_config)
            else:
                print('Invalid dataset {:s} specified'.format(dataset))
                sys.exit(1)

        # Vocab path can be overriden from arguments even for multiple configs:
        main_vocab_path = vocab_path if vocab_path else _cfg_path(
            self.config[datasets[0]], 'vocab_path')

        if main_vocab_path is None:
            print("WARNING: Vocabulary path not specified!")

        _print_info(configs)

        return configs, main_vocab_path

    def _combine_cfg(self, dataset):
        """If dataset name is separated by 'parent_dataset:child_split' (i.e. 'coco:train2014')
        fallback to parent settings when child configuration has no corresponding parameter
        included"""
        child_subset = None
        if ":" in dataset:
            (parent_dataset, child_subset) = tuple(dataset.split(':'))

            # Take defaults from parent and override them as needed:
            for key in self.config[parent_dataset]:
                if self.config[dataset].get(key) is None:
                    self.config[dataset][key] = self.config[parent_dataset][key]

        return self.config[dataset], child_subset


class ExternalFeature:
    def __init__(self, filename, base_path):
        full_path = os.path.expanduser(os.path.join(base_path, filename))
        self.lmdb = None
        self.bin = None
        if not os.path.exists(full_path):
            print('ERROR: external feature file not found:', full_path)
            sys.exit(1)
        if filename.endswith('.h5'):
            import h5py
            self.f = h5py.File(full_path, 'r')
            self.data = self.f['data']
        elif filename.endswith('.lmdb'):
            import lmdb
            self.f = lmdb.open(full_path, max_readers=1, readonly=True, lock=False,
                               readahead=False, meminit=False)
            self.lmdb = self.f.begin(write=False)
        else:
            self.data = np.load(full_path)

        x1 = None
        if self.lmdb is not None:
            c = self.lmdb.cursor()
            assert c.first(), full_path
            x1 = _lmdb_to_numpy(c.value())
            self._vdim = x1.shape[0]
        elif self.bin is not None:
            self._vdim = self.bin.vdim()
        else:
            x1 = self.data[0]
            self._vdim = self.data.shape[1]

        assert x1 is None or not np.isnan(x1).any(), full_path

        print('Loaded feature {} with dim {}.'.format(full_path, self.vdim()))

    def vdim(self):
        return self._vdim

    def get_feature(self, idx):
        if self.lmdb is not None:
            x = _lmdb_to_numpy(self.lmdb.get(str(idx).encode('ascii')))
        elif self.bin is not None:
            x = self.bin.get_float(idx)
        else:
            x = self.data[idx]

        return torch.Tensor(x).float()

    @classmethod
    def load_set(cls, feature_loaders, idx):
        return torch.cat([ef.get_feature(idx) for ef in feature_loaders])

    @classmethod
    def load_sets(cls, feature_loader_sets, idx):
        # We have several sets of features (e.g., initial, persistent, ...)
        # For each set we prepare a single tensor with all the features concatenated
        if feature_loader_sets is None:
            return None
        return [cls.load_set(fls, idx) for fls in feature_loader_sets
                if fls]

    @classmethod
    def loaders(cls, features, base_path):
        ef_loaders = []
        feat_dim = 0
        for fn in features:
            ef = cls(fn, base_path)
            ef_loaders.append(ef)
            feat_dim += ef.vdim()
        return ef_loaders, feat_dim


class VISTDataset(data.Dataset):
    """
    load VIST dataset:
    http://visionandlanguage.net/VIST/dataset.html
    """

    def __init__(self, root, json_file, vocab, transform=None, skip_images=False, feature_loaders=None,
                 multi_decoder=False, glac=False):
        """
        :param root: image directory
        :param json_file: VIST annotation file path
        :param vocab: vocabulary wrapper
        :param transform: pytorch image transformer
        :param skip_images: skip reading images if True
        :param feature_loaders: load features from external loaders (like .lmdb)
        :param multi_decoder: flag indicating data load for multi-decoder architecture
        :param glac: flag indicating data load for glac architecture
        """
        global images
        self.root = root
        self.vocab = vocab
        self.transform = transform
        self.skip_images = skip_images
        self.feature_loaders = feature_loaders
        self.sequence_size = 5
        self.multi_decoder = multi_decoder
        self.glac = glac

        if not self.skip_images:
            images = [str(file).split('.')[0] for file in os.listdir(root)]

        with open(json_file) as raw_data:
            json_data = json.load(raw_data)
            self.anns = json_data['annotations']

        seq_idx = 0
        self.data_hold = []
        self.story_ids = []
        while seq_idx < len(self.anns):
            current_story_id = self.anns[seq_idx][0]['story_id']
            _seq_idx = seq_idx
            current_story = []
            current_sequence = []
            current_album_ids = []
            bad_for_testing = False
            while _seq_idx < len(self.anns) and self.anns[_seq_idx][0]['story_id'] == current_story_id:
                current_story.append(self.anns[_seq_idx][0]['text'])
                current_sequence.append(self.anns[_seq_idx][0]['photo_flickr_id'])
                current_album_ids.append(self.anns[_seq_idx][0]['album_id'])

                # validation to skip bad/absent images
                if not self.skip_images and self.anns[_seq_idx][0]['photo_flickr_id'] not in images:
                    bad_for_testing = True

                _seq_idx += 1

            seq_idx = _seq_idx
            if not bad_for_testing:
                current_story = current_story if self.multi_decoder or self.glac else ' '.join(current_story)
                self.data_hold.append([current_sequence, current_story, current_album_ids])
                self.story_ids.append(current_story_id)

        print("... {} sequences loaded ...".format(len(self.story_ids)))
        with open('./resources/mapping.json', 'r') as file:
            self.mapping = json.load(file)

    def __getitem__(self, index):
        """
        :param index: idx between the 0 ... len of dataset
        :return: data formatted w.r.t architecture
        """
        image_ids = self.data_hold[index][0]
        story = self.data_hold[index][1]

        if not self.skip_images:
            sequence = []
            sequence_for_display = []
            feature_sets = []
            for image_id in image_ids:
                image_path = os.path.join(self.root, str(image_id) + '.jpg')
                if os.path.isfile(image_path):
                    image = Image.open(image_path).convert('RGB')
                else:
                    image_path = os.path.join(self.root, str(image_id) + '.png')
                    image = Image.open(image_path).convert('RGB')

                feature_set = ExternalFeature.load_sets(self.feature_loaders, self.mapping[image_id])
                feature_sets.append(feature_set)

                if self.transform is not None and not self.skip_images:
                    image_display = image
                    image = self.transform(image)

                sequence.append(image)
                sequence_for_display.append(image_display)
        else:
            sequence = [torch.Tensor([])] * self.sequence_size
            feature_sets = [None] * self.sequence_size

        if self.multi_decoder:
            targets = get_descriptions(story, self.vocab)
            return sequence, targets[0], targets[1], targets[2], targets[3], targets[4], self.story_ids[
                index], feature_sets
        elif self.glac:
            return sequence_for_display, torch.stack(sequence), \
                   get_descriptions(story, self.vocab), self.data_hold[index][2], image_ids
        else:
            return sequence, tokenize_story(story, self.vocab), self.story_ids[index], feature_sets

    def __len__(self):
        """
        :return: size of dataset
        """
        return len(self.data_hold)


class NonVISTDataset(data.Dataset):
    """
    - load Non VIST data (such as external video frames and image sequences):
    - useful for generating stories using trained VIST models
    """

    def __init__(self, root, json_file, vocab, transform=None, skip_images=False, feature_loaders=None,
                 multi_decoder=False, glac=False):
        """
        :param root: image directory
        :param json_file: VIST annotation file path
        :param vocab: vocabulary wrapper
        :param transform: pytorch image transformer
        :param skip_images: skip reading images if True
        :param feature_loaders: load features from external loaders (like .lmdb)
        :param multi_decoder: flag indicating data load for multi-decoder architecture
        :param glac: flag indicating data load for glac architecture
        """
        global images
        self.root = root
        self.vocab = vocab
        self.transform = transform
        self.display_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.ToPILImage()])
        self.skip_images = skip_images
        self.feature_loaders = feature_loaders
        self.sequence_size = 5
        self.multi_decoder = multi_decoder
        self.glac = glac

        if not self.skip_images:
            files = sorted(filter(lambda f: not f.startswith('.'), os.listdir(root)),
                           key=lambda f: int(''.join(filter(str.isdigit, f))))
            images = [str(file).split('.')[0] for file in files]

        assert len(images) % 5 == 0, 'number of images should be a multiple of 5\n'
        print(f'{len(images)} images found... grouping into {int(len(images) / 5)} sequences\n')

        img_idx = 0
        self.data_hold = []
        self.story_ids = []
        while img_idx < len(images):
            current_story_id = (img_idx / 5) + 1
            current_story = []
            current_sequence = []
            current_album_ids = []
            img_jdx = img_idx
            while img_jdx < img_idx + 5:
                current_story.append(f'father daughter mother brother .')
                current_sequence.append(images[img_jdx])
                current_album_ids.append(current_story_id)
                img_jdx += 1

            img_idx = img_jdx

            current_story = current_story if self.multi_decoder or self.glac else ' '.join(current_story)
            self.data_hold.append([current_sequence, current_story, current_album_ids])
            self.story_ids.append(current_story_id)

        print(f'... {len(self.story_ids)} sequences loaded ...')
        with open('./resources/mapping.json', 'r') as file:
            self.mapping = json.load(file)

    def __getitem__(self, index):
        """
        :param index: idx between the 0 ... len of dataset
        :return: data formatted w.r.t architecture
        """
        image_ids = self.data_hold[index][0]
        story = self.data_hold[index][1]

        if not self.skip_images:
            sequence = []
            sequence_for_display = []
            for image_id in image_ids:
                image_path = os.path.join(self.root, str(image_id) + '.jpg')
                if os.path.isfile(image_path):
                    image = Image.open(image_path).convert('RGB')
                else:
                    image_path = os.path.join(self.root, str(image_id) + '.png')
                    image = Image.open(image_path).convert('RGB')

                if self.transform is not None and not self.skip_images:
                    image_display = image
                    image_display = self.display_transform(image_display)
                    image = self.transform(image)

                sequence.append(image)
                sequence_for_display.append(image_display)

            feature_sets = [None] * self.sequence_size
        else:
            sequence = [torch.Tensor([])] * self.sequence_size
            feature_sets = [None] * self.sequence_size

        if self.multi_decoder:
            targets = get_descriptions(story, self.vocab)
            return sequence, targets[0], targets[1], targets[2], targets[3], targets[4], self.story_ids[
                index], feature_sets
        elif self.glac:
            return sequence_for_display, torch.stack(sequence), \
                   get_descriptions(story, self.vocab), self.data_hold[index][2], image_ids
        else:
            return sequence, tokenize_story(story, self.vocab), self.story_ids[index], feature_sets

    def __len__(self):
        """
        :return: size of dataset
        """
        return len(self.data_hold)


def collate_fn_baseline(_data):
    """
    useful for padding, tensoring and batching
    :param _data: list of __get_item__ objects
    :return: padded and tensored batches
    """
    # sort a data list by story length (descending order)
    _data.sort(key=lambda x: len(x[1]), reverse=True)

    # unzip according to respective dataset class
    sequences, stories, story_ids, feature_sets = zip(*_data)

    # padding & batching
    lengths = [len(story) for story in stories]
    targets = torch.zeros(len(stories), max(lengths)).long()
    for idx, story in enumerate(stories):
        end = lengths[idx]
        targets[idx, :end] = story[:end]
    return sequences, targets, lengths, story_ids, feature_sets


def collate_fn_multi_decoder(_data):
    """
    useful for padding, tensoring and batching
    :param _data: list of __get_item__ objects
    :return: padded and tensored batches
    """

    def transformer(captions):
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        hold = []
        for idx in range(len(lengths)):
            hold.append((lengths[idx], targets[idx]))

        hold.sort(key=lambda tup: tup[0], reverse=True)
        return hold

    sequences, sentences1, sentences2, sentences3, sentences4, sentences5, story_ids, feature_sets = zip(*_data)

    targets1 = transformer(sentences1)
    targets2 = transformer(sentences2)
    targets3 = transformer(sentences3)
    targets4 = transformer(sentences4)
    targets5 = transformer(sentences5)

    return sequences, targets1, targets2, targets3, targets4, targets5, story_ids, feature_sets


def collate_fn_glac(_data):
    image_stories_display, image_stories, caption_stories, album_ids_set, image_ids_set = zip(*_data)

    targets_set = []
    lengths_set = []

    for captions in caption_stories:
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for idx, cap in enumerate(captions):
            end = lengths[idx]
            targets[idx, :end] = cap[:end]

        targets_set.append(targets)
        lengths_set.append(lengths)

    return image_stories_display, image_stories, targets_set, lengths_set, album_ids_set, image_ids_set


def tokenize_story(text, vocab):
    """
    tokenize a single sentence, convert tokens to vocabulary indices,
    and store the vocabulary index array into a torch tensor
    """

    if vocab is None:
        return text

    tokens = nltk.tokenize.word_tokenize(str(text).lower())
    text = [vocab('<start>')]
    text.extend([vocab(token) for token in tokens])
    text.append(vocab('<end>'))
    target = torch.Tensor(text)

    return target


def get_descriptions(story, vocab):
    targets = []
    for sentence in story:
        tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
        sentence_vector = [vocab('<start>')]
        sentence_vector.extend([vocab(token) for token in tokens])
        sentence_vector.append(vocab('<end>'))
        targets.append(torch.Tensor(sentence_vector))

    return targets


def unzip_image_dir(image_dir):
    # Check if $TMPDIR envirnoment variable is set and use that
    env_tmp = os.environ.get('TMPDIR')
    # Also check if the environment variable points to '/tmp/some/dir' to avoid
    # nasty surprises
    if env_tmp and os.path.commonprefix([os.path.abspath(env_tmp), '/tmp']) == '/tmp':
        tmp_root = os.path.abspath(env_tmp)
    else:
        tmp_root = '/tmp'

    extract_path = os.path.join(tmp_root)

    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    with zipfile.ZipFile(image_dir, 'r') as zipped_images:
        print("Extracting training data from {} to {}".format(image_dir, extract_path))
        zipped_images.extractall(extract_path)
        unzipped_dir = os.path.basename(image_dir).split('.')[0]
        return os.path.join(extract_path, unzipped_dir)


def get_dataset_class(cls_name):
    """
    :param cls_name: dataset class value from datasets.conf
    :return: respective class definition
    """
    if cls_name == 'vist':
        return VISTDataset
    elif cls_name == 'non_vist':
        return NonVISTDataset
    else:
        print('Invalid dataset {:s} specified'.format(cls_name))
        sys.exit(1)


def get_loader(dataset_configs, vocab, transform, batch_size, shuffle, num_workers,
               ext_feature_sets=None, skip_images=False, multi_decoder=False, glac=False, _collate_fn=None):
    """
    :param dataset_configs: dataset config blocks from datasets.conf
    :param vocab: vocab pkl file
    :param transform: pytorch image transformer
    :param batch_size: batch size
    :param shuffle: option to shuffle indices
    :param num_workers: number of workers reading/seeking dataloader object
    :param ext_feature_sets: external feature loader file paths
    :param skip_images: option to skip reading images (for building vocab)
    :param multi_decoder: multi-decoder architecture
    :param glac: glac architecture
    :param _collate_fn: useful for customizing batching
    :return: torch.utils.data.DataLoader for user-specified dataset
    """
    global dims
    datasets = []

    for dataset_config in dataset_configs:
        dataset_cls = get_dataset_class(dataset_config.dataset_class)
        root = dataset_config.image_dir
        json_file = dataset_config.caption_path
        fpath = dataset_config.features_path

        loaders = None
        dims = None
        if ext_feature_sets is not None:
            # Construct external feature loaders for each of the specified feature sets
            loaders_and_dims = [ExternalFeature.loaders(fs, fpath) for fs in ext_feature_sets]

            # List of tuples into two lists...
            loaders, dims = zip(*loaders_and_dims)

        # Unzip training images to /tmp/data if image_dir argument points to zip file:
        if isinstance(root, str) and zipfile.is_zipfile(root):
            root = unzip_image_dir(root)

        print((' root={!s:s}\n json_file={!s:s}\n vocab={!s:s}\n +'
               ' transform={!s:s}\n skip_images={!s:s}\n' +
               ' loaders={!s:s}').format(root, json_file, vocab,
                                         transform, skip_images, loaders))

        dataset = dataset_cls(root=root, json_file=json_file, vocab=vocab,
                              transform=transform, skip_images=skip_images,
                              feature_loaders=loaders, multi_decoder=multi_decoder, glac=glac)

        datasets.append(dataset)

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = data.ConcatDataset(datasets)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=_collate_fn if _collate_fn is not None else default_collate)
    return data_loader, dims


if __name__ == '__main__':
    print('Unit testing dataloader on non-vist data...')
    dataset_configs = DatasetParams('./resources/configs/datasets.local.conf')
    dataset_params, vocab_path = dataset_configs.get_params('non_vist:test')
    from sources.general.vocabulary import Vocabulary, get_vocab
    vocab = get_vocab(vocab_path)

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    data_loader_test, _ = get_loader(dataset_configs=dataset_params, vocab=vocab,
                                     transform=transform, batch_size=1, shuffle=False,
                                     num_workers=0, ext_feature_sets=None,
                                     skip_images=False, glac=True, _collate_fn=collate_fn_glac)

    print(f'Data loader size = {len(data_loader_test)}\n')
