import argparse
import json
import pickle

import gensim
import numpy as np
from torchvision import transforms

from sources.general.data_loader import DatasetParams, get_loader, collate_fn_multi_decoder
from sources.general.vocabulary import get_vocab, Vocabulary


def read_pickle_dict(loc):
    with open(loc, 'rb') as file:
        loaded_dict = pickle.load(file)
    file.close()
    return loaded_dict


def read_txt_file(loc):
    with open(loc) as file:
        list_loaded = file.readlines()
    file.close()
    return list_loaded


def read_json_dict(loc):
    with open(loc) as file:
        loaded_dict = json.load(file)
    file.close()
    return loaded_dict


def save_it(args):
    dataset_configs = DatasetParams(args.dataset_config_file)
    dataset_params, vocab_path = dataset_configs.get_params(args.dataset)
    vocab = get_vocab(vocab_path)

    # all_seq_features = read_pickle_dict(f'../../resources/story_id_2_seq_feature_train.pkl')
    # all_seq_features.update(read_pickle_dict(f'../../resources/story_id_2_seq_feature_val.pkl'))
    # all_seq_features.update(read_pickle_dict(f'../../resources/story_id_2_seq_feature_test.pkl'))

    all_sentence1_train = read_pickle_dict('../../resources/story_id_2_sentence1_train.pkl')
    all_sentence1_val = read_pickle_dict('../../resources/story_id_2_sentence1_val.pkl')
    all_sentence1_test = read_pickle_dict('../../resources/story_id_2_sentence1_test.pkl')

    all_sentence2_train = read_pickle_dict('../../resources/story_id_2_sentence2_train.pkl')
    all_sentence2_val = read_pickle_dict('../../resources/story_id_2_sentence2_val.pkl')
    all_sentence2_test = read_pickle_dict('../../resources/story_id_2_sentence2_test.pkl')

    all_sentence3_train = read_pickle_dict('../../resources/story_id_2_sentence3_train.pkl')
    all_sentence3_val = read_pickle_dict('../../resources/story_id_2_sentence3_val.pkl')
    all_sentence3_test = read_pickle_dict('../../resources/story_id_2_sentence3_test.pkl')

    all_sentence4_train = read_pickle_dict('../../resources/story_id_2_sentence4_train.pkl')
    all_sentence4_val = read_pickle_dict('../../resources/story_id_2_sentence4_val.pkl')
    all_sentence4_test = read_pickle_dict('../../resources/story_id_2_sentence4_test.pkl')

    all_sentence5_train = read_pickle_dict('../../resources/story_id_2_sentence5_train.pkl')
    all_sentence5_val = read_pickle_dict('../../resources/story_id_2_sentence5_val.pkl')
    all_sentence5_test = read_pickle_dict('../../resources/story_id_2_sentence5_test.pkl')

    # stories_2_semantics = read_pickle_dict('../../resources/story_id_2_semantics.pkl')

    # print(f'{len(all_seq_features)} sequence features loaded to dicts!')
    # print(f'{len(all_stories_train)} train, '
    #       f'{len(all_stories_val)} val, '
    #       f'{len(all_stories_test)} test stories loaded to dicts!')

    # 1 data.p --> (stories lists, word2idx, idx2word)
    train = [list(all_sentence1_train.values()),
             list(all_sentence2_train.values()),
             list(all_sentence3_train.values()),
             list(all_sentence4_train.values()),
             list(all_sentence5_train.values()),
             range(len(all_sentence1_train.keys())),
             list(all_sentence1_train.keys())]

    val = [list(all_sentence1_val.values()),
           list(all_sentence2_val.values()),
           list(all_sentence3_val.values()),
           list(all_sentence4_val.values()),
           list(all_sentence5_val.values()),
           range(len(all_sentence1_val.keys())),
           list(all_sentence1_val.keys())]

    test = [list(all_sentence1_test.values()),
            list(all_sentence2_test.values()),
            list(all_sentence3_test.values()),
            list(all_sentence4_test.values()),
            list(all_sentence5_test.values()),
            range(len(all_sentence1_test.keys())),
            list(all_sentence1_test.keys())]

    word2idx = vocab.word2idx
    idx2word = vocab.idx2word
    data_p = [train, val, test, word2idx, idx2word]
    with open('../../resources/data.pkl', 'wb') as file:
        pickle.dump(data_p, file, protocol=2)
    file.close()

    # 2 resnet_feats.mat --> (image sequences)
    # just_seq_features = np.concatenate(list(all_seq_features.values()), axis=0)
    # print(f'\n...shape of just_seq_features: {just_seq_features.transpose().shape}\n')
    # resnet_feats = {'__header__': 'VIST image sequences features', 'feats': just_seq_features.transpose()}
    # scipy.io.savemat('../../resources/resnet_feats.mat', resnet_feats)
    # print('\nsaved resnet_feats!\n')

    # 3 tag_feats --> (tags/semantics)
    # semantics = list(stories_2_semantics.values())
    # semantics = [a.reshape((1, a.shape[0])) for a in semantics]
    # tag_features = np.concatenate(semantics, axis=0)
    # print(f'\n...shape of tag_features: {tag_features.transpose().shape}\n')
    # tag_feats = {'__header__': 'VIST semantic features', 'feats': tag_features.transpose()}
    # scipy.io.savemat('../../resources/tag_feats.mat', tag_feats)
    # print('\nsaved tag_feats!\n')


def sequences_and_stories(args):
    transform = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    dataset_configs = DatasetParams(args.dataset_config_file)
    dataset_params, vocab_path = dataset_configs.get_params(args.dataset)
    vocab = get_vocab(vocab_path)
    data_loader, _ = get_loader(dataset_configs=dataset_params, vocab=vocab,
                                transform=transform,
                                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                ext_feature_sets=args.features, skip_images=False,
                                multi_decoder=True, _collate_fn=collate_fn_multi_decoder)

    print(f'{len(data_loader)} batches loaded')

    # story_id_2_seq_feature = {}
    story_id_2_sentence1 = {}
    story_id_2_sentence2 = {}
    story_id_2_sentence3 = {}
    story_id_2_sentence4 = {}
    story_id_2_sentence5 = {}
    for batch_idx, (sequences, targets1, targets2, targets3, targets4, targets5, story_ids, feature_sets) in enumerate(
            data_loader):
        cur_batch_size = len(feature_sets)
        for idx in range(cur_batch_size):
            cur_story_id = story_ids[idx]
            # features = torch.Tensor()
            # for jdx in range(5):
            #     features = torch.cat([features, feature_sets[idx][jdx][0].unsqueeze(0)])
            #
            # features = features.reshape((1, -1))
            # story_id_2_seq_feature[cur_story_id] = features.numpy()

            story_id_2_sentence1[cur_story_id] = list(targets1[idx][1].numpy())
            story_id_2_sentence2[cur_story_id] = list(targets2[idx][1].numpy())
            story_id_2_sentence3[cur_story_id] = list(targets3[idx][1].numpy())
            story_id_2_sentence4[cur_story_id] = list(targets4[idx][1].numpy())
            story_id_2_sentence5[cur_story_id] = list(targets5[idx][1].numpy())

        print(f'{batch_idx}/{len(data_loader)} batches complete!')

    print('saving as pkls temporarily...\n')
    with open(f'../../resources/story_id_2_sentence1_{args.datasplit}.pkl', 'wb') as file:
        pickle.dump(story_id_2_sentence1, file)
    file.close()

    with open(f'../../resources/story_id_2_sentence2_{args.datasplit}.pkl', 'wb') as file:
        pickle.dump(story_id_2_sentence2, file)
    file.close()

    with open(f'../../resources/story_id_2_sentence3_{args.datasplit}.pkl', 'wb') as file:
        pickle.dump(story_id_2_sentence3, file)
    file.close()

    with open(f'../../resources/story_id_2_sentence4_{args.datasplit}.pkl', 'wb') as file:
        pickle.dump(story_id_2_sentence4, file)
    file.close()

    with open(f'../../resources/story_id_2_sentence5_{args.datasplit}.pkl', 'wb') as file:
        pickle.dump(story_id_2_sentence5, file)
    file.close()

    # with open(f'../../resources/story_id_2_seq_feature_{args.datasplit}.pkl', 'wb') as file:
    #     pickle.dump(story_id_2_seq_feature, file)
    # file.close()

    print('done!')


def characters_vocab(train, val, test):
    char_freq_train = read_txt_file(train)
    char_freq_val = read_txt_file(val)
    char_freq_test = read_txt_file(test)

    tag_vocab = Vocabulary()

    for char in char_freq_train:
        char = char.replace('\'', '').strip()[1:-1].split(', ')
        tag_vocab.add_word(char[0])
    for char in char_freq_val:
        char = char.replace('\'', '').strip()[1:-1].split(', ')
        tag_vocab.add_word(char[0])
    for char in char_freq_test:
        char = char.replace('\'', '').strip()[1:-1].split(', ')
        tag_vocab.add_word(char[0])

    return tag_vocab, (char_freq_train, char_freq_val, char_freq_test)


def get_co_occurring_pairs(chars, pair_freqs):
    co_occurring = []
    for pair_freq in pair_freqs:
        pair_freq = pair_freq.replace('\'', '')[2:].split('),')[0].split(', ')
        if pair_freq[0] in chars and pair_freq[1] not in chars:
            co_occurring.append(pair_freq[1])
        elif pair_freq[0] not in chars and pair_freq[1] in chars:
            co_occurring.append(pair_freq[0])

    return list(set(co_occurring))


def get_semantics_vector(tag_vocab, tags, freqs, denominator):
    n_tags = len(tag_vocab.word2idx)
    tag_vector = np.zeros(shape=n_tags, dtype=np.float)

    freqs_dict = {}
    for freq in freqs:
        freq = freq.replace('\'', '').strip()[1:-1].split(', ')
        freqs_dict[freq[0]] = int(freq[1])

    for tag in tags:
        try:
            tag_freq = float(freqs_dict[tag])
        except KeyError:
            continue

        tag_idx = tag_vocab.word2idx[tag]
        assert tag_idx < n_tags
        tag_vector[tag_idx] = tag_freq / denominator

    return tag_vector


def create_semantics(args):
    dataset_configs = DatasetParams(args.dataset_config_file)
    dataset_params, vocab_path = dataset_configs.get_params(args.dataset)
    vocab = get_vocab(vocab_path)

    all_stories = [read_pickle_dict('../../resources/story_id_2_story_train.pkl'),
                   read_pickle_dict('../../resources/story_id_2_story_val.pkl'),
                   read_pickle_dict('../../resources/story_id_2_story_test.pkl')]

    tag_vocab, char_freqs = characters_vocab(args.characters_frequency_file_loc + '_train.txt',
                                             args.characters_frequency_file_loc + '_val.txt',
                                             args.characters_frequency_file_loc + '_test.txt')

    char_pair_freqs = (read_txt_file(args.character_pairs_frequency_file_loc + '_train.txt'),
                       read_txt_file(args.character_pairs_frequency_file_loc + '_val.txt'),
                       read_txt_file(args.character_pairs_frequency_file_loc + '_test.txt'))

    story_id_2_chars = (read_json_dict(args.characters_file_loc + '_train.json'),
                        read_json_dict(args.characters_file_loc + '_val.json'),
                        read_json_dict(args.characters_file_loc + '_test.json'))

    story_id_2_semantics = {}
    for split in range(3):
        story_id_2_chars_split = story_id_2_chars[split]
        all_stories_split = all_stories[split]

        for story_id, _ in all_stories_split.items():
            semantics = []
            if story_id in story_id_2_chars_split:
                sent_chars = story_id_2_chars_split[story_id]
                for chars in sent_chars:
                    semantics.extend(chars)
                semantics = list(set(semantics))
                semantics.extend(get_co_occurring_pairs(semantics, char_pair_freqs[split]))
            semantics_vector = get_semantics_vector(tag_vocab, semantics, char_freqs[split], len(vocab.word2idx))
            story_id_2_semantics[story_id] = semantics_vector

    print(f'created {len(story_id_2_semantics)} semantic vectors! saving as pkl!...\n')

    with open('../../resources/story_id_2_semantics.pkl', 'wb') as file:
        pickle.dump(story_id_2_semantics, file)
    file.close()

    print('done!')


def word2vec_it(args):
    model = gensim.models.KeyedVectors.load_word2vec_format(args.path_to_weights, binary=True)

    dataset_configs = DatasetParams(args.dataset_config_file)
    dataset_params, vocab_path = dataset_configs.get_params(args.dataset)
    vocab = get_vocab(vocab_path)

    word_vectors = []
    for idx in range(len(vocab.idx2word)):
        word = vocab.idx2word[idx]
        if word in model:
            word_vector = model[word].reshape((1, 300))
        else:
            word_vector = np.zeros((1, 300), dtype=np.float)

        word_vectors.append(word_vector)

    print(f'collected {len(word_vectors)} word vectors!')
    print('saving as pkl')
    word_vectors = np.concatenate(word_vectors, axis=0)
    print(word_vectors.shape)
    with open('../../resources/word2vec.pkl', mode='wb') as file:
        pickle.dump(word_vectors, file, protocol=2)
    file.close()

    print('\ndone!')


if __name__ == '__main__':
    print('extracting data for SCN')

    parser = argparse.ArgumentParser()

    # sequence features
    parser.add_argument('--dataset', type=str, default='vist:train',
                        help='dataset to use')
    parser.add_argument('--dataset_config_file', type=str,
                        default='../../resources/configs/datasets.local.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--features', type=list, default=[['c_in14_gr_pool5_d_ca3.lmdb']],
                        help='features to use as the initialload input for the '
                             'caption generator, given as comma separated list, '
                             'multiple features are concatenated, '
                             'features ending with .npy are assumed to be '
                             'precalculated features read from the named npy file, '
                             'example: "resnet152,c_in14_gr_pool5_d_ca3.lmdb"')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)

    # semantics
    parser.add_argument('--characters_file_loc', type=str,
                        default='../../resources/characters_analysis/characters',
                        help='location of characters mapping json')
    parser.add_argument('--characters_frequency_file_loc', type=str,
                        default='../../resources/characters_analysis/characters_frequency',
                        help='location of characters frequency txt')
    parser.add_argument('--character_pairs_frequency_file_loc', type=str,
                        default='../../resources/characters_analysis/character_pairs_frequency',
                        help='location of character pairs frequency txt')

    # saving as datalist
    parser.add_argument('--datasplit', type=str, default='train',
                        help='datasplit {train/val/test}')

    # word2vec
    parser.add_argument('--path_to_weights', type=str, default='../../resources/GoogleNews-vectors-negative300.bin',
                        help='word2vec pretrained weights')

    # sequences_and_stories(parser.parse_args())

    # create_semantics(parser.parse_args())

    # print('saving as img_feats matrix and respective split datalist\n...')
    save_it(parser.parse_args())
    print('saving complete!')

    # word2vec_it(parser.parse_args())
