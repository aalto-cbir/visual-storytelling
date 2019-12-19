import argparse
import json
import pickle
from collections import Counter

import nltk

from sources.general.vocabulary import Vocabulary


def build_vocab(vocab, json_file, threshold):
    """
    :param vocab: vocabulary instance
    :param json_file: VIST annotations file
    :param threshold: minimum required frequency of each word
    :return: vocabulary wrapper
    """
    counter = Counter()
    with open(json_file) as raw_data:
        json_data = json.load(raw_data)
        annotations = json_data['annotations']

    for idx, annotation in enumerate(annotations):
        caption = str(annotation[0]['text'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (idx + 1) % 1000 == 0:
            print(f'[{idx + 1}/{len(annotations)}] annotations tokenized')

    # if the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # create a vocab wrapper and add some special tokens.
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # add the words to the vocabulary.
    for idx, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = Vocabulary()
    for caption_path in args.caption_path:
        vocab = build_vocab(vocab, json_file=caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print(f'total vocabulary size: {len(vocab)}')
    print(f'saved the vocabulary wrapper to "{vocab_path}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=list,
                        default=['../../resources/sis/train.story-in-sequence.json',
                                 '../../resources/sis/val.story-in-sequence.json',
                                 '../../resources/sis/test.story-in-sequence.json',],
                        help='path for train_validate annotation file')
    parser.add_argument('--vocab_path', type=str, default='../../resources/vocab_baseline.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=3,
                        help='minimum word count threshold')
    arguments = parser.parse_args()
    main(arguments)
