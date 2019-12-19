import argparse
import json
import pickle

import nltk
from nltk.corpus import wordnet as wn

wordnet_fails = set()


def get_anns(json_file):
    with open(json_file) as raw_data:
        json_data = json.load(raw_data)
        anns = json_data['annotations']

    return anns


def extract_characters(text):
    words = nltk.tokenize.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    nouns = [pos_tags[idx] for idx in range(len(pos_tags)) if pos_tags[idx][1].startswith('N')]

    output = []
    for noun in nouns:
        synsets = wn.synsets(noun[0])
        if len(synsets) == 0:
            wordnet_fails.add(noun[0])
            continue

        matches = 0
        for synset in synsets:
            person = synset.lowest_common_hypernyms(wn.synset('person.n.01'))
            animal = synset.lowest_common_hypernyms(wn.synset('animal.n.01'))
            if (len(person) > 0 and 'person' in person[0].name()) or (len(animal) > 0 and 'animal' in animal[0].name()):
                matches += 1

        if matches == len(synsets):
            output.append(noun[0])

    return output


def save(file_name, mapping):
    with open(file_name, mode='wb') as file:
        pickle.dump(mapping, file)

    print(f'saved to {file_name}')


def main(args):
    anns = get_anns(args.read_from)
    print(f'read {len(anns)} annotations')

    mapping = {}
    for idx in range(0, len(anns), 5):
        story_id = anns[idx][0]['story_id']
        characters = []
        for jdx in range(5):
            sentence_story = anns[idx + jdx][0]['text']
            characters.append(extract_characters(sentence_story))

        mapping[story_id] = characters
        if idx % 50 == 0:
            print(f'processed {idx + 5} annotations')

    print(f'wordnet fails for:\n {wordnet_fails}')
    save(args.save_to, mapping)


if __name__ == '__main__':
    print('extracting characters from VIST')

    parser = argparse.ArgumentParser()
    parser.add_argument('--read_from', type=str, default='../../resources/sis/val.story-in-sequence.json',
                        help='annotations file location')
    parser.add_argument('--save_to', type=str, default='../../resources/sis/characters_train.pkl',
                        help='file location to store extracted characters with respective mapping')

    main(parser.parse_args())
