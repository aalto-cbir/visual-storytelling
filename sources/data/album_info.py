import json

import pandas as pd


class AlbumInfo:
    def __init__(self):
        print('this class provides methods for understanding VIST dataset albums')
        self.json_files = ['../../resources/sis/train_validate.story-in-sequence.json',
                           '../../resources/sis/val.story-in-sequence.json',
                           '../../resources/sis/test.story-in-sequence.json']
        self.albums_data = []
        self.annotations_data = []
        self.__read_data()

    def __read_data(self):
        for json_file in self.json_files:
            with open(json_file) as raw_data:
                json_data = json.load(raw_data)
                self.albums_data.append(json_data['albums'])
                self.annotations_data.append(json_data['annotations'])

    def get_album_count(self):
        album_count = []
        for album_data in self.albums_data:
            album_count.append(len(album_data))

        return album_count

    def get_image_count(self):
        image_count = []
        for album_data in self.albums_data:
            _image_count = 0
            for album in album_data:
                _image_count += int(album['photos'])

            image_count.append(_image_count)

        return image_count

    def get_album_ids(self):
        album_ids = []
        for album_data in self.albums_data:
            for album in album_data:
                album_ids.append(album['id'])

        return album_ids

    def get_max_id_len(self):
        album_id_details = {}
        for album_data in self.albums_data:
            for album in album_data:
                if len(album['id']) not in album_id_details:
                    album_id_details[len(album['id'])] = 1
                else:
                    album_id_details[len(album['id'])] += 1

        album_id_details = pd.DataFrame(list(album_id_details.items()),
                                        columns=['album_id_len', 'num_of_albums'])

        # pprint(album_id_details)
        return album_id_details['album_id_len'].max()

    def get_album_seqs(self):
        album_to_stories = {}
        stories_to_sequences = {}
        for annotation_data in self.annotations_data:
            for annotation in annotation_data:
                if annotation[0]['story_id'] not in stories_to_sequences:
                    stories_to_sequences[annotation[0]['story_id']] = list()
                stories_to_sequences[annotation[0]['story_id']].append(annotation[0]['photo_flickr_id'])
                if annotation[0]['album_id'] not in album_to_stories:
                    album_to_stories[annotation[0]['album_id']] = set()
                album_to_stories[annotation[0]['album_id']].add(annotation[0]['story_id'])

        album_to_sequences = {}
        for album_id, stories in album_to_stories.items():
            sequences = set()
            for story_id in stories:
                sequences.add(tuple(stories_to_sequences[story_id]))

            album_to_sequences[album_id] = sequences

        # pprint(album_to_sequences['616890'])
        return album_to_sequences


if __name__ == '__main__':
    print('main')
    album_info = AlbumInfo()
    # print(f'total number of albums in VIST: {sum(album_info.get_album_count())}')
    # print(f'total number of images in VIST: {sum(album_info.get_image_count())}')
    # album_info.get_album_id_details()
    # album_info.get_album_seqs()
