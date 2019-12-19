import json
import os


class ImageInfo:
    def __init__(self):
        print('this class provides methods for understanding VIST dataset images')
        self.json_files = ['../../resources/sis/train_validate.story-in-sequence.json',
                           '../../resources/sis/val.story-in-sequence.json',
                           '../../resources/sis/test.story-in-sequence.json']
        self.images_data = []
        self.annotations_data = []
        self.split_details = {}
        self.split_details_path = '../../resources/split_details/'
        self.__read_data()

    def __read_data(self):
        for json_file in self.json_files:
            with open(json_file) as raw_data:
                json_data = json.load(raw_data)
                self.images_data.append(json_data['images'])
                self.annotations_data.append(json_data['annotations'])

        for file_name in os.listdir(self.split_details_path):
            with open(os.path.join(self.split_details_path, file_name)) as file_path:
                self.split_details[file_name.split('.')[0]] = [line.strip() for line in file_path.readlines()]

    def get_image_count(self):
        image_count = 0
        for image_data in self.images_data:
            image_count += len(image_data)

        return image_count

    def get_max_id_len(self):
        image_id_details = {}
        for image_data in self.images_data:
            for image in image_data:
                if len(image['id']) not in image_id_details:
                    image_id_details[len(image['id'])] = 1
                else:
                    image_id_details[len(image['id'])] += 1

        # pprint(image_id_details)
        return sorted(image_id_details.keys(), reverse=True)[0]

    def get_album_images(self):
        album_to_images = {}
        for image_data in self.images_data:
            for image in image_data:
                album_id = image['album_id']
                image_id = image['id']
                if album_id not in album_to_images:
                    album_to_images[album_id] = list()
                album_to_images[album_id].append(image_id)

        return album_to_images

    def get_split_detail(self, image_id):
        for split_id, split_detail in self.split_details.items():
            for image_name in split_detail:
                if image_id == image_name.split('.')[0]:
                    return split_id, image_name

        return KeyError('Not found in any split')


if __name__ == '__main__':
    print('main')
    image_info = ImageInfo()
    print(image_info.get_split_detail('102718619'))
