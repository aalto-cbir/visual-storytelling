import json

from album_info import AlbumInfo
from image_info import ImageInfo

# skeleton instructions
album_object = '**object** {padded_album_id}{X}{Y} imageset {padded_album_id}{X}{Y}.txt - - image/list - - {X} {Y} -'
album_class = '**class** {split_detail} {padded_album_id}00'

image_object_jpg = '**object** {padded_image_id} image {padded_image_id}.jpg {zip_name}.zip[{zip_path}/{image_id}.jpg] - image/jpeg - - 0 0 -'
image_object_png = '**object** {padded_image_id} image {padded_image_id}.png {zip_name}.zip[{zip_path}/{image_id}.png] - image/png - - 0 0 -'
image_object_gif = '**object** {padded_image_id} image {padded_image_id}.gif {zip_name}.zip[{zip_path}/{image_id}.gif] - image/gif - - 0 0 -'
image_subobjects = '**subobjects** {padded_album_id}{X}{Y} + '


def prepadd_zero(_id, id_max_len):
    return str(_id).zfill(id_max_len)


def get_data_class(split):
    if 'split' in split:
        return 'train'
    return split


def get_image_object(name):
    if '.jpg' in name:
        return image_object_jpg
    elif '.gif' in name:
        return image_object_gif
    return image_object_png


def get_zip_details(data_class, split):
    if data_class == 'train':
        return data_class + '_' + split, data_class + '/' + split
    return data_class, data_class


def write_to_file(file_path, instructions):
    with open(file_path, 'a+') as file:
        file.write('\n')
        file.write('\n'.join(instructions))


def save_mapping(file_path, mapping):
    with open(file_path, 'w') as file:
        file.write(json.dumps(mapping))


def populate_picsom_file(file_path):
    mapping = {}
    album_info = AlbumInfo()
    image_info = ImageInfo()

    album_to_images = image_info.get_album_images()
    album_to_seqs = album_info.get_album_seqs()

    album_id_max_len = album_info.get_max_id_len()
    image_id_max_len = max(image_info.get_max_id_len(), album_id_max_len + 2)

    album_ids = album_info.get_album_ids()
    for album_id in album_ids:
        instructions = []
        padded_image_ids = []
        padded_album_id = '0' + prepadd_zero(album_id, album_id_max_len)

        instructions.append(album_object.format(padded_album_id=padded_album_id, X=0, Y=0))
        instructions.append(album_object.format(padded_album_id=padded_album_id, X=1, Y=0))
        instructions.append(album_object.format(padded_album_id=padded_album_id, X=2, Y=0))

        image_ids = album_to_images[album_id]

        split_detail, image_name = image_info.get_split_detail(image_ids[0])
        instructions.append(
            album_class.format(split_detail=get_data_class(split_detail), padded_album_id=padded_album_id))

        padded_image_ids.append('1' + prepadd_zero(image_ids[0], image_id_max_len))
        zip_name, zip_path = get_zip_details(get_data_class(split_detail), split_detail)
        mapping[image_ids[0]] = zip_path + '/' + image_name
        instructions.append(get_image_object(image_name).format(padded_image_id=padded_image_ids[0],
                                                                zip_name=zip_name, zip_path=zip_path,
                                                                image_id=image_ids[0]))

        for idx in range(1, len(image_ids)):
            split_detail, image_name = image_info.get_split_detail(image_ids[idx])
            padded_image_ids.append('1' + prepadd_zero(image_ids[idx], image_id_max_len))
            zip_name, zip_path = get_zip_details(get_data_class(split_detail), split_detail)
            mapping[image_ids[idx]] = zip_path + '/' + image_name
            instructions.append(get_image_object(image_name).format(padded_image_id=padded_image_ids[idx],
                                                                    zip_name=zip_name, zip_path=zip_path,
                                                                    image_id=image_ids[idx]))

        album_subobject = image_subobjects.format(padded_album_id=padded_album_id, X=0, Y=0)
        for _id in padded_image_ids:
            album_subobject += _id + ' '
        instructions.append(album_subobject.rstrip())

        album_seqs = album_to_seqs[album_id]
        for _X, sequence in enumerate(album_seqs):
            sequence_subobject = image_subobjects.format(padded_album_id=padded_album_id, X=_X + 1, Y=0)
            for image_id in sequence:
                sequence_subobject += '1' + prepadd_zero(image_id, image_id_max_len) + ' '

            instructions.append(sequence_subobject.rstrip())

        # write_to_file(file_path, instructions)

    print('len of mapping: ', len(mapping))
    save_mapping('../../resources/mapping.txt', mapping)


if __name__ == '__main__':
    print('main')
    populate_picsom_file('../../resources/PicSOM_instructions/PicSOM_instructions.txt')
