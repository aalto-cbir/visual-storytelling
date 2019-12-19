import json
import os
from multiprocessing import Pool
from shutil import copy2


def analyze_images(in_file, out_file):
    with open(in_file) as file:
        data = json.load(file)
    file.close()

    images = len(data['images'])
    print(f'images: {images}')
    anns = data['annotations']
    images_with_stories = set()
    for ann in anns:
        images_with_stories.add(ann[0]['photo_flickr_id'])
    print(f'images with stories: {len(images_with_stories)}')

    print(f'number of stories: {len(anns) / 5}')

    with open(out_file, mode='w') as file:
        file.writelines('\n'.join(images_with_stories))
    file.close()


source = '../../resources/images/train_full/resized/'
destination = '../../resources/images/filtered_train/'


def move_image(image):
    image = image.strip()
    # print(source + image + '.jpg')
    if os.path.exists(source + image + '.jpg'):
        copy2(source + image + '.jpg', destination)
    elif os.path.exists(source + image + '.png'):
        copy2(source + image + '.png', destination)
    elif os.path.exists(source + image + '.gif'):
        copy2(source + image + '.gif', destination)
    else:
        print(f'image {image} is of some other format')


if __name__ == '__main__':
    analyze_images('../../resources/sis/train_validate.story-in-sequence.json', '../../resources/data/train_images.txt')
    analyze_images('../../resources/sis/val.story-in-sequence.json', '../../resources/data/val_images.txt')
    analyze_images('../../resources/sis/test.story-in-sequence.json', '../../resources/data/test_images.txt')

    # change global source and destination...
    with open('../../resources/data/train_images.txt') as fp:
        image_names = fp.readlines()
    pool = Pool()
    pool.map(move_image, image_names)
