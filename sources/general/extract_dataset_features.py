import argparse
import os
import sys

import lmdb
import torch
from torchvision import transforms

from sources.general.data_loader import get_loader, DatasetParams
from sources.models.common import FeatureExtractor

try:
    from tqdm import tqdm
except ImportError as e:
    print('WARNING: tqdm module not found. Install it if you want a fancy progress bar :-)')


    def tqdm(x):
        return x

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Image preprocessing
    global features
    if args.feature_type == 'plain':
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    elif args.feature_type == 'avg' or args.feature_type == 'max':
        # Try with no normalization
        # Try with subtracting 0.5 from all values
        # See example here: https://pytorch.org/docs/stable/torchvision/transforms.html

        if args.normalize == 'default':
            transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                # 10-crop implementation as described in PyTorch documentation:
                transforms.TenCrop((args.crop_size, args.crop_size)),
                # Apply next two transforms to each crop in turn and then stack them
                # to a single tensor:
                transforms.Lambda(lambda crops: torch.stack([
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))(transforms.ToTensor()(crop))
                    for crop in crops]))])
        elif args.normalize == 'skip':
            transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.TenCrop((args.crop_size, args.crop_size)),
                transforms.Lambda(lambda crops: torch.stack([
                    transforms.ToTensor()(crop)
                    for crop in crops]))])
        elif args.normalize == 'subtract_half':
            transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.TenCrop((args.crop_size, args.crop_size)),
                transforms.Lambda(lambda crops: torch.stack([
                    transforms.ToTensor()(crop)
                    for crop in crops]) - 0.5)])
        else:
            print("Invalid normalization parameter")
            sys.exit(1)

    else:
        print(f"Invalid feature type specified {args.feature_type}")
        sys.exit(1)

    print("Creating features of type: {}".format(args.feature_type))

    # Get dataset parameters and vocabulary wrapper:
    dataset_configs = DatasetParams(args.dataset_config_file)
    dataset_params, _ = dataset_configs.get_params(args.dataset, vocab_path=None)

    # We ask it to iterate over images instead of all (image, caption) pairs
    data_loader, _ = get_loader(dataset_params, vocab=None, transform=transform,
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers,
                                ext_feature_sets=None,
                                skip_images=False)
    extractor = FeatureExtractor(args.extractor, True).to(device)

    # To open an lmdb handle and prepare it for the right size
    # it needs to fit the total number of elements in the dataset
    # so we set a map_size to a largish value here:
    map_size = 1e10

    if args.output_file:
        file_name = args.output_file
    else:
        file_name = '{}-{}-{}-normalize-{}.lmdb'.format(args.dataset, args.extractor,
                                                        args.feature_type, args.normalize)

    os.makedirs(args.output_dir, exist_ok=True)

    lmdb_path = os.path.join(args.output_dir, file_name)

    print("Preparing to store extracted features to {}...".format(lmdb_path))
    env = lmdb.open(lmdb_path, map_size=map_size)

    print("Starting to extract features from dataset {} using {}...".
          format(args.dataset, args.extractor))
    show_progress = sys.stderr.isatty()
    for i, (images, _, _,
            image_ids, _) in enumerate(tqdm(data_loader)):

        images = images.to(device)

        # If we are dealing with cropped images, image dimensions are is: bs, ncrops, c, h, w
        if images.dim() == 5:
            bs, ncrops, c, h, w = images.size()
            # fuse batch size and ncrops:
            raw_features = extractor(images.view(-1, c, h, w))

            if args.feature_type == 'avg':
                # Average over crops:
                features = raw_features.view(bs, ncrops, -1).mean(1).data.cpu().numpy()
            elif args.feature_type == 'max':
                # Max over crops:
                features = raw_features.view(bs, ncrops, -1).max(1)[0].data.cpu().numpy()
        # Otherwise our image dimensions are bs, c, h, w
        else:
            features = extractor(images).data.cpu().numpy()

        # Write to LMDB object:
        with env.begin(write=True) as txn:
            for j, image_id in enumerate(image_ids):
                txn.put(str(image_id).encode('ascii'), features[j])

        # Print log info
        if not show_progress and ((i + 1) % args.log_step == 0):
            print('Batch [{}/{}]'.format(i + 1, len(data_loader)))
            sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='vist:train_validate',
                        help='dataset that defines images for which features are needed')
    parser.add_argument('--dataset_config_file', type=str,
                        default='../../resources/configs/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--feature_type', type=str, default='avg',
                        help='type of a feature output - can be:'
                             'plain - use the input image as is - no cropping or pooling\n'
                             'following two feature types use transform.TenCrop - each image is '
                             'cropped from corners and center, each crop is horizontally flipped:\n'
                             'avg - elementwise average of features obtained from cropped inputs'
                             'max - elementwise maximum of features obtained from cropped inputs')
    parser.add_argument('--normalize', type=str, default='default',
                        help='image normalization to apply\n'
                             'default: applies default PyTorch normalization parameters\n'
                             'skip: applies no normalization at all\n'
                             'substract_half: subtracts 0.5 from each pixel value')
    parser.add_argument('--image_size', type=int, default=256,
                        help='resize input images to this size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='crop size used by "avg" and "max" feature types')
    parser.add_argument('--num_crops', type=int, default=12,
                        help='number of crops to perform for avg and max feature types')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='features/',
                        help='directory for saving image features')
    parser.add_argument('--output_file', type=str, default='',
                        help='file for saving features, if no name specified it '
                             'defaults to "dataset_name-extractor.lmdb"')
    parser.add_argument('--extractor', type=str, default='resnet152',
                        help='name of the extractor, ex: alexnet, resnet152, densenet201')
    parser.add_argument('--log_step', type=int, default=10,
                        help='How often do we want to log output')

    arguments = parser.parse_args()
    main(args=arguments)
