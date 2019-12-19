import scipy.io
import torch

import models
import opts
from dataset import VISTDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_and_save(opt):

    dataset = VISTDataset(opt)
    opt.vocab_size = dataset.get_vocab_size()
    opt.seq_length = dataset.get_story_length()

    # set up model
    model = models.setup(opt)
    model.to(device)

    model.eval()

    data = scipy.io.loadmat('./resnet_feats.mat')
    img_feats = data['feats']
    print('shape of img_feats: ', img_feats.shape)
    
    contexts = torch.Tensor([])
    num_imgs = img_feats.shape[2]
    for idx in range(num_imgs):
        vis_fea = img_feats[:, :, idx]
        vis_fea = torch.from_numpy(vis_fea).unsqueeze(0).to(device)
        vis_context = torch.flatten(model(vis_fea, [])).unsqueeze(1)
        contexts = torch.cat((contexts, vis_context), 1)
        if (idx + 1) % 1000 == 0:
           print(idx + 1, ' sequences processed!')

    resnet_feats = {'__header__': 'VIST image sequences contexts', 'feats': contexts.detach().numpy()}
    scipy.io.savemat('./resnet_feats_vist_context.mat', resnet_feats)
    print('\nsaved resnet_feats contexts!\n')
    print('shape of contexts: ', contexts.shape)


if __name__ == "__main__":
    opts = opts.parse_opt()

    extract_and_save(opts)

