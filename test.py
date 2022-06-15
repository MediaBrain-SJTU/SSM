import os
import argparse
import matplotlib
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
from models.unet_multi import UNet
from utils.gen_mask import gen_mask
from losses.gms_loss import MSGMS_Score
from losses.ssim_loss import SSIM_Loss
from datasets.retinaOCT import load_data_OCT

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
plt.switch_backend('agg')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--data_path', type=str, default='/mnt/cache/huangchaoqin/OCT2017')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='/mnt/lustre/huangchaoqin/SSM/retinal_mask_1e-2/24_0.9773922546274163_2021-11-16-8435_model.pt')
    parser.add_argument("--grayscale", action='store_true', help='color or grayscale input image')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--refine_num', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.96)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--k_value', type=int, nargs='+', default=[2,4,8])
    args = parser.parse_args()
    args.save_dir = './save_figures/'
    print("Testing on Retinal OCT")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load model and dataset
    print("Loading model and dataset")
    args.input_channel = 1 if args.grayscale else 3
    model = UNet(up_mode='upsample').to(device)
    checkpoint = torch.load(args.checkpoint_dir)
    model.load_state_dict(checkpoint['model'])
    _, test_loader = load_data_OCT(args.data_path, args.batch_size)

    print("Start Testing:")
    scores, scores_ssim, scores_mse, gt_list = test(args, model, test_loader)
    scores = np.asarray(scores)
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    scores_ssim = np.asarray(scores_ssim)
    max_anomaly_score = scores_ssim.max()
    min_anomaly_score = scores_ssim.min()
    scores_ssim = (scores_ssim - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    scores_mse = np.asarray(scores_mse)
    max_anomaly_score = scores_mse.max()
    min_anomaly_score = scores_mse.min()
    scores_mse = (scores_mse - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print('image AUC GMS: %.3f' % (img_roc_auc))

    img_scores_ssim = scores_ssim.reshape(scores_ssim.shape[0], -1).max(axis=1)
    fpr, tpr, _ = roc_curve(gt_list, img_scores_ssim)
    img_roc_auc = roc_auc_score(gt_list, img_scores_ssim)
    print('image AUC SSIM: %.3f' % (img_roc_auc))

    img_scores_mse = scores_mse.reshape(scores_mse.shape[0], -1).max(axis=1)
    fpr, tpr, _ = roc_curve(gt_list, img_scores_mse)
    img_roc_auc = roc_auc_score(gt_list, img_scores_mse)
    print('image AUC MSE: %.3f' % (img_roc_auc))

    final = img_scores + img_scores_mse
    max_anomaly_score = final.max()
    min_anomaly_score = final.min()
    final_scores = (final - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    fpr, tpr, _ = roc_curve(gt_list, final_scores)
    img_roc_auc = roc_auc_score(gt_list, final_scores)
    print('image AUC final: %.3f' % (img_roc_auc))
    plt.plot(fpr, tpr, label='IMG_AUC: %.3f' % (img_roc_auc))
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(args.save_dir, 'roc_curve.png'), dpi=100)


def test(args, model, test_loader):
    model.eval()
    scores = []
    scores_mse = []
    scores_ssim = []
    gt_list = []
    msgms_score = MSGMS_Score()
    ssim = SSIM_Loss()
    mse = nn.MSELoss(reduction='mean')
    retinal_good_dict = {'NORMAL': 3, 'OUT': 2}
    target_class = retinal_good_dict['NORMAL']
    for data_batch in tqdm(test_loader):
        data, label = data_batch
        data = data.to(device)
        gt_list.extend(label.cpu().numpy())
        for num in range(args.refine_num):
            score = 0
            score_mse = 0
            score_ssim = 0
            with torch.no_grad():
                for k in args.k_value:
                    img_size = data.size(-1)
                    N = img_size // k
                    setup_seed(args.seed+num)
                    Ms_generator = gen_mask([k], 3, img_size)
                    Ms = next(Ms_generator)
                    if num > 0:
                        for mask in Ms:
                            for i in range(16):
                                for j in range(16):
                                    if heat_map[i*8:(i+1)*8,j*8:(j+1)*8].mean() > args.ratio:
                                        mask[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = 1
                    mask_new = torch.tensor(np.array(Ms)).to(device).expand(data.shape[0], -1, -1, -1).permute(1, 0, 2, 3)
                    inputs = [data * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms]
                    outputs = []
                    for i in range(len(inputs)):
                        my_output, my_mask = model(torch.cat((inputs[i], mask_new[i].unsqueeze(1)), 1))
                        outputs.append(my_output)
                    output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, Ms))
                    if num < args.refine_num-1:
                        score += msgms_score(data, output) / (N ** 2)
                    else:
                        output_new = torch.where(output==0,data,output)
                        score += msgms_score(data, output_new) / (N ** 2)
                        score_mse += mse(data, output_new)
                        score_ssim += ssim(data, output_new)
                score = score.squeeze().cpu().numpy()
                for i in range(score.shape[0]):
                    score[i] = gaussian_filter(score[i], sigma=7)
                if num < args.refine_num-1:
                    heat_map = score * 255
                    heat_map[heat_map < -args.threshold] = -args.threshold
                    heat_map[heat_map > -args.threshold] = 0
                    heat_map[heat_map == -args.threshold] = 1
                else:
                    score_mse = score_mse.squeeze().cpu().numpy()
                    score_ssim = score_ssim.squeeze().cpu().numpy()
                    scores.extend([score])
                    scores_ssim.append(score_ssim)
                    scores_mse.append(score_mse)
    gt_list = np.asarray(gt_list)
    indx1 = gt_list == target_class
    indx2 = gt_list != target_class
    gt_list[indx1] = 0
    gt_list[indx2] = 1
    return scores, scores_ssim, scores_mse, gt_list

if __name__ == '__main__':
    main()