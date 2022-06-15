import os
import random
import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datasets.retinaOCT import load_data_OCT
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from utils.gen_mask import gen_mask
from models.unet_multi import UNet
from losses.gms_loss import MSGMS_Loss,MSGMS_Score
from losses.ssim_loss import SSIM_Loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def main():
    parser = argparse.ArgumentParser(description='SSMAD')
    parser.add_argument('--data_path', type=str, default='/mnt/cache/huangchaoqin/OCT2017')
    parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--grayscale', action='store_true', help='color or grayscale input image')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--belta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of Adam')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='decay of Adam')
    parser.add_argument('--seed', type=int, default=None, help='manual seed')
    parser.add_argument('--k_value', type=int, nargs='+', default=[2,4,8])
    args = parser.parse_args()

    args.input_channel = 1 if args.grayscale else 3

    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    args.prefix = time_file_str()
    args.save_dir = './save_checkpoints'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'training_log_{}.txt'.format(args.prefix)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    # load model and dataset
    model = UNet(up_mode='upsample').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader, test_loader = load_data_OCT(args.data_path, args.batch_size)

    # start training
    print("Start Training:")
    start_time = time.time()
    epoch_time = AverageMeter()
    img_roc_auc_old = 0.0
    per_pixel_rocauc_old = 0.0
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)
        #train(args, model, epoch, train_loader, optimizer, log)

        scores, gt_list = val(args, model, test_loader)
        scores = np.asarray(scores)
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

        # calculate image-level AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        print('image AUC: %.3f' % (img_roc_auc))
        plt.plot(fpr, tpr, label='img_ROCAUC: %.3f' % (img_roc_auc))
        plt.legend(loc="lower right")
        if img_roc_auc > img_roc_auc_old:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_name = os.path.join(args.save_dir, '{}_{}_{}_model.pt'.format(str(epoch), str(img_roc_auc),args.prefix))
            torch.save(state, save_name)
            img_roc_auc_old = img_roc_auc
        print('max AUC:',img_roc_auc_old)
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    log.close()

def train(args, model, epoch, train_loader, optimizer, log):
    model.train()
    l2_losses = AverageMeter()
    gms_losses = AverageMeter()
    ssim_losses = AverageMeter()
    mask_losses = AverageMeter()
    ssim = SSIM_Loss()
    mse = nn.MSELoss(reduction='mean')
    msgms = MSGMS_Loss()
    for batchdata in tqdm(train_loader):
        data = batchdata[0]
        optimizer.zero_grad()
        data = data.to(device)
        k_value = random.sample(args.k_value, 1)
        Ms_generator = gen_mask(k_value, 3, args.img_size)
        Ms = next(Ms_generator)
        mask_new = torch.tensor(np.array(Ms)).to(device).expand(data.shape[0], -1, -1, -1).permute(1, 0, 2, 3)
        inputs = [data * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms]

        outputs = []
        pmasks = []
        for i in range(len(inputs)):
            my_output, my_mask = model(torch.cat((inputs[i],mask_new[i].unsqueeze(1)),1))
            outputs.append(my_output)
            pmasks.append(my_mask)
        output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, Ms))
        mask_loss = 0
        for i in range(len(outputs)):
            mask_loss += mse(pmasks[i].squeeze(dim=1), mask_new[i].float())

        l2_loss = mse(data, output)
        gms_loss = msgms(data, output)
        ssim_loss = ssim(data, output)
        loss = args.gamma * l2_loss + args.alpha * gms_loss + args.belta * ssim_loss + mask_loss

        l2_losses.update(l2_loss.item(), data.size(0))
        gms_losses.update(gms_loss.item(), data.size(0))
        ssim_losses.update(ssim_loss.item(), data.size(0))
        mask_losses.update(mask_loss.item(), data.size(0))

        loss.backward()
        optimizer.step()

    print_log(('Train Epoch: {} MSE_Loss: {:.6f} GMS_Loss: {:.6f} SSIM_Loss: {:.6f} MASK_Loss: {:.6f}'.format(
        epoch, l2_losses.avg, gms_losses.avg, ssim_losses.avg, mask_losses.avg)), log)

def val(args, model, test_loader):
    model.eval()
    scores = []
    gt_list = []
    msgms_score = MSGMS_Score()
    retinal_good_dict = {'NORMAL': 3, 'OUT':2}
    target_class = retinal_good_dict['NORMAL']
    for databatch in tqdm(test_loader):
        data, label = databatch
        gt_list.extend(label.cpu().numpy())
        score = 0
        with torch.no_grad():
            data = data.to(device)
            for k in args.k_value:
                img_size = data.size(-1)
                N = img_size // k
                Ms_generator = gen_mask([k], 3, img_size)
                Ms = next(Ms_generator)
                mask_new = torch.tensor(np.array(Ms)).to(device).expand(data.shape[0], -1, -1, -1).permute(1, 0, 2, 3)
                inputs = [data * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms]
                outputs = []
                for i in range(len(inputs)):
                    my_output, my_mask = model(torch.cat((inputs[i], mask_new[i].unsqueeze(1)), 1))
                    outputs.append(my_output)
                output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, Ms))
                score += msgms_score(data, output) / (N**2)
        score = score.squeeze().cpu().numpy()
        for i in range(score.shape[0]):
            score[i] = gaussian_filter(score[i], sigma=7)
        scores.extend(score)

    gt_list = np.asarray(gt_list)
    indx1 = gt_list == target_class
    indx2 = gt_list != target_class
    gt_list[indx1] = 0
    gt_list[indx2] = 1
    return scores, gt_list

def adjust_learning_rate(args, optimizer, epoch):
    if epoch == 15:
        lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 30:
        lr = args.lr * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 45:
        lr = args.lr * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch == 60:
        lr = args.lr * 0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == '__main__':
    main()