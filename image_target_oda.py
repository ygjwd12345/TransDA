import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from loss import KnowledgeDistillationLoss


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False, threshold=0.1):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        all_output = nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)

        from sklearn.cluster import KMeans
        kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1, 1))
        labels = kmeans.predict(ent.reshape(-1, 1))

        idx = np.where(labels == 1)[0]
        iidx = 0
        if ent[idx].mean() > ent.mean():
            iidx = 1
        predict[np.where(labels == iidx)[0]] = args.class_num

        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int), :]

        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        unknown_acc = acc[-1:].item()
        return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    else:
        return accuracy * 100, mean_ent


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    else:
        netF = network.ViT().cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False
    ### add teacher module
    if args.net[0:3] == 'res':
        netF_t = network.ResBase(res_name=args.net, se=args.se, nl=args.nl).cuda()
    elif args.net[0:3] == 'vgg':
        netF_t = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        netF_t = network.ViT().cuda()
    netB_t = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    ### initial from student
    netF_t.load_state_dict(netF.state_dict())
    netB_t.load_state_dict(netB.state_dict())

    ### remove grad
    for k, v in netF_t.named_parameters():
        v.requires_grad = False
    for k, v in netB_t.named_parameters():
        v.requires_grad = False
    param_group = []

    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    tt = 0
    iter_num = 0
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0:
            netF.eval()
            netB.eval()
            mem_label, ENT_THRESHOLD,dd = obtain_label(dset_loaders['test'], netF_t, netB_t, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        pred = mem_label[tar_idx]
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        softmax_out = nn.Softmax(dim=1)(outputs_test)
        outputs_test_known = outputs_test[pred < args.class_num, :]
        pred = pred[pred < args.class_num]
        pred_soft = dd[tar_idx]
        pred_soft = torch.tensor(pred_soft).cuda()
        if len(pred) == 0:
            print(tt)
            del features_test
            del outputs_test
            tt += 1
            continue

        if args.cls_par > 0:
            classifier_loss = nn.CrossEntropyLoss()(outputs_test_known, pred)
            classifier_loss *= args.cls_par
        if args.kd:
            kd_loss = KnowledgeDistillationLoss()(outputs_test, pred_soft)
            classifier_loss += kd_loss
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out_known = nn.Softmax(dim=1)(outputs_test_known)
            entropy_loss = torch.mean(loss.Entropy(softmax_out_known))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            classifier_loss += entropy_loss * args.ent_par
        # EMA update for the teacher
        with torch.no_grad():
            m = 0.001  # momentum parameter
            for param_q, param_k in zip(netF.parameters(), netF_t.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(netB.parameters(), netB_t.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            acc_os1, acc_os2, acc_unknown = cal_acc(dset_loaders['test'], netF, netB, netC, True, ENT_THRESHOLD)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.name, iter_num,
                                                                                            max_iter, acc_os2, acc_os1,
                                                                                            acc_unknown)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB, netC


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()

    from sklearn.cluster import KMeans
    kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1, 1))
    labels = kmeans.predict(ent.reshape(-1, 1))

    idx = np.where(labels == 1)[0]
    iidx = 0
    if ent[idx].mean() > ent.mean():
        iidx = 1
    known_idx = np.where(kmeans.labels_ != iidx)[0]

    all_fea = all_fea[known_idx, :]
    all_output = all_output[known_idx, :]
    predict = predict[known_idx]
    all_label_idx = all_label[known_idx]
    ENT_THRESHOLD = (kmeans.cluster_centers_).mean()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    guess_label = args.class_num * np.ones(len(all_label), )
    guess_label[known_idx] = pred_label
    D =np.ones((len(all_label),dd.shape[1] ))
    D[known_idx] = dd
    acc = np.sum(guess_label == all_label.float().numpy()) / len(all_label_idx)
    log_str = 'Threshold = {:.2f}, Accuracy = {:.2f}% -> {:.2f}%'.format(ENT_THRESHOLD, accuracy * 100, acc * 100)

    return guess_label.astype('int'), ENT_THRESHOLD,D


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['office-home'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='vit', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--kd', type=bool, default=False)

    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='oda', choices=['oda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = args.t_dset_path

        if args.dset == 'office-home':
            if args.da == 'oda':
                args.class_num = 25
                args.src_classes = [i for i in range(25)]
                args.tar_classes = [i for i in range(65)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)