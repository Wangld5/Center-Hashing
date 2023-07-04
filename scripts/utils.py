from tkinter.tix import Tree
from scripts.head import *
import torch
import torch.nn.functional as F



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_config():
    config = {
        # "remarks": "OurLossWithPair",
        "seed": 60,
        "m": 16,
        "alpha": 1,
        "beta": 1.0,
        "beta2": 0.01,
        "mome": 0.9,
        "epoch_change": 9,
        "sigma": 1.0,
        "gamma": 20.0,
        "lambda": 0.0001,
        "mu": 1,
        "nu": 1,
        "eta": 55,
        "dcc_iter": 10,
        "optimizer": {
            "type": optim.RMSprop,
            # "epoch_lr_decrease": 30,
            "optim_param": {
                "lr": 1e-5,
                "weight_decay": 1e-5,
                # "momentum": 0.9
                # "betas": (0.9, 0.999)

            },
        },
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": MoCo,
        "dataset": "car_ims",
        "without_BN": False,
        "epoch": 1000,
        "test_map": 3,
        "stop_iter": 10,
        # "save_path": "./results/imagenet/OurLossWithPair",
        # "center_path": "./centerswithoutVar/CSQ_init_True_100_64.npy"
        # "save_center": "./results/imagenet/OurLossWithPair/Ours_64.npy",
        "device": torch.device('cuda'),
        "n_gpu": torch.cuda.device_count(),
        "bit_list": [16],
        # "info": "[OurLossWithPair]",
        # "loss_way": "OurLossWithPair",
        "max_norm": 5.0,
        "T": 1e-3,
        "resnet_url": '../../Pretrain/resnet50-19c8e357.pth',
        "url": '../../Pretrain/moco_v2_800ep_pretrain.pth',
        "txt_path": '../../data/glove/glove.6B.50d.txt',
        "label_size": 100,
        "update_center": False,
    }
    config = config_dataset(config)
    return config


draw_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def pretrained_hash(config, bit, train_loader):
    pretrain_model = models.__dict__['resnet50']()
    for name, param in pretrain_model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    pretrain_model.fc.weight.data.normal_(mean=0.0, std=0.01)
    pretrain_model.fc.bias.data.zero_()
    checkpoint = torch.load(config['url'])
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            state_dict[k[len('module.encoder_q.'):]] = state_dict[k]
        del state_dict[k]
    pretrain_model.load_state_dict(state_dict, strict=False)
    model = Binary_hash(bit, pretrain_model.cuda()).cuda()
    epoch_i = 0
    with torch.no_grad():
        for image, label, ind in tqdm(train_loader):
            image = image.cuda()
            label = label.cuda()
            origin_binary = model(image)
            if epoch_i == 0:
                train_origin_binary = origin_binary
                train_origin_label = label
                epoch_i += 1
            else:
                train_origin_binary = torch.cat((train_origin_binary, origin_binary), dim=0)
                train_origin_label = torch.cat((train_origin_label, label), dim=0)
    return train_origin_binary, train_origin_label


def compute_result(dataloader, net, device, T, label_vector):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device), T, label_vector)[0]).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    # pdb.set_trace()
    num_query = queryL.shape[0]
    topkmap = 0
    class_map = []
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
        class_map.append(topkmap_)
    topkmap = topkmap / num_query
    return topkmap, class_map


def calc_sim(sample_label, database_label):
    S = (sample_label @ database_label.t() > 0).float().cuda()
    # soft constraint
    r = S.sum() / (1 - S).sum()
    S = S * (1 + r) - r
    return S


# 第一轮的pair的权重是1，因为还没有让模型预测出第一轮的结果。
# def class_diff(all_map: list, queryL: torch.Tensor):
#     """
#     :param all_map: map for all query sample, list with size queryL.shape[0]
#     :param queryL: all query label with size queryL.shape[0] x class_num
#     :return: difficulty for all class: dictionary
#     """
#     # pdb.set_trace()
#     q_class = torch.argmax(queryL, dim=1).numpy()  # queryL.shape[0]
#     class_map = {}
#     for i in range(q_class.shape[0]):
#         if q_class[i] not in class_map.keys():
#             class_map[q_class[i]] = [all_map[i]]
#         else:
#             class_map[q_class[i]].append(all_map[i])
#     class_weight = []
#     for i in range(queryL.shape[1]):
#         bias = np.var(class_map[i]) - 1
#         tao = np.exp(bias) / (1 + np.exp(bias))
#         class_weight.append((1 - np.mean(class_map[i])) ** tao)
#     weight = torch.tensor(class_weight)
#     return weight


def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
    n_query = qF.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    Rank = np.argsort(CalcHammingDist(qF, rF))
    P, R = [], []
    for k in tqdm(draw_range):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all
        P.append(np.mean(p))
        R.append(np.mean(r))
    return P, R


def get_precision_recall_by_Hamming_Radius(database_data, database_label, query_data, query_label, radius=2):
    # 生成哈希码
    query_output = query_data
    database_output = database_data
    # 获取哈希码长度
    bit_n = query_output.shape[1]
    # 计算汉明距离并排序，按行排序，即第i个哈希码和其他哈希码的距离的从小到大排序
    ips = np.dot(query_output, database_output.T)
    ips = (bit_n - ips) / 2
    ids = np.argsort(ips, 1)

    precX = []
    recX = []
    mAPX = []
    query_labels = query_label
    database_labels = database_label

    for i in range(ips.shape[0]):
        label = query_labels[i, :]
        label[label == 0] = -1
        idx = np.reshape(np.argwhere(ips[i, :] <= radius), (-1)) # 计算所有和第i个哈希码匹配的哈希码的数量
        all_num = len(idx)

        if all_num != 0:
            imatch = np.sum(database_labels[idx[:], :] == label, 1) > 0 # 在所有匹配的哈希码中筛选
            match_num = np.sum(imatch) # 计算与第i个哈希码匹配的且有相同label的哈希码的数量
            precX.append(np.float(match_num) / all_num)

            all_sim_num = np.sum(
                np.sum(database_labels[:, :] == label, 1) > 0)
            recX.append(np.float(match_num) / all_sim_num)

            if radius < 10:
                ips_trad = np.dot(
                    query_output[i, :], database_output[ids[i, 0:all_num], :].T)
                ids_trad = np.argsort(-ips_trad, axis=0)
                db_labels = database_labels[ids[i, 0:all_num], :]

                rel = match_num
                imatch = np.sum(db_labels[ids_trad, :] == label, 1) > 0
                Lx = np.cumsum(imatch)
                Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
                if rel != 0:
                    mAPX.append(np.sum(Px * imatch) / rel)
            else:
                mAPX.append(np.float(match_num) / all_num)

        else:
            precX.append(np.float(0.0))
            recX.append(np.float(0.0))
            mAPX.append(np.float(0.0))

    return np.mean(np.array(precX)), np.mean(np.array(recX)), np.mean(np.array(mAPX))
