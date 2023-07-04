from loss.HashNetLoss import HashNetLoss
from scripts.head import *
from scripts.utils import CalcTopMap, compute_result, pr_curve

writer = SummaryWriter()


def step_lr_scheduler(param_lr, optimizer, iter_num, gamma, step, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (gamma ** (iter_num // step))

    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1

    return optimizer


def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def train_val(config, bit, l):
    device = config['device']
    if config['dataset'] == 'imagenet' or config['dataset'] == 'mscoco':
        train_loader, test_loader, database_loader, num_train, num_test, num_database = get_imagenet_data(config)
    else:
        train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data(config)
    net = config['net'](config, bit, config['label_size']).cuda()
    if config['n_gpu'] > 1:
        net = torch.nn.DataParallel(net)
    optimizer = config['optimizer']['type'](net.parameters(), **(config['optimizer']['optim_param']))
    config['num_train'] = num_train

    Best_map = 0
    print('finish load config')

    criterion = OurLoss(config, bit, l)
    print(f'dmin: {criterion.d_min}, dmax: {criterion.d_max}')

    print('baseline...')

    count = 0
    logger.info(f"config: {str(config)}")
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    for epoch in range(config['epoch']):
        current_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
        logger.info(
            f"{config['info']} {epoch + 1}/{config['epoch']} {current_time} bits: {bit} dataset: {config['dataset']} loss way: {config['loss_way']} training...")
        net.train()

        train_loss = 0
        train_center_loss = 0
        train_pair_loss = 0
        for img, label, ind in tqdm(train_loader):
            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()

            u1, u2 = net(img, None, None)
            loss, center_loss, pair_loss = criterion(u1, u2, label, ind, epoch)
            if config['n_gpu'] > 1:
                loss = loss.mean()
                center_loss = center_loss.mean()
                pair_loss = pair_loss.mean()
            train_loss += loss.item()
            train_center_loss += center_loss.item()
            if epoch >= config['epoch_change']:
                train_pair_loss += pair_loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config['max_norm'])
            optimizer.step()

        train_loss /= len(train_loader)
        logger.info(f"train loss: {train_loss}, center loss: {train_center_loss}, pair loss: {train_pair_loss}")
        writer.add_scalar('train_loss', train_loss, epoch)

        if (epoch + 1) % config['test_map'] == 0:
            net.eval()
            tst_binary, tst_label = compute_result(test_loader, net, config['device'], None, None)
            trn_binary, trn_label = compute_result(database_loader, net, config['device'], None, None)
            mAP, class_map = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                                        config['topk'])
            writer.add_scalar('map', mAP, epoch)
            if mAP > Best_map:
                Best_map = mAP
                count = 0
                if 'save_path' in config:
                    if not os.path.exists(config['save_path']):
                        os.makedirs(config['save_path'])
                    logger.info(f'save in ./results/{config["dataset"]}/{config["loss_way"]}')
            else:
                if count == config['stop_iter']:
                    logger.info(f"valid mAP: {Best_map}")
                    end_time = time.strftime("%H:%M:%S", time.localtime(time.time()))
                    with open(f'./results/{config["dataset"]}/{config["loss_way"]}/map_result.txt', 'a') as f:
                        f.write(str(bit) + '\t' + 'valid: ' + str(Best_map) + '\t' + 'start time: ' + str(start_time) +
                                '\t' + 'end_time:' + str(end_time) + '\t' + str(config) + '\n')
                    break
                count += 1
            logger.info(
                f"{config['info']} {epoch + 1}/{config['epoch']} {current_time} bits: {bit} dataset: {config['dataset']} Best mAP: {Best_map}, current mAP: {mAP}")
        if (epoch + 1) == config['epoch']:
            logger.info(f"valid mAP: {Best_map}")
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            with open(f'./results/{config["dataset"]}/{config["loss_way"]}/map_result.txt', 'a') as f:
                f.write(str(bit) + '\t' + 'valid: ' + str(Best_map) + '\t' + 'start time: ' + str(start_time) +
                        '\t' + 'end_time:' + str(end_time) + '\t' + str(config) + '\n')
            if 'save_path' in config:
                if not os.path.exists(config['save_path']):
                    os.makedirs(config['save_path'])
                logger.info(f'save in ./results/{config["dataset"]}/{config["loss_way"]}')
    writer.close()
