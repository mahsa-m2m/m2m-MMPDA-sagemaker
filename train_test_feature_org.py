from __future__ import print_function, division
import argparse

from data_loader.Low_Level_Augmentation import *

from data_loader import *
# from models.fusion_net import FusionModule
from models_comp.fusion_model import FusionModule, CrossEntropyLabelSmooth
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import cv2
from utils import AvgrageMeter, performances, performances_test
import torch.utils.data
from torch.autograd import Variable

# from MMPareto import MMPareto
import DALoss
from DALoss import *
import DANetwork

import os
# os.environ['CUDA_VISIBLE_DEVICES']="3" # debug

import zipfile

def file2zip(zip_file_name: str, file_names: list):
    """ 将多个文件夹中文件压缩存储为zip
    
    :param zip_file_name:   /root/Document/test.zip
    :param file_names:      ['/root/user/doc/test.txt', ...]
    :return: 
    """
    # 读取写入方式 ZipFile requires mode 'r', 'w', 'x', or 'a'
    # 压缩方式  ZIP_STORED： 存储； ZIP_DEFLATED： 压缩存储
    with zipfile.ZipFile(zip_file_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in file_names:
            parent_path, name = os.path.split(fn)
            
            # zipfile 内置提供的将文件压缩存储在.zip文件中， arcname即zip文件中存入文件的名称
            # 给予的归档名为 arcname (默认情况下将与 filename 一致，但是不带驱动器盘符并会移除开头的路径分隔符)
            zf.write(fn, arcname=name)
            
            # 等价于以下两行代码
            # 切换目录， 直接将文件写入。不切换目录，则会在压缩文件中创建文件的整个路径
            # os.chdir(parent_path)
            # zf.write(name)

def setup_seed(seed):
    np.random.seed(seed) # random seed for numpy
    random.seed(seed) # random seed for random module
    torch.manual_seed(seed) # random seed for CPU
    # torch.backends.cudnn.enabled = False # Forbid CUDNN from using undeterministic algorithm
    torch.backends.cudnn.benchmark = False # benchmark mode (True) can increase computing speed while having randomness
    torch.backends.cudnn.deterministic = True # avoid the randomness in the calculation for feedward network
    torch.cuda.manual_seed(seed) # random seed for GPU
    torch.cuda.manual_seed_all(seed) # random seed for all GPUs

# feature  -->   [ batch, channel, height, width ]
def get_train_dataset_loader(args):
    # print(args.train_list)
    if args.train_dataset == "RLT":
        train_data = getattr(Load_RLT_face_audio_OpenFace_Affect, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    elif args.train_dataset == "BagOfLies":
        train_data = getattr(Load_BagOfLies_face_audio_OpenFace_Affect, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    elif args.train_dataset == "BoxOfLies":
        train_data = getattr(Load_BoxOfLies_face_audio_OpenFace_Affect, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    elif args.train_dataset == "MU3D":
        train_data = getattr(Load_MU3D_face_audio_OpenFace_Affect, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    elif args.train_dataset == "All":
        # train_data = getattr(Load_All_face_audio_wave_OpenFace_Affect, args.train_dataset + '_train')(
        train_data = getattr(Load_All_DbD_face_audio_wave_OpenFace_Affect, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    
    else:
        raise Exception("Train dataset name not exists!")
        # train_data = None

    return train_data


def get_valid_dataset_loader(args):
    if args.valid_dataset == "RLT":
        valid_data = getattr(Load_RLT_face_audio_OpenFace_Affect, args.valid_dataset + '_test')(
            args.valid_list, transform=transforms.Compose([ToTensor_test(), Normaliztion()]))
    elif args.valid_dataset == "BagOfLies":
        valid_data = getattr(Load_BagOfLies_face_audio_OpenFace_Affect, args.valid_dataset + '_test')(
            args.valid_list, transform=transforms.Compose([ToTensor_test(), Normaliztion()]))
    elif args.valid_dataset == "BoxOfLies":
        valid_data = getattr(Load_BoxOfLies_face_audio_OpenFace_Affect, args.valid_dataset + '_test')(
            args.valid_list, transform=transforms.Compose([ToTensor_test(), Normaliztion()]))
        
    elif args.valid_dataset == "MU3D":
        valid_data = getattr(Load_MU3D_face_audio_OpenFace_Affect, args.valid_dataset + '_test')(
            args.valid_list, transform=transforms.Compose([ToTensor_test(), Normaliztion()]))
    else:
        raise Exception("Test dataset name not exists!")
        # valid_data = None

    return valid_data

def get_test_dataset_loader(args):
    if args.test_dataset == "RLT":
        test_data = getattr(Load_RLT_face_audio_OpenFace_Affect, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([ToTensor_test(), Normaliztion()]))
    elif args.test_dataset == "BagOfLies":
        test_data = getattr(Load_BagOfLies_face_audio_OpenFace_Affect, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([ToTensor_test(), Normaliztion()]))
    elif args.test_dataset == "BoxOfLies":
        test_data = getattr(Load_BoxOfLies_face_audio_OpenFace_Affect, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([ToTensor_test(), Normaliztion()]))
        
    elif args.test_dataset == "MU3D":
        test_data = getattr(Load_MU3D_face_audio_OpenFace_Affect, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([ToTensor_test(), Normaliztion()]))
   
    elif args.test_dataset == "MMDD2025_final":
        test_data = getattr(Load_BoxOfLies_face_audio_OpenFace_Affect, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([ToTensor_test(), Normaliztion()]))
    else:
        raise Exception("Test dataset name not exists!")
        # test_data = None

    return test_data


def FeatureMap2Heatmap(x, x2):
    ## initial images
    org_img = x[0, :, :, :].cpu()

    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    # org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    # # 逆标准化
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # org_img = org_img * std + mean  # 回到 [0,1] 范围
    # org_img = org_img.clamp(0, 1)
    # org_img = (org_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) # [H, W, C]
    # org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log + '/' + args.log + '_visual.jpg', org_img)

    org_img = x2[0, :, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    # org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log + '/' + args.log + '_audio.jpg', org_img)

# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    # os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log) and os.path.exists(args.log + '/submit_' + args.test_dataset)
    isExists_score = os.path.exists(args.log + '/test_scores') and os.path.exists(args.log + '/valid_scores')
    if not isExists:
        os.makedirs(args.log, exist_ok=True)
        os.makedirs(args.log + '/submit_' + args.test_dataset, exist_ok=True)
    if not isExists_score:
        os.makedirs(args.log + '/test_scores', exist_ok=True)
        os.makedirs(args.log + '/valid_scores', exist_ok=True)
    log_file = open(
        args.log + '/' + args.fusion_type + "_" + args.modalities + "_" + args.train_dataset + "_to_" +
        args.test_dataset + "_" + str(args.test_list.split('/')[-1].split('.')[0]) + '_log.txt',
        'a') # 'w'

    echo_batches = args.echo_batches

    log_file.write('Start Deception Detection:\n ')
    log_file.write(f'Parameters settings:\n {args}\n')
    print('Start Deception Detection:\n ')
    print(f'Parameters settings:\n {args}\n')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?

    print('Train from scratch!\n')
    log_file.write('Train from scratch!\n')
    log_file.flush()

    # model = ResNet18_GRU(pretrained=True, GRU_layers=1)
    # model = ResNet18_BiGRU(pretrained=True, GRU_layers=2)
    # model = ResNet18(pretrained=True)
    model = FusionModule(args)

    # domain adaption
    class_num = 2
    random_adv_net = False
    if random_adv_net:
        random_dim=1024
        random_layer = DANetwork.RandomLayer([2*len(args.modalities) * args.common_dim, class_num], random_dim)
        ad_net = DANetwork.AdversarialNetwork(random_dim, 1024, max_epochs=args.max_epochs)
    else:
        random_layer = None
        ad_net = DANetwork.AdversarialNetwork(in_feature=2*len(args.modalities) * args.common_dim * class_num, hidden_size=1024, max_epochs=args.max_epochs)

    ad_net = ad_net.cuda()

    # # model = OpenFaceAU_MLP_MLP()
    # # model = OpenFaceGaze_MLP_MLP()
    # # model = OpenFaceGaze_AllMLP()

    model = model.cuda()
    # model = model.to(device[0])
    # model = nn.DataParallel(model, device_ids=device, output_device=device[0])
    lr = args.lr
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0000)
    # optimizer_adv = optim.AdamW(ad_net.parameters(), lr=lr, weight_decay=0.0000)
    optimizer = optim.AdamW(list(model.parameters()) + list(ad_net.parameters()), lr=lr, weight_decay=0.0000)

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.00005)
    # optimizer_adv = torch.optim.SGD(ad_net.parameters(), lr=lr, momentum=0.9, weight_decay=0.00005)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # scheduler_adv = optim.lr_scheduler.StepLR(optimizer_adv, step_size=args.step_size, gamma=args.gamma)

    # for MMPareto
    # record_names_al = []
    # record_names_vl = []
    # record_names_face = []
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         if 'classifier' in name or 'trans' in name or 'fusion' in name: 
    #             continue
    #         if ('vision' in name):
    #             record_names_vl.append((name, param))
    #             continue
    #         if ('face' in name):
    #             record_names_face.append((name, param))
    #             continue
    #         if ('audio' in name):
    #             record_names_al.append((name, param))
    #             continue

    # print(model)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion=CrossEntropyLabelSmooth(num_classes=2)
    best_val_acc, best_val_acc_test = 0.0, 0.0
    
    # domain by domain training
    train_data = get_train_dataset_loader(args)
    
    # source data
    sampler = Load_All_DbD_face_audio_wave_OpenFace_Affect.DomainShuffleSampler(train_data)
    dataloader_train_DbD = DataLoader(train_data, batch_size=args.batchsize, sampler=sampler,num_workers=4, drop_last=True)

    # Scheduler 1：warmup
    warmup_epochs = 1
    steps_per_epoch = len(dataloader_train_DbD)
    total_steps = args.max_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    # Warmup LR scheduler (线性升温)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return 1.0
    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # Cosine scheduler (注意 eta_min 可以设置最小lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)

    for epoch in range(args.max_epochs):  # loop over the dataset multiple times
        # scheduler.step()  # possible warining that scheduler.step is before optimizer.step() --> should be after it!!!
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        loss_global = AvgrageMeter()
        loss_vl = AvgrageMeter()
        loss_face = AvgrageMeter()
        loss_al = AvgrageMeter()
        loss_adaption = AvgrageMeter()
        ###########################################
        '''                train             '''
        ###########################################
        model.train()

        # train_data = get_train_dataset_loader(args)
     
        # # source data
        # dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=True)

        # target data
        domain_adaption=args.adaption
        if domain_adaption:
            # valid_data = get_valid_dataset_loader(args)
            # target_dataloader_val = DataLoader(valid_data, batch_size=args.batchsize, shuffle=True, num_workers=4)
            test_data = get_test_dataset_loader(args)
            target_dataloader_test = DataLoader(test_data, batch_size=args.batchsize, shuffle=True, num_workers=4)    

        len_dataloader=len(dataloader_train_DbD)
        # for i, sample_batched in enumerate(dataloader_train):
        for i, sample_batched in enumerate(dataloader_train_DbD):
            # for _ in range(1000): #  过拟合一个 mini-batch检查梯度不下降的问题
            # get the inputs
            inputs_vision_face = sample_batched['video_x'].cuda()
            inputs_audio_mel = sample_batched['audio_x'].cuda()
            inputs_audio_wave = sample_batched['audio_wave'].cuda()
            inputs_vision_openface = sample_batched['OpenFace_x'].cuda()
            inputs_affect = sample_batched['x_affect'].cuda()

            DD_label = sample_batched['DD_label'].cuda()
            inputs_vision_behavior = torch.cat((inputs_affect, inputs_vision_openface), dim=1)
            # join affect feature to openface  feature
            optimizer.zero_grad()
            # optimizer_adv.zero_grad()

            # log_file.write('\n')
            # log_file.flush()

            # logits =  model(inputs)
            # inputs_vision_behavior combined with affect features are the behavioral
            fused_logit, vl_logit, face_logit, al_logit, feat_source_list = model(inputs_vision_behavior, inputs_vision_face, inputs_audio_mel, inputs_audio_wave)

            # logits =  model(inputs_OpenFace[:,:8,:])
            # logits =  model(inputs_OpenFace[:,8:,:])

            # pdb.set_trace()

            # pdb.set_trace()
            global_loss = criterion(fused_logit, DD_label.squeeze(-1)) # * 0.5
            if vl_logit is not None:
                vl_loss = criterion(vl_logit, DD_label.squeeze(-1))
                face_loss = criterion(face_logit, DD_label.squeeze(-1))
                al_loss = criterion(al_logit, DD_label.squeeze(-1))

            if domain_adaption:
                # sample_target_batched=next(iter(target_dataloader_val))
                sample_target_batched=next(iter(target_dataloader_test))
                inputs_vision_face_target = sample_target_batched['video_x'].cuda()
                inputs_audio_mel_target = sample_target_batched['audio_x'].cuda()
                inputs_audio_wave_target = sample_target_batched['audio_wave'].cuda()
                # DD_label_target = sample_target_batched['DD_label'].cuda()
                inputs_vision_behavior_target = torch.cat((sample_target_batched['x_affect'][:, 0, :, :].cuda(), sample_target_batched['OpenFace_x'][:, 0, :, :].cuda()), dim=1)

                fused_logit_target, _, _, _, feat_target_list = model(
                            inputs_vision_behavior_target,
                            inputs_vision_face_target[:, 0, :, :, :, :],
                            inputs_audio_mel_target[:, 0, :, :, :],
                            inputs_audio_wave_target[:, 0, :]) #
            
                loss_mdd, loss_entropymax, loss_coral = 0, 0, 0
                # for feat_source, feat_target in zip(feat_source_list, feat_target_list):
                feat_source, feat_target = feat_source_list[-1], feat_target_list[-1]
                feat_source, feat_target = feat_source.view(args.batchsize,-1), feat_target.view(args.batchsize,-1)

                #======================================MDD===================================
                features = torch.cat((feat_source, feat_target), dim=0)
                labels_target_fake = torch.max(nn.Softmax(dim=1)(fused_logit_target), 1)[1]
                labels = torch.cat((DD_label.squeeze(-1), labels_target_fake))
                loss_mdd += DALoss.mdd_loss(
                    features=features, labels=labels, left_weight=1, right_weight=1)

                #======================================CoRAL===================================
                # loss_coral = 0.0
                # for feat_source, feat_target in zip(feat_source_list, feat_target_list):
                    # loss_coral += 1.0 * CoralLoss(feat_source.view(args.batchsize,-1), feat_target.view(args.batchsize,-1))
                # loss_coral.backward()
                loss_coral += DALoss.CoralLoss(feat_source, feat_target)
                
                feat_source, feat_target = feat_source_list[-1], feat_target_list[-1]
                features = torch.cat((feat_source, feat_target), dim=0)

                #======================================Entropy Max===================================
                loss_entropymax += DALoss.EntropicConfusion(features)

                #======================================CDAN===================================
                outputs = torch.cat((fused_logit, fused_logit_target), dim=0)
                softmax_out = nn.Softmax(dim=1)(outputs)
                entropy = DALoss.Entropy(softmax_out)
                use_reverse=True
                if use_reverse:
                    p = float(i + epoch * len_dataloader) / args.max_epochs / len_dataloader
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    reverse_features = DANetwork.ReverseLayerF.apply(features, alpha) ### + gradient reversal layer
                    loss_adv = DALoss.CDAN([reverse_features, softmax_out], ad_net, entropy, DANetwork.calc_coeff(epoch, max_iter=args.max_epochs), random_layer)
                else:
                    loss_adv = DALoss.CDAN([features, softmax_out], ad_net, entropy, DANetwork.calc_coeff(epoch, max_iter=args.max_epochs), random_layer)

            # use_mmpareto = False
            # if use_mmpareto:
            #     # add multimodal unimodal non-conflict MMPareto from https://github.com/GeWu-Lab/MMPareto_ICML2024/blob/main/code/MMPareto.py
            #     MMPareto(model, loss_list=[global_loss, vl_loss, face_loss, al_loss], record_names=[record_names_vl, record_names_face, record_names_al], loss_adaption=loss_coral)
            # else:
            loss = global_loss + face_loss + al_loss # + vl_loss
            # Why Openface can not capture effective deception feature ???
            if epoch > 0:
                if domain_adaption:
                    adpation_loss = 10.0*loss_coral + 0.1*loss_mdd + 0.01*loss_entropymax + 0.1*loss_adv
                    (loss+0.1*adpation_loss).backward()
                    # loss.backward()
                else:
                    loss.backward()
            else:
                adpation_loss = torch.tensor(0., dtype=torch.float32).cuda()
                loss.backward()

            # 记录梯度总范数
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Total gradient norm: {total_norm:.6f}")

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            # if epoch > 0:
            #     optimizer_adv.step()

            # 学习率调度
            if epoch == 0:
                scheduler_warmup.step()
            else:
                scheduler_cosine.step()
            # scheduler.step()
            # scheduler_adv.step()

            n = inputs_vision_face.size(0)
            loss_global.update(global_loss.data, n)
            loss_vl.update(vl_loss.data, n)
            loss_face.update(face_loss.data, n)
            loss_al.update(al_loss.data, n)
            if domain_adaption:
                loss_adaption.update(adpation_loss.data, n)

            if i % echo_batches == echo_batches - 1:  # print every 50 mini-batches

                # visualization T=1
                FeatureMap2Heatmap(inputs_vision_face[:, :, 1, :, :], inputs_audio_mel) # B * C X T x H X W
                # FeatureMap2Heatmap(inputs_vision_face[:, 1, :, :, :], inputs_audio_mel) # B * # [T,C,H,W]

                # log written
                current_lr=optimizer.param_groups[0]['lr']
                print('epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f , CE_vl= %.4f , CE_face= %.4f , CE_al= %.4f, L_da= %.4f \n' % (
                    epoch + 1, i + 1, current_lr, loss_global.avg, loss_vl.avg, loss_face.avg, loss_al.avg, loss_adaption.avg))
                
                # break # for debugging val and test

        # whole epoch average
        log_file.write('epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f , CE_vl= %.4f , CE_face= %.4f , CE_al= %.4f, L_da= %.4f \n' % (
                    epoch + 1, i + 1, current_lr, loss_global.avg, loss_vl.avg, loss_face.avg, loss_al.avg, loss_adaption.avg))
        log_file.flush()

        torch.cuda.empty_cache()

        epoch_valid = 1
        if epoch % epoch_valid == epoch_valid - 1:  # valid every epoch
            model.eval()

            with torch.no_grad():

                ###########################################
                '''                val             '''
                ##########################################
                # # differenet clip num for each video, it cannot stack into large batachsize, set = 1
                valid_data = get_valid_dataset_loader(args)
                # valid_data = getattr(data_loader, args.valid_dataset + '_test')(args.test_list, args.test_root,
                #                                                               transform=transforms.Compose(
                #                                                                   [Normaliztion(),
                #                                                                    ToTensor_test()]))
                # valid_data = RLtraial_test(args.test_list, args.test_root,
                #                           transform=transforms.Compose([ToTensor_test(), Normaliztion()]))
                dataloader_valid = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=4)

                map_score_list, video_score_list = [], []

                for i, sample_batched in enumerate(dataloader_valid):

                    # print(sample_batched)
                    inputs_vision_face, DD_label, videoname = sample_batched['video_x'].cuda(), sample_batched['DD_label'].cuda(), sample_batched['videoname']
                    inputs_vision_openface = sample_batched['OpenFace_x'].cuda()
                    inputs_audio_mel = sample_batched['audio_x'].cuda()
                    inputs_audio_wave = sample_batched['audio_wave'].cuda()
                    inputs_affect = sample_batched['x_affect'].cuda()

                    optimizer.zero_grad()

                    # pdb.set_trace()

                    for clip_t in range(inputs_vision_face.shape[1]):
                        fused_logit, _, _, _, _ = model(
                            torch.cat((inputs_affect[:, clip_t, :, :], inputs_vision_openface[:, clip_t, :, :]), dim=1),
                            inputs_vision_face[:, clip_t, :, :, :, :],
                            inputs_audio_mel[:, clip_t, :, :, :],
                            inputs_audio_wave[:, clip_t, :])

                        if args.fusion:
                            if clip_t == 0:
                                logits_accumulate = F.softmax(fused_logit, -1)
                            else:
                                logits_accumulate += F.softmax(fused_logit, -1)
                        else:
                            raise Exception("validing for fusion only!")
                        # todo: valid for other single modality models with/without fusion
                
                    logits_accumulate = logits_accumulate / inputs_vision_face.shape[1]
                    for valid_batch in range(inputs_vision_face.shape[0]):
                        map_score_list.append(
                            '{} {}\n'.format(logits_accumulate[valid_batch][1], DD_label[valid_batch][0]))
                        
                        # for submission
                        video_score_list.append(
                            '{} {}\n'.format(videoname[valid_batch], logits_accumulate[valid_batch][1]))

                valid_filename = args.log + '/valid_scores' + '/' + args.fusion_type + "_" + args.modalities + "_" + \
                                args.train_dataset + "_to_" + args.valid_dataset + "_" + \
                                str(args.valid_list.split('/')[-1].split('.')[0]) + "_scores.txt"
                with open(valid_filename, 'w') as file:
                    file.writelines(map_score_list)
                
                ###########################################################################
                #        performance evaluation
                ##########################################################################
                ACC_RLtrial, AUC_RLtrial, EER_RLtrial = performances(valid_filename)

                print('--------------------valid_epoch:%d, ACC= %.4f, AUC= %.4f, EER= %.4f\n' % (
                        epoch + 1, ACC_RLtrial, AUC_RLtrial, EER_RLtrial))
                log_file.write('valid_epoch:%d, ACC= %.4f, AUC= %.4f, EER= %.4f\n' % (
                    epoch + 1, ACC_RLtrial, AUC_RLtrial, EER_RLtrial))

                # if ACC_RLtrial >= best_val_acc:
                #     best_val_acc = ACC_RLtrial
                #     best_epoch = epoch + 1
                #     best_acc, best_auc, best_err = ACC_RLtrial, AUC_RLtrial, EER_RLtrial
                #     # save best model for DG initial
                #     torch.save({
                #         'args': args,
                #         'epoch': epoch + 1,
                #         'model_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'best_acc': best_val_acc,
                #     }, "saved_checkpoint/MMDD_Tr_{}_Te_{}.pt".format(args.train_dataset, args.valid_dataset))
                    
                testing=args.testing
                if testing:
                    ###########################################
                    '''                test             '''
                    ##########################################
                    test_data = get_test_dataset_loader(args)
                    dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

                    map_score_list, video_score_list, videoname_list = [], [], []

                    for i, sample_batched in enumerate(dataloader_test):

                        # print(sample_batched)
                        inputs_vision_face, DD_label, videoname = sample_batched['video_x'].cuda(), sample_batched['DD_label'].cuda(), sample_batched['videoname']
                        inputs_vision_behavior = sample_batched['OpenFace_x'].cuda()
                        inputs_audio_mel = sample_batched['audio_x'].cuda()
                        inputs_audio_wave = sample_batched['audio_wave'].cuda()
                        inputs_affect = sample_batched['x_affect'].cuda()

                        optimizer.zero_grad()

                        for clip_t in range(inputs_vision_face.shape[1]):
                            fused_logit, _, _, _, _ = model(
                                torch.cat((inputs_affect[:, clip_t, :, :], inputs_vision_behavior[:, clip_t, :, :]), dim=1),
                                inputs_vision_face[:, clip_t, :, :, :, :],
                                inputs_audio_mel[:, clip_t, :, :, :],
                                inputs_audio_wave[:, clip_t, :])

                            if args.fusion:
                                if clip_t == 0:
                                    logits_accumulate = F.softmax(fused_logit, -1)
                                else:
                                    logits_accumulate += F.softmax(fused_logit, -1)
                            else:
                                raise Exception("testing for fusion only!")
                            # todo: test for other single modality models with/without fusion
                        logits_accumulate = logits_accumulate / inputs_vision_face.shape[1]
                        for test_batch in range(inputs_vision_face.shape[0]):
                            map_score_list.append(
                                '{} {}\n'.format(logits_accumulate[test_batch][1], DD_label[test_batch][0]))
                            
                            # for submission
                            video_score_list.append(
                                '{} {}\n'.format(videoname[test_batch], logits_accumulate[test_batch][1]))
                            
                            videoname_list.append(videoname[test_batch])

                    test_filename = args.log + '/test_scores' + '/epoch_' + str(epoch + 1) + args.fusion_type + "_" + args.modalities + "_" + \
                                    args.train_dataset + "_to_" + args.test_dataset + "_" + \
                                    str(args.test_list.split('/')[-1].split('.')[0]) + "_scores.txt"
                    with open(test_filename, 'w') as file:
                        file.writelines(map_score_list)

                    ########### generate submission zip file
                    submit_filename = args.log + '/submit_' + args.test_dataset + '/epoch_' + str(epoch + 1) + args.fusion_type + "_" + args.modalities + "_" + \
                                    args.train_dataset + "_to_" + \
                                    str(args.test_list.split('/')[-1].split('.')[0]) + "_submit.txt"
                    with open(submit_filename, 'w') as file:
                        file.writelines(video_score_list)

                    submission_zip_fname = args.log + '/submit_' + args.test_dataset + '/epoch_' + str(epoch + 1) + args.fusion_type + "_" + args.modalities + "_" +  args.train_dataset + '_submit.zip'
                    file2zip(submission_zip_fname, [submit_filename]) # turn .txt to .zip for submission
                    os.remove(submit_filename)

                    print('-----Saved submission file at: %s\n\n' % (submission_zip_fname))
                    log_file.write('-----Saved submission file at: %s\n\n' % (submission_zip_fname))
                
                    if epoch + 1 >= args.max_epochs:
                        torch.save({
                            'args': args,
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_acc': best_val_acc_test,
                        }, "saved_checkpoint/MMDD_Tr_{}_Te_{}.pt".format(args.train_dataset, args.test_dataset))
                        
                        # ==================================================== #
                        from thop import profile
                        model.eval()
                        # dummy_input = torch.zeros((1, 3, 800, 1422)).to(device)
                        flops, params = profile(model, inputs=(torch.cat((inputs_affect[:, clip_t, :, :], inputs_vision_openface[:, clip_t, :, :]), dim=1),
                            inputs_vision_face[:, clip_t, :, :, :, :],
                            inputs_audio_mel[:, clip_t, :, :, :],
                            inputs_audio_wave[:, clip_t, :]))
                        
                        print('-----Compute Flops: %.2f G, params: %.2f M \n' % (flops / 1e9, params / 1e6))
                        log_file.write('-----Compute Flops: %.2f G, params: %.2f M \n' % (flops / 1e9, params / 1e6))
                    
                # print('--------------------test_epoch:%d, ACC= %.4f, AUC= %.4f, EER= %.4f\n' % (
                #     epoch + 1, ACC_RLtrial, AUC_RLtrial, EER_RLtrial))
                # log_file.write('test_epoch:%d, ACC= %.4f, AUC= %.4f, EER= %.4f\n\n' % (
                #     epoch + 1, ACC_RLtrial, AUC_RLtrial, EER_RLtrial))

                log_file.flush()

        # # EARLY_STOP
        # if (epoch + 1) - best_epoch_test >= args.early_stop:
        #     print(f"end epoch {(epoch + 1)}")
        #     break

    # print('Best_epoch_test:%d, Best ACC= %.4f, Best AUC= %.4f, Best EER= %.4f\n\n' % (
    #     best_epoch_test, best_acc_test, best_auc_test, best_err_test))
    # log_file.write('Best_epoch_test:%d, Best ACC= %.4f, Best AUC= %.4f, Best EER= %.4f\n\n' % (
    #     best_epoch_test, best_acc_test, best_auc_test, best_err_test))
    
    print('Finished Training!!!')
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--device', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')  # default=0.001
    parser.add_argument('--batchsize', type=int, default=8, help='initial batchsize')  # 32
    parser.add_argument('--step_size', type=int, default=5, help='how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='gamma of optim.lr_scheduler.StepLR, decay of lr')  # 0.1
    parser.add_argument('--echo_batches', type=int, default=1, help='how many batches display once')  # 50
    parser.add_argument('--max_epochs', type=int, default=15, help='max training epochs')
    # parser.add_argument('--early_stop', type=int, default=10, help='early_stop training epochs')
    parser.add_argument('--log', type=str, default="logs", help='log and save model name')
    parser.add_argument('--finetune', default=True, help='whether finetune other models') # action='store_true', default=False

    # dataset dirs
    parser.add_argument('--train_dataset', type=str, default='All')
    parser.add_argument('--train_root', type=str, default='',
                        help='train dataset root dir')
    parser.add_argument('--train_list', type=lambda s: [str(item) for item in s.split(',')], default='../Deception_datasets/MMDD2025_features_unify/RLT/RLT_features.pkl',
                        help='train feature list')

    parser.add_argument('--valid_dataset', type=str, default='BoxOfLies')
    parser.add_argument('--valid_root', type=str, default='',
                        help='valid data root')
    parser.add_argument('--valid_list', type=str, default='../Deception_datasets/MMDD2025_features_unify/BoxOfLies/BOL_test_features.pkl',
                        help='valid feature list')
    parser.add_argument('--test_dataset', type=str, default='BoxOfLies')
    parser.add_argument('--test_root', type=str, default='',
                        help='test data root')
    parser.add_argument('--test_list', type=str, default='../Deception_datasets/MMDD2025_features_unify/MMDD2025_final_test_features.pkl',
                        help='test feature list')
    
    parser.add_argument('--testing', action='store_true', help='test testing set or not')
    parser.add_argument('--adaption', action='store_true', help='add domain adpation loss or not')

    # config for individual modal
    parser.add_argument('--fusion', default=True, help='true when fusion module is used') # action='store_true', 
    parser.add_argument('--fusion_modal', type=str, default='FusionModule')

    parser.add_argument('--modalities', type=str, default='vaf', help='modalities in v-affect+openface, a-audio, f-face frames')

    # parser.add_argument('--v_model', type=str, default='OpenFace_Affect7_MLP_MLP')  # OpenFace_Affect7_MLP_MLP # OpenFace_MLP_MLP # behaviour most important in deception deteciton
    # parser.add_argument('--a_model', type=str, default='ResNet18_audio') # ResNet152_audio / ResNet18_audio
    # parser.add_argument('--f_model', type=str, default='ResNet18_GRU') # ResNet152_GRU / ResNet18_GRU

    # dimensions for each modality (the embedding size)
    parser.add_argument('--v_dim', type=int, default=64) # fixed due to T=64frame
    parser.add_argument('--a_dim', type=int, default=512) # 512 / 2048
    parser.add_argument('--f_dim', type=int, default=512) # 256 / 1024 *2
    parser.add_argument('--common_dim', type=int, default=128) # 64 / 768
    parser.add_argument('--kernel_size', type=int, default=3) # 3

    # train with fusion parameters
    parser.add_argument('--fusion_type', type=str, default='mult', help='modality fusion type in '
                                                                          'concat/transformer/senet/mlpmix')
    # parser.add_argument('--concat_dim', type=int, default=-1, help='concatenation dim for concat fusion method')
    
    # config for mult
    parser.add_argument('--mult_layer', type=int, default=3,
                        help='mult attention layers for bimodal and trimodal interaction')
    parser.add_argument('--attn_dropout_mult', type=int, default=0.1,
                        help='attention dropout rate for bimodal and trimodal interaction')

    # config for transformer
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='attention dropout (for audio)')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='number of heads for the transformer network (default: 5)') # 8 2
    parser.add_argument('--layers', type=int, default=4, 
                        help='number of layers in the network (default: 5)') # 4 2
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.0,
                        help='embedding dropout') #0.25
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')
    # config for senet
    parser.add_argument('--channel', type=int, default=64, help='channel dimension for linear layer')
    parser.add_argument('--reduction', type=int, default=16, help='linear dimension reduction')


    # https://aclanthology.org/2022.emnlp-main.189.pdf ensemble

    
    args = parser.parse_args()
    setup_seed(42) # fix seed
    train_test()
