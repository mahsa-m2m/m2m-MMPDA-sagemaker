import torch
import numpy as np
import torch.nn as nn

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        l = nn.BCELoss(reduction='none')(ad_out, dc_target)
        return torch.sum(weight.view(-1, 1) * nn.BCELoss()(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)
    
def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def mdd_loss(features, labels, left_weight=1, right_weight=1):
    softmax_out = nn.Softmax(dim=1)(features)
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        raise Exception('Incorrect batch size provided')

    batch_left = softmax_out[:int(0.5 * batch_size)]
    batch_right = softmax_out[int(0.5 * batch_size):]

    loss = torch.norm((batch_left - batch_right).abs(), 2, 1).sum() / float(batch_size)

    labels_left = labels[:int(0.5 * batch_size)]
    batch_left_loss = get_pari_loss1(labels_left, batch_left)

    labels_right = labels[int(0.5 * batch_size):]
    batch_right_loss = get_pari_loss1(labels_right, batch_right)
    return loss + left_weight * batch_left_loss + right_weight * batch_right_loss


def mdd_digit(features, labels, left_weight=1, right_weight=1, weight=1):
    softmax_out = nn.Softmax(dim=1)(features)
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        raise Exception('Incorrect batch size provided')

    batch_left = softmax_out[:int(0.5 * batch_size)]
    batch_right = softmax_out[int(0.5 * batch_size):]

    loss = torch.norm((batch_left - batch_right).abs(), 2, 1).sum() / float(batch_size)

    labels_left = labels[:int(0.5 * batch_size)]
    labels_left_left = labels_left[:int(0.25 * batch_size)]
    labels_left_right = labels_left[int(0.25 * batch_size):]

    batch_left_left = batch_left[:int(0.25 * batch_size)]
    batch_left_right = batch_left[int(0.25 * batch_size):]
    batch_left_loss = get_pair_loss(labels_left_left, labels_left_right, batch_left_left, batch_left_right)

    labels_right = labels[int(0.5 * batch_size):]
    labels_right_left = labels_right[:int(0.25 * batch_size)]
    labels_right_right = labels_right[int(0.25 * batch_size):]

    batch_right_left = batch_right[:int(0.25 * batch_size)]
    batch_right_right = batch_right[int(0.25 * batch_size):]
    batch_right_loss = get_pair_loss(labels_right_left, labels_right_right, batch_right_left, batch_right_right)

    return weight*loss + left_weight * batch_left_loss + right_weight * batch_right_loss

def get_pair_loss(labels_left, labels_right, features_left, features_right):
    loss = 0
    for i in range(len(labels_left)):
        if (labels_left[i] == labels_right[i]):
            loss += torch.norm((features_left[i] - features_right[i]).abs(), 2, 0).sum()
    return loss

def get_pari_loss1(labels, features):
    loss = 0
    count = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if (labels[i] == labels[j]):
                count += 1
                loss += torch.norm((features[i] - features[j]).abs(), 2, 0).sum()
    return loss / count


def EntropicConfusion(features):
    softmax_out = nn.Softmax(dim=1)(features)
    batch_size = features.size(0)
    loss = torch.mul(softmax_out, torch.log(softmax_out)).sum() * (1.0 / batch_size)
    return loss

def CoralLoss(source, target):

    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c