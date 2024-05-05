import torch
import torch.nn as nn
from loss import batch_episym
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
import torch.nn.functional as F


class SE(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):    # num_channels=64
        super(SE, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.conv0 = Conv2d(num_channels, num_channels,
                            kernel_size=1, stride=1, bias=True)
        self.in0 = nn.InstanceNorm2d(num_channels)
        self.bn0 = nn.BatchNorm2d(num_channels)
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        x = self.in0(input_tensor)
        x = self.bn0(x)
        x = self.relu(x)  # b,128,2000,1
        input_tensor = self.conv0(x)
        squeeze_tensor = input_tensor.view(
            batch_size, num_channels, -1).mean(dim=2)  # 对每个通道求平均值  b,128
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))  # b,64
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))   # b,128
        a, b = squeeze_tensor.size()  # a:batch_size, b:128
        # b,128,2000,1    b,128,1,1---->b,128,2000,1
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class MBSE(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, bottleneck_width=64):  # planes=64
        super(MBSE, self).__init__()
        SE_channel = int(planes * (bottleneck_width / 64.))
        self.shot_cut = None
        if planes*2 != inplanes:
            self.shot_cut = nn.Conv2d(inplanes, planes*2, kernel_size=1)
        self.conv1 = nn.Conv2d(inplanes, SE_channel, kernel_size=1, bias=True)
        self.in1 = nn.InstanceNorm2d(inplanes, eps=1e-5)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = SE(SE_channel)
        self.conv3 = nn.Conv2d(SE_channel, planes*2, kernel_size=1, bias=True)
        self.in3 = nn.InstanceNorm2d(SE_channel, eps=1e-5)
        self.bn3 = nn.BatchNorm2d(SE_channel)
        self.conv_branch1_1 = nn.Sequential(
            nn.InstanceNorm2d(inplanes, eps=1e-3),
            nn.BatchNorm2d(inplanes),
            nn.GELU(),
            nn.Conv2d(inplanes, planes*2, kernel_size=1),
        )
        # self.conv_branch1_2 = nn.Sequential(
        #     nn.InstanceNorm2d(planes*2, eps=1e-3),
        #     nn.BatchNorm2d(planes*2),
        #     nn.GELU(),
        #     nn.Conv2d(planes*2, planes*2, kernel_size=1),
        #
        # )
        self.conv_merge = nn.Sequential(
            nn.InstanceNorm2d(planes*2, eps=1e-3),
            nn.BatchNorm2d(planes*2),
            nn.GELU(),
            nn.Conv2d(planes*2, planes*2, kernel_size=1),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.in1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.in3(out)
        out = self.bn3(out)
        out = self.conv3(out)
        branch_out = self.conv_branch1_1(x)
        #branch_out = self.conv_branch1_2(branch_out)
        if self.shot_cut:
            residual = self.shot_cut(x)
        else:
            residual = x
        out = out+branch_out+residual
        out = self.conv_merge(out)
        return out


class MBMS(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MBMS, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, xa):
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xgm = self.global_att_max(xa)
        xlg = xl + xg + xgm
        wei = self.sigmoid(xlg)
        return wei*xa


class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            trans(1, 2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_geo, x_down):
        embed = self.conv(x_geo)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class DP_OA_DOP_block(nn.Module):
    def __init__(self, channels, l2_nums, num):
        nn.Module.__init__(self)
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = []
        for _ in range(num):
            self.l2.append(OAFilter(channels, l2_nums))
        self.up1 = diff_unpool(channels, l2_nums)
        self.l2 = nn.Sequential(*self.l2)

    def forward(self, pre):
        x_down = self.down1(pre)
        x2 = self.l2(x_down)
        x_up = self.up1(pre, x2)
        return x_up


def transforto0_1(w, l, r):
    w = torch.where(w <= l, 0, w)
    w = torch.where(w > r, 0, w)
    w = torch.where(w != 0, 1, w)
    return w

def get_pos_neg_index(logits, channel, k):
    w = torch.tanh(logits)
    w = torch.unsqueeze(torch.unsqueeze(w, 1), 3)
    flag = 2/k
    nodes = []
    nums = []
    for i in range(k):
        w_i = transforto0_1(w, -1+flag*i, -1+flag*(i+1))
        num = torch.sum(w_i, dim=2)
        num = torch.where(num == 0, 1, num)
        nums.append(num)
        w_i = w_i.repeat(1, channel, 1, 1)
        nodes.append(w_i)
    return nodes, nums


class transformer(nn.Module):
    def __init__(self, dim ):
        nn.Module.__init__(self)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.temperature = torch.sqrt(torch.tensor(dim))
        #self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        q = self.q(x.unsqueeze(3)).squeeze(3)
        k = self.k(x.unsqueeze(3)).squeeze(3)
        v = self.v(x.unsqueeze(3)).squeeze(3)
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        #attn = self.dropout(F.softmax(attn, dim=-1))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output


class GSA(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        #self.se = SE(channels)
        self.se2 = SE(channels)
        #self.se3 = SE(channels)
        self.channels = channels
        self.transformer = transformer(channels)
        self.transformer2 = transformer(channels)
        self.transformer3 = transformer(channels)
        self.layernorm = nn.LayerNorm(channels, eps=1e-6)
        #self.pc = PointCN(channels, channels)

    def forward(self,x, logits, k):
        nodes, nums = get_pos_neg_index(logits, self.channels, k)
        level_1 = []
        nodes_split = []
        for i in range(len(nodes)):
            nodei = x*nodes[i]
            nodes_split.append(nodei)
            nodei = torch.sum(nodei, dim=2)/nums[i]
            level_1.append(nodei)
        #print(w_h)
        feature = torch.cat([node for node in level_1], dim=2)
        out = self.transformer(feature)
        out = out + feature
        out = out.transpose(-1, -2)
        out = self.layernorm(out)
        feature = out.transpose(-1, -2)

        out = self.transformer2(feature)
        out = out + feature
        out = out.transpose(-1, -2)
        out = self.layernorm(out)
        feature = out.transpose(-1, -2)

        out = self.transformer3(feature)
        out = out + feature
        out = out.transpose(-1, -2)
        out = self.layernorm(out)
        out = out.transpose(-1, -2)

        out = out.unsqueeze(3)
        #out = self.se(out)
        level_2 = []
        out = out.chunk(len(nodes), dim=2)
        for node in out:
            level_2.append(node)
        result = torch.zeros_like(x)
        for i in range(len(nodes)):
            result += nodes_split[i] + nodes[i]*level_2[i]
        result = self.se2(result)
        #print(merge[0])
        #result = self.pc(result)
        return result


class sub_MSGSA(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters, isInit=False):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)
        fchannels = channels
        if not isInit:
            fchannels = 2*channels
            self.conv641 = nn.Conv2d(channels, channels//2, kernel_size=1)
            self.conv642 = nn.Conv2d(channels, channels//2, kernel_size=1)
            self.se = SE(channels)
            self.gsa = GSA(channels)
        l2_nums = clusters
        self.l1_1 = []
        self.l1_1.append(MBSE(fchannels, channels//2, 1))
        self.l1_1.append(MBMS(channels))
        self.l1_1.append(MBSE(channels, channels//2, 1))
        self.l1_1.append(PointCN(channels))
        self.l1_1.append(PointCN(channels))
        self.geo = DP_OA_DOP_block(channels, clusters, 3)
        self.l1_2 = []
        self.l1_2.append(PointCN(2*channels, channels))
        self.l1_2.append(PointCN(channels))
        self.l1_2.append(MBSE(channels, channels//2, 1))
        self.l1_2.append(MBMS(channels))
        self.l1_2.append(MBSE(channels, channels//2, 1))
        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        # self.l2 = nn.Sequential(*self.l2)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)
        self.linear1 = nn.Conv2d(channels, 2, kernel_size=1)

    def forward(self, data, xs, x_last=None, x_last2=None, logits_last=None, k=None):
        batch_size, num_pts = data.shape[0], data.shape[2]
        x1_1 = self.conv1(data)
        if x_last is not None:
            x1_1 = self.gsa(x1_1, logits_last, k)
            x_last = self.conv641(x_last)
            x_last2 = self.conv642(x_last2)
            x_last = torch.cat([x_last, x_last2], dim=1)
            x_last = self.se(x_last)
            x1_1 = torch.cat([x1_1, x_last], dim=1)
        x1_1 = self.l1_1(x1_1)
        x2 = self.geo(x1_1)
        out = self.l1_2(torch.cat([x1_1, x2], dim=1))
        logits = torch.squeeze(torch.squeeze(self.output(out), 3), 1)
        logits1, indices = torch.sort(logits, dim=-1, descending=True)

        x_out, feature_out = down_sampling(xs, indices, out)
        w = self.linear1(feature_out)
        e_hat = weighted_8points(x_out, w)
        x1, x2 = xs[:, 0, :, :2], xs[:, 0, :, 2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(
            batch_size, 1, num_pts, 1)
        return logits, e_hat, residual, out, x1_1

def down_sampling(x, indices, features=None):
    B, _, N , _ = x.size()
    # 取t
    indices = indices[:, :int(N*0.5)]
    indices = indices.view(B, 1, -1, 1)

    with torch.no_grad():
        x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
    feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
    return x_out, feature_out

class MSGSA(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = 2
        depth_each_stage = config.net_depth//(config.iter_num+1)
        self.subnetwork_init = sub_MSGSA(
            config.net_channels, 4, depth_each_stage, config.clusters, isInit=True)
        self.subnetwork = [sub_MSGSA(config.net_channels, 6, depth_each_stage,
                                 config.clusters) for _ in range(self.iter_num)]
        self.subnetwork = nn.Sequential(*self.subnetwork)
        self.l = []
        self.l.append(MBSE((self.iter_num+1) * config.net_channels,
                      config.net_channels // 2, 1))
        for _ in range(1):
            self.l.append(MBSE(config.net_channels,
                          config.net_channels // 2, 1))
        self.l = nn.Sequential(*self.l)
        self.covn = nn.Conv2d(128, 1, kernel_size=1)
        self.linear1 = nn.Conv2d(128, 2, kernel_size=1)

    # def down_sampling(self, x, indices):
    #     B, _, N , _ = x.size()
    #     # 取t
    #     indices = indices[:, :int(N*0.5)]
    #
    #     indices = indices.view(B, 1, -1, 1)
    #     x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
    #     return x_out

    def forward(self, data):
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        #data: b*1*n*c
        input = data['xs'].transpose(1, 3)
        res_weights, res_e_hat = [], []
        logits, e_hat, residual, out, x_last = self.subnetwork_init(
            input, data['xs'])
        More_weight = out
        res_weights.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):
            logits, e_hat, residual, out, x_last = self.subnetwork[i](
                torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()], dim=1), data['xs'], x_last, out, logits, 10)
            More_weight = torch.cat([More_weight, out], dim=1)
            res_weights.append(logits), res_e_hat.append(e_hat)
        More_weight = self.l(More_weight)
        feature_out = self.covn(More_weight)
        logits = torch.squeeze(torch.squeeze(feature_out, 3), 1)
        logits1, indices = torch.sort(logits, dim=-1, descending=True)
        x_out, feature_out = down_sampling(data['xs'], indices, More_weight)
        w = self.linear1(feature_out)
        e_hat = weighted_8points(x_out, w)
        res_weights.append(logits), res_e_hat.append(e_hat)
        return res_weights, res_e_hat


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    mask = logits[:, 0, :, 0]
    weights = logits[:, 1, :, 0]

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    # weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)

    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat
