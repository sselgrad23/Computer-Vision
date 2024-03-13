import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # TODO

        self.conv1 = nn.Conv2d(3, 8, (3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(8, 8, (3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)

        self.conv3 = nn.Conv2d(8, 16, (5, 5), stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        self.conv5 = nn.Conv2d(16, 16, (3, 3), stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(16, 32, (5, 5), stride=2, padding=2)
        self.bn6 = nn.BatchNorm2d(32)

        self.conv7 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(32)

        self.conv8 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(32)

        self.conv9 = nn.Conv2d(32, 32, (3, 3), stride=1, padding=1)

    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        x = self.conv1(x.float())
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)

        x = self.conv9(x)
        return x


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO
        self.c0l = nn.Conv2d(G, 8, (3, 3), stride=1, padding=1)
        self.relul = nn.ReLU()

        self.c1l = nn.Conv2d(8, 16, (3, 3), stride=2, padding=1)
        self.c2l = nn.Conv2d(16, 32, (3, 3), stride=2, padding=1)
        self.c3l = nn.ConvTranspose2d(32, 16, (3, 3), stride=2, padding=1, output_padding=1)
        self.c4l = nn.ConvTranspose2d(16, 8, (3, 3), stride=2, padding=1, output_padding=1)
        self.s_barl = nn.Conv2d(8, 1, (3, 3), stride=1, padding=1)

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO

        B, G, D, H, W = x.shape
        x1 = x.permute(0, 2, 1, 3, 4).reshape(B * D, G, H, W).float()

        c0 = self.c0l(x1)
        c0 = self.relul(c0)

        c1 = self.c1l(c0)
        c1 = self.relul(c1)

        c2 = self.c2l(c1)
        c2 = self.relul(c2)

        c3 = self.c3l(c2)
        c3 = self.relul(c3)

        c4 = self.c4l(c3 + c1)
        s_bar = self.s_barl(c4 + c0)
        s_bar = s_bar.squeeze(1).reshape(B, D, H, W)
        return s_bar


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B, C, H, W = src_fea.size()
    D = depth_values.size(1)

    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        # TODO

        # Rotate_3d = Rotation * 3d pixel coordinates
        to_3d = torch.stack((x, y, torch.ones_like(x))).unsqueeze(0).repeat(B, 1, 1)
        rotate_3d = rot.float() @ to_3d

        # Rotate_depth = rotate_3d * depth
        rotate_3d = rotate_3d.unsqueeze(2).repeat(1, 1, D, 1)
        depth = depth_values.view(B, 1, D, 1)
        rotate_depth = rotate_3d * depth

        # Pixel coords in camera 2 = rotated depth + relative translation vector
        rel_trans = trans.view(B, 3, 1, 1)
        project_3d = rotate_depth + rel_trans
        project_2d = project_3d[:, :2, :, :] / project_3d[:, 2:3, :, :]

        # Normalized points
        proj_normal_x = project_2d[:, 0, :, :] / ((W-1)/2) - 1
        proj_normal_y = project_2d[:, 1, :, :] / ((H-1)/2) - 1
        project_2d = torch.stack((proj_normal_x, proj_normal_y), dim=3)
        grid = project_2d.float()

    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO
    warped_src_fea = F.grid_sample(src_fea, grid.view(B, D * H, W, 2), mode='bilinear', padding_mode='zeros')

    # Reshape so the dimensions are [B, C, D, H, W]
    warped_src_fea = warped_src_fea.view(B, C, D, H, W)

    return warped_src_fea


def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    B, C, D, H, W = warped_src_fea.size()
    S = torch.zeros((B, G, D, H, W), dtype=torch.float64, device=ref_fea.device)
    cg = C // G
    ref_bigger = ref_fea.unsqueeze(2)
    for i in range(G):
        S[:, i] = torch.sum(ref_bigger[:, i * cg: (i + 1) * cg] * warped_src_fea[:, i * cg: (i + 1) * cg], 1)
    return S


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO

    # For batch index b_idx and depth index d_idx, weight all
    # p[b_idx, d_idx, :, :] by depth_values[b_idx, d_idx]. Then,
    # sum over all d_idx in p[b_idx, d_idx, :, :] for a given batch
    # b_idx ("collapse the D dimension by summing"). The output
    # should be [B, H, W].
    unsqueezed_depth = depth_values.view(*depth_values.shape, 1, 1)
    intermediate = p * unsqueezed_depth
    dp = torch.sum(intermediate, 1)
    return dp


def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    est = depth_est[mask.bool()]
    gt = depth_gt[mask.bool()]
    return F.l1_loss(est, gt)
