import torchgeometry as tgm
import numpy as np
import cv2
import torch
import torch.nn as nn
from kornia.losses import SSIM
from layers import Project3D, BackprojectDepth
import torch.nn.functional as F

class ProNet(nn.Module):
    """Layer to transform a depth image into a point cloud
        """

    def __init__(self, batch_size, height, width):
        super(ProNet, self).__init__()

        self.back = BackprojectDepth(batch_size, height, width)
        self.proj = Project3D(batch_size, height, width)

    def forward(self, depth, K, inv_K, T):
        # print("K",K)
        # print("K_inv", inv_K)
        b = self.back(depth, inv_K)
        f = self.proj(b, K, T)
        return b, f


device = "cpu"#("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

reduce = 1.0
cam0 = np.eye(4)
cam1 = np.eye(4)
cam0[:3, :3] = np.array([500., 0, 640. / 2., 0, 500., 480. / 2., 0, 0, 1]).reshape((3, 3))
cam1[:3, :3] = np.array([500., 0, 640. / 2., 0, 500., 480. / 2., 0, 0, 1]).reshape((3, 3))

K = np.array([500., 0, 640. / 2., 0, 500., 480. / 2., 0, 0, 1]).reshape((3, 3))
K_inv = np.linalg.inv(K)

K = torch.Tensor(K).unsqueeze(0)
K_inv = torch.Tensor(K_inv).unsqueeze(0)

K = K.to(device)
K_inve = K_inv.to(device)

T0 = np.eye(4)
T1 = np.eye(4)
T2 = np.eye(4)
baseline = 0.02
T1[0, 3] = baseline
T2[0, 3] = -baseline

T0 = T0[:3,:]
T1 = T1[:3,:]
T2 = T2[:3,:]

T0 = torch.Tensor(T0).unsqueeze(0)
T1 = torch.Tensor(T1).unsqueeze(0)
T2 = torch.Tensor(T2).unsqueeze(0)
T0 = T0.to(device)
T1 = T1.to(device)
T2 = T2.to(device)
T0.requires_grad = False
T1.requires_grad = False
T2.requires_grad = False
K.requires_grad = False
K_inv.requires_grad = False

f0 = '/home/daniele/data/backups/Sister/RobotStereoExperiments/Datasets/Acquisitions/05_feb_2019/Plate2/frame_00015.png'
f1 = '/home/daniele/data/backups/Sister/RobotStereoExperiments/Datasets/Acquisitions/05_feb_2019/Plate2/frame_00018.png'
f2 = '/home/daniele/data/backups/Sister/RobotStereoExperiments/Datasets/Acquisitions/05_feb_2019/Plate2/frame_00012.png'

img0 = cv2.imread(f0)
img1 = cv2.imread(f1)
img2 = cv2.imread(f2)


img_src = torch.Tensor(img0).float().permute(2, 0, 1).unsqueeze(0) / 255.
img_dst = torch.Tensor(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.
img_dst2 = torch.Tensor(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.
img_src.to(device)
img_dst.to(device)
img_dst2.to(device)

h, w = img_src.shape[2], img_src.shape[3]
net = ProNet(1, h, w)
net = net.to(device)

depth_src = torch.nn.Parameter(torch.ones(1, 1, h, w))
depth_src.to(device)

torch.nn.init.uniform_(depth_src, 0.3,0.301)
depth_src.requires_grad = True


def toDisplayImage(t,b=0):
    if len(t.shape)>3:
        return t[b, ::].permute(1, 2, 0).detach().cpu().numpy()
    else:
        return t[b, ::].detach().cpu().numpy()


optimizer = torch.optim.RMSprop([depth_src], lr=0.001)
# optimizer = torch.optim.SGD([depth_src],lr = 10.01)
L1 = torch.nn.L1Loss()
prediction = nn.Sigmoid()

val = 0.3
for i in range(100000):
    # val = val -0.001
    # torch.nn.init.constant(depth_src, val=val)

    D = prediction(depth_src)

    b, f = net(D, K, K_inv, T1)
    w = F.grid_sample(img_src, f)

    b2, f2 = net(D, K, K_inv, T2)
    w2 = F.grid_sample(img_src, f)


    loss = L1(w[:,0,::], img_dst[:,0,::])# + L1(w2[:,0,::], img_dst[:,0,::])
    #loss = torch.mean(torch.abs(w - img_dst)) + torch.mean(torch.abs(w2 - img_dst))


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss ", loss, depth_src.shape)


    out = toDisplayImage(w[:,0,::])

    dout =toDisplayImage(D)
    print(out.shape, np.min(out), np.max(out))

    cv2.imshow("image", (out * 255.).astype(np.uint8))
    cv2.imshow("depth", cv2.applyColorMap( (dout * 255.).astype(np.uint8), cv2.COLORMAP_JET))
    cv2.waitKey(1)

# # optimizer = torch.optim.SGD([depth_src], lr=0.1)

# LossSSIM = SSIM(5, reduction='mean')
#
# warper = tgm.DepthWarper(pinhole_dst, h, w)
# warper.compute_projection_matrix(pinhole_src)
#

#     warper.compute_subpixel_step()
#     image_src = tgm.depth_warp(pinhole_dst, pinhole_src, 1. / depth_src, img_dst, h, w)  # NxCxHxW
#     out = image_src[0, ::].permute(1, 2, 0).detach().numpy()
#     inp = img_dst[0, ::].permute(1, 2, 0).detach().numpy()
#
#     # print("OUT", out.shape, np.min(out), np.max(out), np.mean(out))
#     # print("INP", inp.shape, np.min(inp), np.max(inp), np.mean(inp))
#     # print("INP2", img_src.shape, image_src.shape)
#
#     loss = L1(image_src, img_src)
#     # loss = LossSSIM(image_src[:, 0, ::].unsqueeze(0), img_src[:, 0, ::].unsqueeze(0))
#
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # print("loss ", loss, np.mean(depth_src.detach().numpy()))
#
#     depth = depth_src.squeeze().squeeze().detach().numpy()
#     depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
#     # print(depth.shape, np.min(depth), np.max(depth))
#     depth = (depth * 255).astype(np.uint8)
#     cv2.imshow("imaage", cv2.applyColorMap(depth, cv2.COLORMAP_JET))
#     cv2.imshow("out", (out * 255.).astype(np.uint8))
#     cv2.waitKey(1)

# pinhole_src = tgm.PinholeCamera(...)
# >>> # create the depth warper, compute the projection matrix
# >>> warper = tgm.DepthWarper(pinhole_dst, height, width)
# >>> warper.compute_projection_matrix(pinhole_src)
# >>> # warp the destionation frame to reference by depth
# >>> depth_src = torch.ones(1, 1, 32, 32)  # Nx1xHxW
# >>> image_dst = torch.rand(1, 3, 32, 32)  # NxCxHxW
# >>> image_src = warper(depth_src, image_dst)  # NxCxHxW
