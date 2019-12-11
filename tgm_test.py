import torchgeometry as tgm
import numpy as np
import cv2
import torch

reduce = 4.0
cam0 = np.eye(4)
cam1 = np.eye(4)
cam0[:3, :3] = np.array([2826.171, 0, 1292.2 / reduce, 0, 2826.171, 965.806 / reduce, 0, 0, 1]).reshape((3, 3))
cam1[:3, :3] = np.array([2826.171, 0, 1415.97 / reduce, 0, 2826.171, 965.806 / reduce, 0, 0, 1]).reshape((3, 3))

T0 = np.eye(4)
T1 = np.eye(4)
T1[0, 3] = 178.089 / 1000.

f0 = '/tmp/testepi/im0.png'
f1 = '/tmp/testepi/im1.png'

img0 = cv2.imread(f0)
img1 = cv2.imread(f1)

img0 = cv2.resize(img0, (int(img0.shape[1] / reduce), int(img0.shape[0] / reduce)))
img1 = cv2.resize(img1, (int(img1.shape[1] / reduce), int(img1.shape[0] / reduce)))

print(img0.shape)

cam0 = torch.Tensor(cam0).unsqueeze(0)
cam1 = torch.Tensor(cam1).unsqueeze(0)
T0 = torch.Tensor(T0).unsqueeze(0)
T1 = torch.Tensor(T1).unsqueeze(0)
h = img0.shape[0]
w = img0.shape[1]
img_h = torch.Tensor(h)
img_w = torch.Tensor(w)

img_src = torch.Tensor(img0).permute(2, 0, 1).unsqueeze(0)
img_dst = torch.Tensor(img1).permute(2, 0, 1).unsqueeze(0)
print(h,w)
# pinholes camera models
pinhole_dst = tgm.PinholeCamera(intrinsics=cam1, extrinsics=T1, height=img_h, width=img_w)
pinhole_src = tgm.PinholeCamera(intrinsics=cam0, extrinsics=T0, height=img_h, width=img_w)

print(pinhole_src)

depth_src = torch.nn.Parameter(torch.ones(1, 1, h, w))
torch.nn.init.uniform_(depth_src,)
depth_src.requires_grad = True
# optimizer = torch.optim.Adam([depth_src], lr=0.1)
optimizer = torch.optim.SGD([depth_src], lr=0.1)
L1 = torch.nn.L1Loss()

warper = tgm.DepthWarper(pinhole_dst, h, w)
warper.compute_projection_matrix(pinhole_src)

for i in range(10000):
    image_src = tgm.depth_warp(pinhole_dst, pinhole_src, depth_src, img_dst, h, w)  # NxCxHxW
    loss = L1(image_src, img_src)

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    print("loss ", loss)

    depth = depth_src.squeeze().squeeze().detach().numpy()
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    print(depth.shape, np.min(depth), np.max(depth))
    depth = (depth*255).astype(np.uint8)
    cv2.imshow("imaage", cv2.applyColorMap(depth, cv2.COLORMAP_JET))
    cv2.waitKey(1)

# pinhole_src = tgm.PinholeCamera(...)
# >>> # create the depth warper, compute the projection matrix
# >>> warper = tgm.DepthWarper(pinhole_dst, height, width)
# >>> warper.compute_projection_matrix(pinhole_src)
# >>> # warp the destionation frame to reference by depth
# >>> depth_src = torch.ones(1, 1, 32, 32)  # Nx1xHxW
# >>> image_dst = torch.rand(1, 3, 32, 32)  # NxCxHxW
# >>> image_src = warper(depth_src, image_dst)  # NxCxHxW
