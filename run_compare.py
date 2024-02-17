import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

def process_image(image_path, depth_anything, transform, DEVICE):
    raw_image = cv2.imread(image_path)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    h, w = image.shape[:2]
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        depth = depth_anything(image)
    
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    return depth.cpu().numpy().astype(np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path1', type=str, help='Path to the first image')
    parser.add_argument('--img-path2', type=str, help='Path to the second image')
    parser.add_argument('--outdir', type=str, default='./vis_depth_diff')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    os.makedirs(args.outdir, exist_ok=True)
    
    depth1 = process_image(args.img_path1, depth_anything, transform, DEVICE)
    depth2 = process_image(args.img_path2, depth_anything, transform, DEVICE)
    
    # Compute and visualize the depth difference
    depth_diff = cv2.absdiff(depth1, depth2)
    depth_diff = cv2.applyColorMap(depth_diff, cv2.COLORMAP_INFERNO)
    
    diff_filename = 'depth_difference.png'
    cv2.imwrite(os.path.join(args.outdir, diff_filename), depth_diff)
