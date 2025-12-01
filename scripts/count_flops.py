import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string

#from mmseg.models import build_segmentor
import torch
import torch.nn as nn


from isegm.model.is_cm_refiner_model_v2 import CMRefinerModel_V2


model_cfg = {
        'crop_size': (448, 448),
        'num_max_points':24,
        'with_prev_mask':True,
        "use_attn_weight": [False, True, False, True],
        'lr': 5e-5,
        'optim': 'adam',
        'use_fp16': False
    }
model = CMRefinerModel_V2(pipeline_version = 's2', model_version = 'b3',
                       use_leaky_relu=False, use_rgb_conv=False, use_disks=True, norm_radius=5, binary_prev_mask=False,
                       with_aux_output=False, **model_cfg)

def sra_flops(h, w, r, dim, num_heads):
    dim_h = dim / num_heads
    n1 = h * w
    n2 = h / r * w / r

    f1 = n1 * dim_h * n2 * num_heads
    f2 = n1 * n2 * dim_h * num_heads

    return f1 + f2


def get_tr_flops(net, input_shape, input_cons = None):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False, input_constructor=input_cons)
    _, H, W = input_shape
    net = net.backbone
    try:
        stage1 = sra_flops(H // 4, W // 4,
                           net.block1[0].attn.sr_ratio,
                           net.block1[0].attn.dim,
                           net.block1[0].attn.num_heads) * len(net.block1)
        stage2 = sra_flops(H // 8, W // 8,
                           net.block2[0].attn.sr_ratio,
                           net.block2[0].attn.dim,
                           net.block2[0].attn.num_heads) * len(net.block2)
        stage3 = sra_flops(H // 16, W // 16,
                           net.block3[0].attn.sr_ratio,
                           net.block3[0].attn.dim,
                           net.block3[0].attn.num_heads) * len(net.block3)
        stage4 = sra_flops(H // 32, W // 32,
                           net.block4[0].attn.sr_ratio,
                           net.block4[0].attn.dim,
                           net.block4[0].attn.num_heads) * len(net.block4)
    except:
        stage1 = sra_flops(H // 4, W // 4,
                           net.block1[0].attn.squeeze_ratio,
                           64,
                           net.block1[0].attn.num_heads) * len(net.block1)
        stage2 = sra_flops(H // 8, W // 8,
                           net.block2[0].attn.squeeze_ratio,
                           128,
                           net.block2[0].attn.num_heads) * len(net.block2)
        stage3 = sra_flops(H // 16, W // 16,
                           net.block3[0].attn.squeeze_ratio,
                           320,
                           net.block3[0].attn.num_heads) * len(net.block3)
        stage4 = sra_flops(H // 32, W // 32,
                           net.block4[0].attn.squeeze_ratio,
                           512,
                           net.block4[0].attn.num_heads) * len(net.block4)

    print(stage1 + stage2 + stage3 + stage4)
    flops += stage1 + stage2 + stage3 + stage4
    return flops_to_string(flops), params_to_string(params)




def prepare_input(resolution):
    image = torch.FloatTensor(1, *resolution)
    B,C,H,W = image.shape
    click_map = torch.FloatTensor(1, 2, H, W)
    final_feature = torch.FloatTensor(1, 256, H//4, W//4)
    cropped_logits = torch.FloatTensor(1, 1, H, W)
    return dict(x = image, x1 = click_map, x2 = final_feature, x3=cropped_logits)


'''
flops_seg, params_seg = get_tr_flops(segformer_model, (3,256,256))
flops_refine, params_refine = get_model_complexity_info(refine_layer, (3,256,256))
print('segmentor: ',flops_seg, params_seg)
print('refiner: ',flops_refine, params_refine)
'''


'''
flops_seg, params_seg = get_model_complexity_info(hr18s_ocr, (3,600,600))
flops_refine, params_refine = get_model_complexity_info(refine_layer, (3,256,256))
print('segmentor: ',flops_seg, params_seg)
print('refiner: ',flops_refine, params_refine)
'''

'''
flops_seg, params_seg = get_model_complexity_info(deeplab, (3,512,512))
print('segmentor: ',flops_seg, params_seg)
'''


flops_seg, params_seg = get_model_complexity_info(model, (4,448,448))
# flops_refine, params_refine = get_model_complexity_info(refine_layer, (3,256,256))
print('segmentor: ',flops_seg, params_seg)

# --- memory ---

device = torch.device('cuda:1')

model.to(device)

print(f"Model is on device: {next(model.parameters()).device}")

model.eval()

torch.cuda.reset_peak_memory_stats(device)

input_tensor = torch.ones(1, 4, 448, 448).to(device)

with torch.no_grad():
    output = model(input_tensor)

torch.cuda.synchronize(device)

peak_memory_bytes = torch.cuda.max_memory_allocated(device)

peak_memory_gb = peak_memory_bytes / (1024 ** 3)

print(f"Mem/G: {peak_memory_gb:.2f}")