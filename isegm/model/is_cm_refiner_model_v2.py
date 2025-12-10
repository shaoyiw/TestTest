import torch
import torch.nn as nn
import torch.nn.functional as F
from isegm.utils.serialization import serialize
from .my_is_model import ISModel
from .modeling.segformer.my_segformer_model_v2 import SegFormer
from isegm.model.ops import DistMaps
from isegm.model.modulation_disPro import modulate_prevMask
from .modifiers import LRMult


class CMRefinerModel_V2(ISModel):
    @serialize
    def __init__(self, feature_stride=4, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, pipeline_version='s2', model_version='b3',
                 **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.pipeline_version = pipeline_version
        self.model_version = model_version
        self.feature_extractor = SegFormer(self.model_version, **kwargs)
        self.feature_extractor.backbone.apply(LRMult(backbone_lr_mult))

        if self.pipeline_version == 's1':
            base_radius = 3
        else:
            base_radius = 5
        self.dist_maps_base = DistMaps(norm_radius=base_radius, spatial_scale=1.0,
                                       cpu_mode=False, use_disks=True)

        self.mfp_r_max = 100

        self.maps_transform = nn.Identity()

    def get_coord_features(self, image, prev_masks, points):
        coord_features = self.dist_maps_base(image, points)
        if prev_masks is not None:
            coord_features = torch.cat((prev_masks, coord_features), dim=1)
        return coord_features

    def backbone_forward(self, image, side_feature, guidance_signal, points, gate):
        outs = self.feature_extractor(image, side_feature, guidance_signal, points, gate)
        outs['instances'] = outs['pred']
        # outs['instances_aux'] = outs['pred']
        return outs

    def forward(self, image, points, gate):
        # points[0,0]=torch.tensor([0, 0, 1])
        # points = points.to(image.device)
        image, prev_mask = self.prepare_input(image)

        is_first_click = torch.all(prev_mask == torch.zeros_like(prev_mask))

        if is_first_click and gate == 0 :
            prev_masks_init = torch.cat([prev_mask, prev_mask], dim=1)
            coord_features_init = self.get_coord_features(image, prev_masks_init, points)
            click_map_init = coord_features_init[:, 2:, :, :]
            guidance_signal_init = torch.cat((prev_masks_init, click_map_init), dim=1)

            if self.pipeline_version == 's1':
                small_image = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=True)
                small_coord_features = F.interpolate(coord_features_init, scale_factor=0.5, mode='bilinear',
                                                     align_corners=True)
                small_guidance_signal = F.interpolate(guidance_signal_init, scale_factor=0.5, mode='bilinear',
                                                      align_corners=True)
                points = torch.div(points, 2, rounding_mode='floor')
            else:
                small_image = image
                small_coord_features = coord_features_init
                small_guidance_signal = guidance_signal_init

            outputs_init = self.backbone_forward(small_image, small_coord_features, small_guidance_signal, points, gate)

            outputs_init['click_map'] = click_map_init
            outputs_init['instances'] = nn.functional.interpolate(outputs_init['instances'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)
            prev_mask = torch.sigmoid(outputs_init['instances'])

            batch_size = points.shape[0]
            points_for_first_correction = torch.full_like(points, -1.0)

            for i in range(batch_size):
                valid_indices = torch.where(points[i, :, -1] >= 0)[0]

                if len(valid_indices) > 0:
                    first_valid_idx = valid_indices[0]
                    points_for_first_correction[i, 0, :2] = points[i, first_valid_idx, :2]
                    points_for_first_correction[i, 0, -1] = 0

            prev_mask_corrected = modulate_prevMask(prev_mask, points_for_first_correction, self.mfp_r_max)
            gate = 1
        else:
            prev_mask_corrected = modulate_prevMask(prev_mask, points, self.mfp_r_max)

        prev_masks = torch.cat([prev_mask, prev_mask_corrected], dim=1)
        coord_features = self.get_coord_features(image, prev_masks, points)
        click_map = coord_features[:, 2:, :, :]
        guidance_signal = torch.cat((prev_masks, click_map), dim=1)

        if self.pipeline_version == 's1':
            small_image = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=True)
            small_coord_features = F.interpolate(coord_features, scale_factor=0.5, mode='bilinear', align_corners=True)
            small_guidance_signal = F.interpolate(guidance_signal, scale_factor=0.5, mode='bilinear',
                                                  align_corners=True)
            points = torch.div(points, 2, rounding_mode='floor')
        else:
            small_image = image
            small_coord_features = coord_features
            small_guidance_signal = guidance_signal

        outputs = self.backbone_forward(small_image, small_coord_features, small_guidance_signal, points, gate)

        outputs['click_map'] = click_map
        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        return outputs