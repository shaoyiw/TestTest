import sys
import pickle
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import io


sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.utils.exp import load_config_file
from isegm.utils.vis import draw_probmap, draw_with_blend_and_clicks, draw_with_clicks, create_vis_images
from isegm.inference.predictors import get_predictor
from isegm.inference.evaluation import evaluate_dataset
from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference

from isegm.utils.visual_attn import visualize_grid_to_grid, tensor_to_PIL, highlight_grid
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['NoBRS', 'RGB-BRS', 'DistMap-BRS',
                                         ' f-BRS-A', 'f-BRS-B', 'f-BRS-C', 'CMRefiner-V2'], default='CMRefiner-V2',
                        help='')

    group_checkpoints = parser.add_mutually_exclusive_group(required=False)
    group_checkpoints.add_argument('--checkpoint', type=str, default='/home/ubuntu/code/SRAFG/experiments/SRAFS/001_our_method/checkpoints/083.pth',
    # group_checkpoints.add_argument('--checkpoint', type=str, default='',
                                   help='The path to the checkpoint. '
                                        'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                                        'or an absolute path. The file extension can be omitted.')
    # group_checkpoints.add_argument('--exp_path', type=str, default='SRAFS/001_our_method',
    group_checkpoints.add_argument('--exp_path', type=str, default='',
                                   help='The relative path to the experiment with checkpoints.'
                                        '(relative to cfg.EXPS_PATH)')

    parser.add_argument('--datasets', type=str, default='GrabCut,DAVIS',
                        help='List of datasets on which the model should be tested. '
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, SBD, PascalVOC, COCO_MVal, COCO_MOIS, HIM2K'
                             'GrabCut,Berkeley,DAVIS,PascalVOC,SBD,BraTS,ssTEM,OAIZIB,COCO_MVal')

    group_device = parser.add_mutually_exclusive_group()
    group_device.add_argument('--gpus', type=str, default='0',
                              help='ID of used GPU.')
    group_device.add_argument('--cpu', action='store_true', default=False,
                              help='Use only CPU for inference.')

    group_iou_thresh = parser.add_mutually_exclusive_group()
    group_iou_thresh.add_argument('--target-iou', type=float, default=0.90,
                                  help='Target IoU threshold for the NoC metric. (min possible value = 0.8)')
    group_iou_thresh.add_argument('--iou-analysis', action='store_true', default=False,
                                  help='Plot mIoU(number of clicks) with target_iou=1.0.')

    parser.add_argument('--n-clicks', type=int, default=20,
                        help='Maximum number of clicks for the NoC metric.')
    parser.add_argument('--min-n-clicks', type=int, default=1,
                        help='Minimum number of clicks for the evaluation.')
    parser.add_argument('--thresh', type=float, required=False, default=0.49,
                        help='The segmentation mask is obtained from the probability outputs using this threshold.')
    parser.add_argument('--clicks-limit', type=int, default=None)
    parser.add_argument('--eval-mode', type=str, default='cvpr',
                        help="Possible choices: cvpr, fixed<number>, or fixed<number>,<number>,(e.g. fixed400, fixed400,600).")

    parser.add_argument('--eval-ritm', action='store_true', default=False)
    parser.add_argument('--save-ious', action='store_true', default=False)
    parser.add_argument('--print-ious', action='store_true', default=False)
    parser.add_argument('--vis-preds', action='store_true', default=False)
    parser.add_argument('--vis_attn_preds', action='store_true', default=False)
    parser.add_argument('--model-name', type=str, default=None,
                        help='The model name that is used for making plots.')
    parser.add_argument('--config-path', type=str, default='../config.yml',
                        help='The path to the config file.')
    parser.add_argument('--logs-path', type=str, default='/home/ubuntu/code/ca_mfp_icm_v2/experiments/SRAFS/001_our_method/evaluation_logs',
                        help='The path to the evaluation logs. Default path: cfg.EXPS_PATH/evaluation_logs.')

    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")

    if (args.iou_analysis or args.print_ious) and args.min_n_clicks <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    if args.logs_path == '':
        args.logs_path = cfg.EXPS_PATH / 'evaluation_logs'
    else:
        args.logs_path = Path(args.logs_path)

    return args, cfg


def main():
    args, cfg = parse_args()

    checkpoints_list, logs_path, logs_prefix = get_checkpoints_list_and_logs_path(args, cfg)
    logs_path.mkdir(parents=True, exist_ok=True)

    single_model_eval = len(checkpoints_list) == 1
    assert not args.iou_analysis if not single_model_eval else True, \
        "Can't perform IoU analysis for multiple checkpoints"
    print_header = single_model_eval
    for dataset_name in args.datasets.split(','):
        dataset = utils.get_dataset(dataset_name, cfg)

        for checkpoint_path in checkpoints_list:
            model = utils.load_is_model(checkpoint_path, args.device, args.eval_ritm)

            predictor_params, zoomin_params = get_predictor_and_zoomin_params(args, dataset_name, eval_ritm=args.eval_ritm)

            # if not args.eval_ritm:
            #     interpolate_pos_embed_inference(model.feature_extractor.backbone, zoomin_params['target_size'], args.device)

            predictor = get_predictor(model, args.mode, args.device,
                                      prob_thresh=args.thresh,
                                      predictor_params=predictor_params,
                                      zoom_in_params=zoomin_params)

            # 保留原有的可视化回调
            vis_callback = get_prediction_vis_callback(logs_path, dataset_name, args.thresh) if args.vis_preds else None
            # 新增注意力图可视化回调
            attn_callback = get_attn_vis_callback(logs_path, dataset_name) if args.vis_attn_preds else None

            dataset_results = evaluate_dataset(dataset, predictor, pred_thr=args.thresh,
                                               max_iou_thr=args.target_iou,
                                               min_clicks=args.min_n_clicks,
                                               max_clicks=args.n_clicks,
                                               callback=vis_callback,
                                               attn_callback=attn_callback)  # 传入新的回调
            # vis_callback = get_prediction_vis_callback(logs_path, dataset_name, args.thresh) if args.vis_preds else None
            # dataset_results = evaluate_dataset(dataset, predictor, pred_thr=args.thresh,
            #                                    max_iou_thr=args.target_iou,
            #                                    min_clicks=args.min_n_clicks,
            #                                    max_clicks=args.n_clicks,
            #                                    callback=vis_callback)

            row_name = args.mode if single_model_eval else checkpoint_path.stem
            if args.iou_analysis:
                save_iou_analysis_data(args, dataset_name, logs_path,
                                       logs_prefix, dataset_results,
                                       model_name=args.model_name)

            save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                         save_ious=single_model_eval and args.save_ious,
                         single_model_eval=single_model_eval,
                         print_header=print_header)
            print_header = False


def get_predictor_and_zoomin_params(args, dataset_name, apply_zoom_in=True, eval_ritm=False):
    predictor_params = {}

    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params['net_clicks_limit'] = args.clicks_limit

    zoom_in_params = None
    if apply_zoom_in and eval_ritm:
        if args.eval_mode == 'cvpr':
            zoom_in_params = {
                'target_size': 600 if dataset_name == 'DAVIS' else 400
            }
        elif args.eval_mode.startswith('fixed'):
            crop_size = int(args.eval_mode[5:])
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (crop_size, crop_size)
            }
        else:
            raise NotImplementedError

    if apply_zoom_in and not eval_ritm:
        if args.eval_mode == 'cvpr':
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (672, 672) if dataset_name == 'DAVIS' else (448, 448)
            }
        elif args.eval_mode.startswith('fixed'):
            crop_size = args.eval_mode.split(',')
            crop_size_h = int(crop_size[0][5:])
            crop_size_w = crop_size_h
            if len(crop_size) == 2:
                crop_size_w = int(crop_size[1])
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (crop_size_h, crop_size_w)
            }
        else:
            raise NotImplementedError

    return predictor_params, zoom_in_params


def get_checkpoints_list_and_logs_path(args, cfg):
    logs_prefix = ''
    if args.exp_path:
        rel_exp_path = args.exp_path
        checkpoint_prefix = ''
        if ':' in rel_exp_path:
            rel_exp_path, checkpoint_prefix = rel_exp_path.split(':')

        exp_path_prefix = cfg.EXPS_PATH / rel_exp_path
        candidates = list(exp_path_prefix.parent.glob(exp_path_prefix.stem + '*'))
        assert len(candidates) == 1, "Invalid experiment path."
        exp_path = candidates[0]
        checkpoints_list = sorted((exp_path / 'checkpoints').glob(checkpoint_prefix + '*.pth'), reverse=True)
        assert len(checkpoints_list) > 0, "Couldn't find any checkpoints."

        if checkpoint_prefix:
            if len(checkpoints_list) == 1:
                logs_prefix = checkpoints_list[0].stem
            else:
                logs_prefix = f'all_{checkpoint_prefix}'
        else:
            logs_prefix = 'all_checkpoints'

        logs_path = args.logs_path / exp_path.relative_to(cfg.EXPS_PATH)
    else:
        checkpoints_list = [Path(utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint))]
        logs_path = args.logs_path / 'others' / checkpoints_list[0].stem

    return checkpoints_list, logs_path, logs_prefix


def save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                 save_ious=False, print_header=True, single_model_eval=False):
    all_ious, all_bious, all_assds, elapsed_time = dataset_results
    mean_spc, mean_spi = utils.get_time_metrics(all_ious, elapsed_time)

    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()
    noc_list, noc_list_std, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=iou_thrs, max_clicks=args.n_clicks)


    row_name = 'last' if row_name == 'last_checkpoint' else row_name
    model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem
    header, table_row = utils.get_results_table(noc_list, over_max_list, row_name, dataset_name,
                                                mean_spc, elapsed_time, args.n_clicks,
                                                model_name=model_name)

    if args.print_ious:
        min_num_clicks = min(len(x) for x in all_ious)
        mean_ious = np.array([x[:min_num_clicks] for x in all_ious]).mean(axis=0)
        miou_str = ' '.join([f'mIoU@{click_id}={mean_ious[click_id - 1]:.4f};'
                             for click_id in range(1, args.n_clicks + 1) if click_id <= min_num_clicks])
        table_row += '\n\n' + miou_str

        mean_bious = np.array([x[:min_num_clicks] for x in all_bious]).mean(axis=0)
        mbiou_str = ' '.join([f'mBIoU@{click_id}={mean_bious[click_id - 1]:.4f};'
                              for click_id in range(1, args.n_clicks + 1) if click_id <= min_num_clicks])
        table_row += '\n\n' + mbiou_str

        mean_assds = np.array([x[:min_num_clicks] for x in all_assds]).mean(axis=0)
        massds_str = ' '.join([f'ASSD@{click_id}={mean_assds[click_id - 1]:.4f};'
                               for click_id in range(1, args.n_clicks + 1) if click_id <= min_num_clicks])
        table_row += '\n\n' + massds_str
    else:
        target_iou_int = int(args.target_iou * 100)
        if target_iou_int not in [80, 85, 90]:
            noc_list, _, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=[args.target_iou],
                                                               max_clicks=args.n_clicks)
            table_row += f' NoC@{args.target_iou:.1%} = {noc_list[0]:.2f};'
            table_row += f' >={args.n_clicks}@{args.target_iou:.1%} = {over_max_list[0]}'

    if print_header:
        print(header)
    print(table_row)

    if save_ious:
        ious_path = logs_path / 'ious' / (logs_prefix if logs_prefix else '')
        ious_path.mkdir(parents=True, exist_ok=True)
        with open(ious_path / f'{dataset_name}_{args.eval_mode}_{args.mode}_{args.n_clicks}.pkl', 'wb') as fp:
            pickle.dump(all_ious, fp)

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
        if not single_model_eval:
            name_prefix += f'{dataset_name}_'

    log_path = logs_path / f'{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.txt'
    if log_path.exists():
        with open(log_path, 'a') as f:
            f.write(table_row + '\n')
    else:
        with open(log_path, 'w') as f:
            if print_header:
                f.write(header + '\n')
            f.write(table_row + '\n')


def save_iou_analysis_data(args, dataset_name, logs_path, logs_prefix, dataset_results, model_name=None):
    all_ious, _, _, _ = dataset_results

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
    name_prefix += dataset_name + '_'
    if model_name is None:
        model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem

    pkl_path = logs_path / f'plots/{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.pickle'
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open('wb') as f:
        pickle.dump({
            'dataset_name': dataset_name,
            'model_name': f'{model_name}_{args.mode}',
            'all_ious': all_ious
        }, f)


def iou_cal(gt_mask, pred_mask):
    area1 = gt_mask.sum()
    area2 = pred_mask.sum()
    inter = (gt_mask & pred_mask).sum()
    iou_c = inter * 1000 / (area1 + area2 - inter)
    return int(iou_c)


def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh, ):
    save_path = logs_path / 'predictions_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    save_color_image_path = logs_path / 'predictions_color_vis' / dataset_name
    save_color_image_path.mkdir(parents=True, exist_ok=True)

    def callback(image, gt_mask, pred_probs, iou, sample_id, click_indx, clicks_list):
        sample_path = save_path / f'{sample_id}_{click_indx}.jpg'
        sample_color_path = save_color_image_path / f'{sample_id}_{click_indx}.jpg'
        prob_map = draw_probmap(pred_probs)
        image_with_mask, mask_with_points = draw_with_blend_and_clicks(image, pred_probs > prob_thresh, clicks_list=clicks_list)
        #  SBD Berkeley
        cv2.putText(mask_with_points, 'iou=%.2f%%' % (iou * 100), (mask_with_points.shape[1] - 160, mask_with_points.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(mask_with_points, 'NoC=%d' % (click_indx + 1),
                    (mask_with_points.shape[1] - 100, mask_with_points.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # # DAVIS
        # cv2.putText(mask_with_points, 'iou=%.2f%%' % (iou * 100), (mask_with_points.shape[1] - 280, mask_with_points.shape[0] - 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        #
        # cv2.putText(mask_with_points, 'NoC=%d' % (click_indx + 1), (mask_with_points.shape[1] - 180, mask_with_points.shape[0] - 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        #
        # cv2.imwrite(str(sample_path), np.concatenate((image_with_mask, mask_with_points), axis=1)[:, :, ::-1])

        cv2.imwrite(str(sample_path),  mask_with_points[:, :, ::-1])
        cv2.imwrite(str(sample_color_path),  image_with_mask[:, :, ::-1])
    return callback


def get_attn_vis_callback(logs_path, dataset_name, model_input_size=(448, 448)):
    """
    [最终需求-v2版]
    创建一个回调函数，该函数：
    1. 在模型输入的尺寸(448x448)上，将注意力图与处理后的背景图完美叠加，生成一个临时的合成图。
    2. 将这个临时的合成图，作为一个整体，缩放回原始图像的尺寸。
    3. 只输出一张最终的、尺寸正确的、无红框的合成图。
    """
    save_path = logs_path / 'predictions_attn_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    def callback(predictor, sample_id, click_indx):
        # 从 predictor 中获取所有需要的数据
        original_image_nd = predictor.original_image
        processed_image_nd = predictor.last_processed_image_nd
        att_maps = predictor.last_att_maps

        if att_maps is None or processed_image_nd is None:
            return

        try:
            # --- [步骤 1: 获取原始尺寸和处理后的背景] ---
            # 获取原始目标尺寸
            pil_orig = tensor_to_PIL(original_image_nd)
            W_orig, H_orig = pil_orig.size

            # 获取 448x448 的背景图用于叠加
            processed_rgb_tensor = processed_image_nd[0, :3, :, :]
            bg_448 = tensor_to_PIL(processed_rgb_tensor)

            # --- [步骤 2: 准备 448x448 的注意力热力图] ---
            attn_tensor = att_maps[-1][0][0].mean(dim=0).cpu().numpy()
            grid_size = int(np.sqrt(attn_tensor.shape[0]))
            grid_index = (grid_size // 2) * grid_size + (grid_size // 2)
            attn_column = attn_tensor[:, grid_index]
            attn_map = attn_column.reshape(grid_size, grid_size)

            mask_448 = Image.fromarray(attn_map).resize(model_input_size, Image.Resampling.BICUBIC)
            mask_448_np = np.array(mask_448)

            # --- [步骤 3: 在内存中生成 448x448 的合成图] ---
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))  # 创建一个正方形画布
            fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            ax.margins(0, 0)
            ax.axis('off')

            ax.imshow(bg_448)
            ax.imshow(mask_448_np / mask_448_np.max(), alpha=0.6, cmap='rainbow')

            # 将 matplotlib 图像保存到内存缓冲区
            buf = io.BytesIO()
            plt.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0)
            buf.seek(0)

            # 从缓冲区读取为 PIL 图像
            blended_448_img = Image.open(buf)
            plt.close(fig)  # 关闭画布，释放内存

            # --- [步骤 4: 将合成图缩放回原始尺寸] ---
            final_image = blended_448_img.resize((W_orig, H_orig), Image.Resampling.LANCZOS)

            # --- [步骤 5: 保存最终结果] ---
            sample_path = save_path / f'{sample_id}_{click_indx}_attn.jpg'
            final_image.save(str(sample_path))

        except Exception as e:
            import traceback
            print(f"Could not visualize attention for {sample_id}_{click_indx}: {e}")
            traceback.print_exc()

    return callback


# def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh):
#     save_path = logs_path / 'predictions_vis' / dataset_name
#     save_path.mkdir(parents=True, exist_ok=True)
#
#     # Define subdirectories for better organization
#     mask_save_path = save_path / 'masks'
#     pos_clicks_save_path = save_path / 'pos_clicks'
#     neg_clicks_save_path = save_path / 'neg_clicks'
#
#     mask_save_path.mkdir(exist_ok=True)
#     pos_clicks_save_path.mkdir(exist_ok=True)
#     neg_clicks_save_path.mkdir(exist_ok=True)
#
#     def callback(image, gt_mask, pred_probs, iou, sample_id, click_indx, clicks_list, **kwargs):
#         # Calculate IoU for filename
#         iou_curr = iou_cal(gt_mask > 0.5, pred_probs > prob_thresh)
#
#         # Generate the three visualization images
#         pred_mask_binary = pred_probs > prob_thresh
#         binary_mask_vis, pos_clicks_vis, neg_clicks_vis = create_vis_images(
#             image,
#             mask=pred_mask_binary,
#             clicks_list=clicks_list,
#             pos_color=(0, 255, 0),  # Green for positive
#             neg_color=(0, 0, 255),  # Red for negative
#             radius=4
#         )
#
#         # Define file paths
#         base_filename = f'{sample_id}_{click_indx}_{iou_curr}.jpg'
#
#         mask_filepath = mask_save_path / base_filename
#         pos_clicks_filepath = pos_clicks_save_path / base_filename
#         neg_clicks_filepath = neg_clicks_save_path / base_filename
#
#         # Save the images
#         cv2.imwrite(str(mask_filepath), binary_mask_vis)
#         cv2.imwrite(str(pos_clicks_filepath), pos_clicks_vis)
#         cv2.imwrite(str(neg_clicks_filepath), neg_clicks_vis)
#
#         # The part for saving ground truth can remain the same
#         gt_mask_path = save_path / f'{sample_id}_gt.jpg'
#         if not os.path.exists(gt_mask_path):
#             # Create a visual representation of the GT mask for reference
#             gt_mask_binary = gt_mask > 0.5
#             gt_mask_vis = (gt_mask_binary.astype(np.uint8) * 255)[:, :, np.newaxis]
#             gt_mask_vis = np.concatenate([gt_mask_vis] * 3, axis=2)
#             cv2.imwrite(str(gt_mask_path), gt_mask_vis)
#
#     return callback
# def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh):
#     save_path = logs_path / 'predictions_vis' / dataset_name
#     save_path.mkdir(parents=True, exist_ok=True)
#
#     def callback(image, gt_mask, pred_probs, sample_id, click_indx, clicks_list):
#         iou_curr = iou_cal(gt_mask > prob_thresh, pred_probs > prob_thresh)
#
#         sample_path = save_path / f'{sample_id}_{click_indx}_{iou_curr}.jpg'
#         gt_mask_path = save_path / f'{sample_id}.jpg'
#         gt_prob_path = save_path / f'{sample_id}_gt.jpg'
#         prob_map = draw_probmap(pred_probs)
#         image_with_mask = draw_with_blend_and_clicks(image, pred_probs > prob_thresh, clicks_list=clicks_list)
#         image_with_gt_mask = draw_with_blend_and_clicks(image, gt_mask > 0.98)
#
#         cv2.imwrite(str(sample_path), np.concatenate((image_with_mask, prob_map), axis=1)[:, :, ::-1])
#
#         if os.path.exists(gt_mask_path):
#             pass
#         else:
#             image_with_gt_mask[:, :, [0, 2]] = image_with_gt_mask[:, :, [2, 0]]
#             # prob_map[:, :, [0, 2]] = prob_map[:, :, [2, 0]]
#             # cv2.imwrite(str(gt_prob_path), gt_mask*255)
#             cv2.imwrite(str(gt_mask_path), image_with_gt_mask)
#
#     return callback


if __name__ == '__main__':
    main()

    
