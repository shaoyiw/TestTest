import math
import torch


def get_last_point(points):
    last_point = torch.zeros((points.shape[0], 1, 4), device=points.device, dtype=points.dtype)
    last_point[:, 0, :3] = points[points[:, :, -1] == points[:, :, -1].max(dim=1)[0].unsqueeze(1)]
    last_point[:, 0, -1][
        torch.argwhere(points[:, :, -1] == points[:, :, -1].max(dim=1)[0].unsqueeze(1))[:, -1] < points.shape[
            1] // 2] = 1
    last_point[:, 0, -1][
        torch.argwhere(points[:, :, -1] == points[:, :, -1].max(dim=1)[0].unsqueeze(1))[:, -1] >= points.shape[
            1] // 2] = 0

    return last_point


def modulate_prevMask(prev_mask, points, R_max):
    alpha = 1.2
    sigma = 0.2

    with torch.no_grad():
        last_point = get_last_point(points)

        if torch.any(last_point < 0):
            return prev_mask

        num_points = points.shape[1] // 2
        row_array = torch.arange(start=0, end=prev_mask.shape[2], step=1, dtype=torch.float64, device=points.device)
        col_array = torch.arange(start=0, end=prev_mask.shape[3], step=1, dtype=torch.float64, device=points.device)
        coord_rows, coord_cols = torch.meshgrid(row_array, col_array, indexing='ij')

        prevMod = prev_mask.detach().clone().to(torch.float64)
        prev_mask = prev_mask.detach().clone()

        for bindx in range(points.shape[0]):
            pos_points = points[bindx, :num_points][points[bindx, :num_points, -1] != -1]
            neg_points = points[bindx, num_points:][points[bindx, num_points:, -1] != -1]

            y, x = last_point[bindx, 0, :2]
            p = prev_mask[bindx, 0, y.long(), x.long()]

            dist = torch.sqrt((coord_rows - y) ** 2 + (coord_cols - x) ** 2)

            if last_point[bindx, :, -1] == 1:

                if neg_points.shape[0] != 0:
                    min_dist = torch.cdist(neg_points[:, :2], last_point[bindx, 0, :2].unsqueeze(0)).min(dim=0)[0]
                    r = min_dist / 2
                    modWindow = (dist <= r)
                    if r < 10:
                        r = 10
                        modWindow = (dist <= r)
                        if min_dist < 10:
                            in_modWindow = neg_points[
                                (torch.cdist(neg_points[:, :2], last_point[bindx, 0, :2].unsqueeze(0)) < 10)[:, 0]]
                            for n_click in in_modWindow:
                                dist_n = torch.sqrt((coord_rows - n_click[0]) ** 2 + (coord_cols - n_click[1]) ** 2)
                                modWindow_n = (dist_n <= torch.sqrt((last_point[bindx, 0, 0] - n_click[0]) ** 2 + (
                                        last_point[bindx, 0, 1] - n_click[1]) ** 2))
                                modWindow[modWindow_n] = 0

                else:
                    r = R_max
                    modWindow = (dist <= r)

                if p == 0:
                    prevMod[bindx, 0][modWindow] = 1 - (dist[modWindow] / (dist[modWindow].max() + 1e-8))
                    continue
                elif p < 0.99:
                    max_gamma = 1 / (math.log(p if p > 0 else 1e-8, 0.99 if 0.99 > p else 1.01) + 1e-8)
                else:
                    max_gamma = 1

                Px = prevMod[bindx, 0][modWindow]
                omega = 1 + (alpha - 1) * torch.exp(-((Px - p) ** 2) / (2 * sigma * sigma))

                exp = omega * max_gamma * (1 - (dist[modWindow] / r)) + (dist[modWindow] / r)

                prevMod[bindx, 0][modWindow] = prevMod[bindx, 0][modWindow] ** (1 / exp)
                prevMod[bindx, 0][int(y.round()), int(x.round())] = 1

            else:
                if pos_points.shape[0] != 0:
                    min_dist = torch.cdist(pos_points[:, :2], last_point[bindx, 0, :2].unsqueeze(0)).min(dim=0)[0]
                    r = min_dist / 2
                    modWindow = (dist <= r)
                    if r < 10:
                        r = 10
                        modWindow = (dist <= r)
                        if min_dist < 10:
                            in_modWindow = pos_points[
                                (torch.cdist(pos_points[:, :2], last_point[bindx, 0, :2].unsqueeze(0)) < 10)[:, 0]]
                            for p_click in in_modWindow:
                                dist_p = torch.sqrt((coord_rows - p_click[0]) ** 2 + (coord_cols - p_click[1]) ** 2)
                                modWindow_p = (dist_p <= torch.sqrt((last_point[bindx, 0, 0] - p_click[0]) ** 2 + (
                                        last_point[bindx, 0, 1] - p_click[1]) ** 2))
                                modWindow[modWindow_p] = 0
                else:
                    r = R_max
                    modWindow = (dist <= r)
                if p == 1:
                    prevMod[bindx, 0][modWindow] = dist[modWindow] / (dist[modWindow].max() + 1e-8)
                    continue
                elif p > 0.01:
                    max_gamma = math.log(0.01, p)
                else:
                    max_gamma = 1

                Px = prevMod[bindx, 0][modWindow]
                omega = 1 + (alpha - 1) * torch.exp(-((Px - p) ** 2) / (2 * sigma * sigma))

                exp = omega * max_gamma * (1 - (dist[modWindow] / r)) + (dist[modWindow] / r)

                prevMod[bindx, 0][modWindow] = prevMod[bindx, 0][modWindow] ** (exp)
                prevMod[bindx, 0][int(y.round()), int(x.round())] = 0

    return prevMod.to(torch.float32)