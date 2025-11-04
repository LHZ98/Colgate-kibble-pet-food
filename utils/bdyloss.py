import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

from scipy.ndimage import distance_transform_edt




class SurfaceLoss(nn.Module):
    def __init__(self):
        super(SurfaceLoss, self).__init__()

    def forward(self, y_true, y_pred,true_masks):
        # Calculate the distance map for each sample in the 
        
        # Element-wise multiplication between predicted logits and distance maps
        # print('true:',torch.unique(y_true))
        # print(torch.unique(y_pred))

        ###################
        # dist_maps = self.calc_dist_map_batch(y_true)
        # dist_maps = torch.from_numpy(dist_maps)
        # multipled = y_pred * dist_maps.to(y_pred.device)  # Transfer dist_maps to the same device as y_pred

        # # Calculate the mean loss
        # loss = torch.mean(multipled)
        #####################

        weighted_map = self.unet_weight_map_batch(y_true)
        weighted_map = torch.from_numpy(weighted_map)
        # wrong_pred = 1*torch.logical_not(torch.eq(y_true,y_pred))
        wrong_pred = 1*torch.logical_not(torch.eq(true_masks,y_pred))
        multipled = wrong_pred * weighted_map.to(wrong_pred.device)

        loss_mean = self.mean_batch(multipled,wrong_pred)
        # print(loss_mean)
        loss = torch.nanmean(loss_mean)

        return loss,weighted_map,multipled

    def mean_batch(self,multipled,wrong_pred):
        mean = torch.empty(multipled.shape[0])
        for i in range(multipled.shape[0]):  
            x = multipled[i, :, :]
            wrong_batch = wrong_pred[i,:,:]
            nonzero = torch.count_nonzero(wrong_batch)
            total = torch.numel(wrong_batch)
            mean[i] = x[x !=0].mean()*nonzero/total
        return mean




    def calc_dist_map(self,seg):
        res = np.zeros_like(seg)
        posmask = seg.astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res = distance(negmask) * negmask -  (distance(posmask) - 1) * posmask

        return res


    def calc_dist_map_batch(self,y_true):
        y_true_n = y_true.cpu()
        y_true_numpy = y_true_n.numpy()
        return np.array([self.calc_dist_map(y)
                         for y in y_true_numpy]).reshape(y_true.shape).astype(np.float32)

    

    def unet_weight_map(self,y, wc=None, w0 = 10, sigma = 20):
        labels = y
        labels = labels.squeeze()
        no_labels = labels == 0
        label_ids = sorted(np.unique(labels))
        # print(label_ids)
        if len(label_ids) > 1:
            distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))
            for i, label_id in enumerate(label_ids):
                distances[:,:,i] = distance_transform_edt(labels != label_id)
            distances = np.sort(distances, axis=2)
            d1 = distances[:,:,0]
            d2 = distances[:,:,1]
            w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
        else:
            w = np.zeros_like(y)
        if wc:
            class_weights = np.zeros_like(y)
            for k, v in wc.items():
                class_weights[y == k] = v
            w = w + class_weights
        return w

    def unet_weight_map_batch(self, y_true, wc=None, w0 = 10, sigma = 20):
        y_true_n = y_true.cpu()
        y_true_numpy = y_true_n.numpy()
        return np.array([self.unet_weight_map(y0)
                         for y0 in y_true_numpy]).reshape(y_true.shape).astype(np.float32)

    # def calc_dist_map(self, seg):
    #     # Convert seg to binary mask (0s and 1s)
    #     posmask = (seg > 0).float()

    #     # Use the signed distance transform
    #     dist = self.signed_distance_transform(posmask)
    #     dist = dist * (1 - 2 * posmask)  # Apply sign to the distance
    #     # print(torch.unique(dist))

    #     return dist

    # def calc_dist_map_batch(self, y_true):
    #     dist_maps = []
    #     for y in y_true:
    #         dist_map = self.calc_dist_map(y)
    #         dist_maps.append(dist_map)
    #     return torch.stack(dist_maps)

    # def signed_distance_transform(self, binary_mask):
    #     """
    #     Compute the signed distance transform of a binary mask.
    #     """
    #     distance_transform = torch.zeros_like(binary_mask).to(binary_mask.device)

    #     pos_indices = torch.where(binary_mask > 0)
    #     neg_indices = torch.where(binary_mask == 0)

    #     for i in range(len(pos_indices[0])):
    #         p = (pos_indices[0][i], pos_indices[1][i])
    #         for j in range(len(neg_indices[0])):
    #             q = (neg_indices[0][j], neg_indices[1][j])
    #             dist = torch.norm(torch.tensor(p).to(binary_mask.device) - torch.tensor(q).to(binary_mask.device))
    #             distance_transform[q] = torch.minimum(distance_transform[q], dist)

    #     return distance_transform



    def signed_distance_transform(self, binary_mask):
        """
        Compute the signed distance transform of a binary mask.
        """
        device = binary_mask.device
        dist = torch.zeros_like(binary_mask, dtype=torch.float32).to(device)
        
        neg_indices = torch.where(binary_mask == 0)
        for i in range(neg_indices[0].shape[0]):
            y, x = neg_indices[0][i], neg_indices[1][i]
            dist[y, x] = self.compute_signed_distance(binary_mask, y, x)
            
        return dist

    def compute_signed_distance(self, binary_mask, y, x):
        """
        Compute the signed distance from (y, x) to the nearest boundary in binary_mask.
        """
        h, w = binary_mask.shape
        min_dist = float('inf')
        
        for i in range(h):
            for j in range(w):
                if binary_mask[i, j] == 1:
                    dist = torch.norm(torch.tensor([y - i, x - j], dtype=torch.float32))
                    min_dist = min(min_dist, dist)
                    
        # print(min_dist)
        return min_dist
