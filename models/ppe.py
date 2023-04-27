from datasets import points_utils
from models import base_model
from models.backbone.pointnet import MiniPointNet, SegPointNet
import torch
from torch import nn
import torch.nn.functional as F

from torchmetrics import Accuracy
import numpy as np
from attention import encoder, decoder

np.set_printoptions(threshold=10000)


class PPE(base_model.MotionBaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.seg_acc = Accuracy(num_classes=2, average='none')

        self.box_aware = getattr(config, 'box_aware', False)  # 使用box_aware
        self.use_motion_cls = getattr(config, 'use_motion_cls',  # 使用运动状态估计
                                      True)  # target segmentation and motion state classification
        self.use_second_stage = getattr(config, 'use_second_stage', True)  # 是否使用两步骤算法
        self.use_prev_refinement = getattr(config, 'use_prev_refinement', True)
 
        self.seg_pointnet = SegPointNet(input_channel=3 + 1 + 1 + (9 if self.box_aware else 0),
                                        per_point_mlp1=[64, 64, 64, 128, 1024],
                                        per_point_mlp2=[512, 256, 128, 128],
                                        output_size=2 + (9 if self.box_aware else 0))
        self.mini_pointnet = MiniPointNet(input_channel=3 + 1 + (9 if self.box_aware else 0),
                                          per_point_mlp=[64, 128, 256, 512],
                                          hidden_mlp=[512, 256],
                                          output_size=-1)
        if self.use_second_stage:
            self.mini_pointnet2 = MiniPointNet(input_channel=3 + (9  if self.box_aware else 0),
                                               per_point_mlp=[64, 128, 256, 512],
                                               hidden_mlp=[512, 256],
                                               output_size=-1)

            self.box_mlp = nn.Sequential(nn.Linear(256, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 4))
        if self.use_prev_refinement:
            self.final_mlp = nn.Sequential(nn.Linear(256, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 4))
        if self.use_motion_cls:
            self.motion_state_mlp = nn.Sequential(nn.Linear(256, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 2))
            self.motion_acc = Accuracy(num_classes=2, average='none')

        self.motion_mlp = nn.Sequential(nn.Linear(256, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 4))
        self.encoder=encoder
        #self.decoder=decoder

    def forward(self, input_dict):
        """
        Args:
            input_dict: {
            "points": (B,N,3+1+1) batch,number of point,(x,y,z,t,si=>3+1+1)
            "candidate_bc": (B,N,9) the bbox and nine points of bbox

        }
        dict:10
        Returns: B,4

        """

        output_dict = {}

        #  swap dimension
        x = input_dict["points"].transpose(1, 2)  # [B,5,2048]  # x y z θ score
        if self.box_aware:
            candidate_bc = input_dict["candidate_bc"].transpose(1, 2)  # [B,9,2048]

            x = torch.cat([x, candidate_bc], dim=1)  # [B,14,2048]

        B, _, N = x.shape  # [B,14,2048] B batch N number 14=9+5
        # ============================================
        seg_out = self.seg_pointnet(x)  # [B,11,2048]
        seg_logits = seg_out[:, :2, :]  # [B,2 ,2048] # B,2,N motion state
        x = x.permute(0, 2, 1)
        seg_out = seg_out.permute(2, 0, 1)

        seg_out[:, :, 2:] = self.encoder(seg_out[:, :, 2:],
                                         query_pos=x[:, :, :3])
        x = x.permute(0, 2, 1)
        seg_out = seg_out.permute(1, 2, 0)

        # ========================================================================================
        pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)  # B,1,N
        # ============================================
        mask_points = x[:, :4, :] * pred_cls  # [B,4,2048]
        mask_xyz_t0 = mask_points[:, :3, :N // 2]  # [B,3,1024] # B,3,N//2
        mask_xyz_t1 = mask_points[:, :3, N // 2:]  # [B,3,1024]

        if self.box_aware:
            pred_bc = seg_out[:, 2:, :]  # [B,18,2048]
            mask_pred_bc = pred_bc * pred_cls  # [B,18,2048]

            mask_points = torch.cat([mask_points, mask_pred_bc], dim=1)  # [B,22,2048]
            output_dict['pred_bc'] = pred_bc.transpose(1, 2)
        # pointnet extract feature
        point_feature = self.mini_pointnet(mask_points)  # [B,256]
        # print('=============================')
        # motion state prediction(🔺x，🔺y，🔺z，🔺θ)
        motion_pred = self.motion_mlp(point_feature)  # B,4

        if self.use_motion_cls:
            motion_state_logits = self.motion_state_mlp(point_feature)  # B,2
            motion_mask = torch.argmax(motion_state_logits, dim=1, keepdim=True)  # B,1
            motion_pred_masked = motion_pred * motion_mask  # (B,4)
            output_dict['motion_cls'] = motion_state_logits
        else:
            motion_pred_masked = motion_pred

        if self.use_prev_refinement:
            prev_boxes = self.final_mlp(point_feature)  # [B,4]
            output_dict["estimation_boxes_prev"] = prev_boxes[:, :4]
        else:
            prev_boxes = torch.zeros_like(motion_pred)

        # 1st stage prediction
        aux_box = points_utils.get_offset_box_tensor(prev_boxes, motion_pred_masked)  # previous (B,4)

        # 2nd stage refinement
        if self.use_second_stage:
            mask_xyz_t0_2_t1 = points_utils.get_offset_points_tensor(mask_xyz_t0.transpose(1, 2),  # [B,3,1024]
                                                                     prev_boxes[:, :4],
                                                                     motion_pred_masked).transpose(1, 2)
            mask_xyz_t01 = torch.cat([mask_xyz_t0_2_t1, mask_xyz_t1], dim=-1)  # B,3,2048

            # transform to the aux_box coordinate system
            mask_xyz_t01 = points_utils.remove_transform_points_tensor(mask_xyz_t01.transpose(1, 2),  # [B,3,2048]
                                                                       aux_box).transpose(1, 2)

            if self.box_aware:
                mask_xyz_t01 = torch.cat([mask_xyz_t01, mask_pred_bc], dim=1)  # [B,12,2048]
            output_offset = self.box_mlp(self.mini_pointnet2(mask_xyz_t01))  # B,4
            output = points_utils.get_offset_box_tensor(aux_box, output_offset)
            output_dict["estimation_boxes"] = output  # 2nd box
        else:
            output_dict["estimation_boxes"] = aux_box
        output_dict.update({"seg_logits": seg_logits,  # motion state
                            "motion_pred": motion_pred,  # RTM
                            'aux_estimation_boxes': aux_box,  # 1st box
                            })

        return output_dict

    def compute_loss(self, data, output):
        loss_total = 0.0
        loss_dict = {}
        aux_estimation_boxes = output['aux_estimation_boxes']  # B,4
        motion_pred = output['motion_pred']  # B,4
        seg_logits = output['seg_logits']  # B,2,2048

        with torch.no_grad():
            seg_label = data['seg_label'].long()
            box_label = data['box_label']
            box_label_prev = data['box_label_prev']
            motion_label = data['motion_label']
            motion_state_label = data['motion_state_label'].long()
            center_label = box_label[:, :3]
            angle_label = torch.sin(box_label[:, 3])
            center_label_prev = box_label_prev[:, :3]
            angle_label_prev = torch.sin(box_label_prev[:, 3])
            center_label_motion = motion_label[:, :3]
            angle_label_motion = torch.sin(motion_label[:, 3])

        loss_seg = F.cross_entropy(seg_logits, seg_label, weight=torch.tensor([0.5, 2.0]).cuda())
        # =========================================================================================

        if self.use_motion_cls:
            motion_cls = output['motion_cls']  # B,2
            loss_motion_cls = F.cross_entropy(motion_cls, motion_state_label)
            # =====================================================================================
            loss_total += loss_motion_cls * self.config.motion_cls_seg_weight
            loss_dict['loss_motion_cls'] = loss_motion_cls

            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion, reduction='none')
            loss_center_motion = (motion_state_label * loss_center_motion.mean(dim=1)).sum() / (
                    motion_state_label.sum() + 1e-6)
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion, reduction='none')
            loss_angle_motion = (motion_state_label * loss_angle_motion).sum() / (motion_state_label.sum() + 1e-6)
        else:
            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion)
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion)

        if self.use_second_stage:
            estimation_boxes = output['estimation_boxes']  # B,4
            loss_center = F.smooth_l1_loss(estimation_boxes[:, :3], center_label)
            loss_angle = F.smooth_l1_loss(torch.sin(estimation_boxes[:, 3]), angle_label)
            loss_total += 1 * (loss_center * self.config.center_weight + loss_angle * self.config.angle_weight)
            loss_dict["loss_center"] = loss_center
            loss_dict["loss_angle"] = loss_angle
        if self.use_prev_refinement:
            estimation_boxes_prev = output['estimation_boxes_prev']  # B,4
            loss_center_prev = F.smooth_l1_loss(estimation_boxes_prev[:, :3], center_label_prev)
            loss_angle_prev = F.smooth_l1_loss(torch.sin(estimation_boxes_prev[:, 3]), angle_label_prev)
            loss_total += (loss_center_prev * self.config.center_weight + loss_angle_prev * self.config.angle_weight)
            loss_dict["loss_center_prev"] = loss_center_prev
            loss_dict["loss_angle_prev"] = loss_angle_prev

        loss_center_aux = F.smooth_l1_loss(aux_estimation_boxes[:, :3], center_label)

        loss_angle_aux = F.smooth_l1_loss(torch.sin(aux_estimation_boxes[:, 3]), angle_label)

        loss_total += loss_seg * self.config.seg_weight \
                      + 1 * (loss_center_aux * self.config.center_weight + loss_angle_aux * self.config.angle_weight) \
                      + 1 * (
                              loss_center_motion * self.config.center_weight + loss_angle_motion * self.config.angle_weight)
        loss_dict.update({
            "loss_total": loss_total,  # loss total
            "loss_seg": loss_seg,  # segementation
            "loss_center_aux": loss_center_aux,  # 1st estimation
            "loss_center_motion": loss_center_motion,  # 1st motion state estimation
            "loss_angle_aux": loss_angle_aux,
            "loss_angle_motion": loss_angle_motion,
        })
        if self.box_aware:
            prev_bc = data['prev_bc']
            this_bc = data['this_bc']
            bc_label = torch.cat([prev_bc, this_bc], dim=1)
            pred_bc = output['pred_bc']
            # print(pred_bc.shape)
            loss_bc = F.smooth_l1_loss(pred_bc[:,:,:9], bc_label)
            loss_total += loss_bc * self.config.bc_weight
            loss_dict.update({
                "loss_total": loss_total,
                "loss_bc": loss_bc
            })

        return loss_dict

    # increase the training loop
    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: {
            "points": stack_frames, (B,N,3+9+1)
            "seg_label": stack_label,
            "box_label": np.append(this_gt_bb_transform.center, theta),
            "box_size": this_gt_bb_transform.wlh
        }
        Returns:

        """
        output = self(batch)
        loss_dict = self.compute_loss(batch, output)
        loss = loss_dict['loss_total']

        # log
        seg_acc = self.seg_acc(torch.argmax(output['seg_logits'], dim=1, keepdim=False), batch['seg_label'])
        self.log('seg_acc_background/train', seg_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('seg_acc_foreground/train', seg_acc[1], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if self.use_motion_cls:
            motion_acc = self.motion_acc(torch.argmax(output['motion_cls'], dim=1, keepdim=False),
                                         batch['motion_state_label'])
            self.log('motion_acc_static/train', motion_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('motion_acc_dynamic/train', motion_acc[1], on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

        log_dict = {k: v.item() for k, v in loss_dict.items()}

        self.logger.experiment.add_scalars('loss', log_dict,
                                           global_step=self.global_step)
        return loss

