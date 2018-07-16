# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Construct minibatches for Cascade R-CNN training. Handles the minibatch blobs
that are specific to Cascade R-CNN. Other blobs that are generic to RPN, etc.
are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import numpy.random as npr

from detectron.core.config import cfg
import detectron.modeling.FPN as fpn
import detectron.roi_data.keypoint_rcnn as keypoint_rcnn_roi_data
import detectron.roi_data.mask_rcnn as mask_rcnn_roi_data
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

logger = logging.getLogger(__name__)


def get_cascade_rcnn_stage_2_blob_names(is_training=True):
    """Cascade R-CNN blob names."""
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)
    blob_names = ['rois_stage_2']
    if is_training and cfg.TRAIN.CASCADE_RCNN:
        # labels_int32 blob: R categorical labels in [0, ..., K] for K
        # foreground classes plus background
        blob_names += ['labels_stage_2']
    if is_training and cfg.TRAIN.CASCADE_RCNN:
        # bbox_targets blob: R bounding-box regression targets with 4
        # targets per class
        blob_names += ['bbox_targets_stage_2']
        # bbox_inside_weights blob: At most 4 targets per roi are active
        # this binary vector sepcifies the subset of active targets
        blob_names += ['bbox_inside_weights_stage_2']
        blob_names += ['bbox_outside_weights_stage_2']
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ['rois_stage_2_fpn' + str(lvl)]
        blob_names += ['rois_stage_2_idx_restore_int32']
        
    return blob_names

def get_cascade_rcnn_stage_3_blob_names(is_training=True):
    """Cascade R-CNN blob names."""
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)
    blob_names = ['rois_stage_3']
    if is_training and cfg.TRAIN.CASCADE_RCNN:
        # labels_int32 blob: R categorical labels in [0, ..., K] for K
        # foreground classes plus background
        blob_names += ['labels_stage_3']
    if is_training and cfg.TRAIN.CASCADE_RCNN:
        # bbox_targets blob: R bounding-box regression targets with 4
        # targets per class
        blob_names += ['bbox_targets_stage_3']
        # bbox_inside_weights blob: At most 4 targets per roi are active
        # this binary vector sepcifies the subset of active targets
        blob_names += ['bbox_inside_weights_stage_3']
        blob_names += ['bbox_outside_weights_stage_3']
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ['rois_stage_3_fpn' + str(lvl)]
        blob_names += ['rois_stage_3_idx_restore_int32']
        
    return blob_names

    

def add_cascade_rcnn_stage_2_blobs(blobs, im_scales, roidb):
    """Add blobs needed for training Cascade R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        frcn_blobs = _sample_rois_stage_2(entry, im_scales[im_i], im_i)
        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois(blobs, 2)

    # Perform any final work and validity checks after the collating blobs for
    # all minibatch images
    valid = True

    return valid

def add_cascade_rcnn_stage_3_blobs(blobs, im_scales, roidb):
    """Add blobs needed for training Cascade R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        frcn_blobs = _sample_rois_stage_3(entry, im_scales[im_i], im_i)
        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois(blobs, 3)

    # Perform any final work and validity checks after the collating blobs for
    # all minibatch images
    valid = True

    return valid


def _sample_rois_stage_2(roidb, im_scale, batch_idx):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
    max_overlaps_stage_2 = roidb['max_overlaps_stage_2']

    # Select foreground RoIs as those with >= FG_THRESH overlap, cascade rcnn has 3 stages:0.5, 0.6, 0.7

    fg_inds_stage_2 = np.where(max_overlaps_stage_2 >= cfg.TRAIN.CASCADE_THRESHOLDS[1])[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_stage_2_per_this_image = np.minimum(fg_rois_per_image, fg_inds_stage_2.size)

    # Sample foreground regions without replacement    
    if fg_inds_stage_2.size > 0:
        fg_inds_stage_2 = npr.choice(
            fg_inds_stage_2, size=fg_rois_stage_2_per_this_image, replace=False
        )    

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    
    bg_inds_stage_2 = np.where(
        (max_overlaps_stage_2 < cfg.TRAIN.BG_THRESH_HI) &
        (max_overlaps_stage_2 >= cfg.TRAIN.BG_THRESH_LO)
    )[0]
    
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_stage_2_per_this_image = rois_per_image - fg_rois_stage_2_per_this_image
    bg_rois_stage_2_per_this_image = np.minimum(bg_rois_stage_2_per_this_image, bg_inds_stage_2.size)
    
    # Sample foreground regions without replacement
    if bg_inds_stage_2.size > 0:
        bg_inds_stage_2 = npr.choice(
            bg_inds_stage_2, size=bg_rois_stage_2_per_this_image, replace=False
        )
    

    # The indices that we're selecting (both fg and bg)
    keep_inds_stage_2 = np.append(fg_inds_stage_2, bg_inds_stage_2)
    
    # Label is the class each RoI has max overlap with
    sampled_labels_stage_2 = roidb['max_classes_stage_2'][keep_inds_stage_2]
    

    sampled_labels_stage_2[fg_rois_stage_2_per_this_image:] = 0  # Label bg RoIs with class 0
    sampled_boxes_stage_2 = roidb['boxes_stage_2'][keep_inds_stage_2]
    
    bbox_targets_stage_2, bbox_inside_weights_stage_2 = _expand_bbox_targets(
        roidb['bbox_targets_stage_2'][keep_inds_stage_2, :]
    )
    bbox_outside_weights_stage_2 = np.array(
        bbox_inside_weights_stage_2 > 0, dtype=bbox_inside_weights_stage_2.dtype
    )

    # Scale rois and format as (batch_idx, x1, y1, x2, y2)    
    sampled_rois_stage_2 = sampled_boxes_stage_2 * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((sampled_rois_stage_2.shape[0], 1))
    sampled_rois_stage_2 = np.hstack((repeated_batch_idx, sampled_rois_stage_2))

    # Base Fast R-CNN blobs
    blob_dict = dict(
        labels_stage_2=sampled_labels_stage_2.astype(np.int32, copy=False),
        rois_stage_2=sampled_rois_stage_2,
        bbox_targets_stage_2=bbox_targets_stage_2,
        bbox_inside_weights_stage_2=bbox_inside_weights_stage_2,
        bbox_outside_weights_stage_2=bbox_outside_weights_stage_2
    )

    return blob_dict

def _sample_rois_stage_3(roidb, im_scale, batch_idx):

    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
    max_overlaps_stage_3 = roidb['max_overlaps_stage_3']

    fg_inds_stage_3 = np.where(max_overlaps_stage_3 >= cfg.TRAIN.CASCADE_THRESHOLDS[2])[0]

    fg_rois_stage_3_per_this_image = np.minimum(fg_rois_per_image, fg_inds_stage_3.size)

    if fg_inds_stage_3.size > 0:
        fg_inds_stage_3 = npr.choice(
            fg_inds_stage_3, size=fg_rois_stage_3_per_this_image, replace=False
        )
    
    bg_inds_stage_3 = np.where(
        (max_overlaps_stage_3 < cfg.TRAIN.BG_THRESH_HI) &
        (max_overlaps_stage_3 >= cfg.TRAIN.BG_THRESH_LO)
    )[0]

    bg_rois_stage_3_per_this_image = rois_per_image - fg_rois_stage_3_per_this_image
    bg_rois_stage_3_per_this_image = np.minimum(bg_rois_stage_3_per_this_image, bg_inds_stage_3.size)
 
    if bg_inds_stage_3.size > 0:
        bg_inds_stage_3 = npr.choice(
            bg_inds_stage_3, size=bg_rois_stage_3_per_this_image, replace=False
        )

    keep_inds_stage_3 = np.append(fg_inds_stage_3, bg_inds_stage_3)
    
    sampled_labels_stage_3 = roidb['max_classes_stage_3'][keep_inds_stage_3]

    sampled_labels_stage_3[fg_rois_stage_3_per_this_image:] = 0  # Label bg RoIs with class 0
    sampled_boxes_stage_3 = roidb['boxes_stage_3'][keep_inds_stage_3]

    bbox_targets_stage_3, bbox_inside_weights_stage_3 = _expand_bbox_targets(
        roidb['bbox_targets_stage_3'][keep_inds_stage_3, :]
    )
    bbox_outside_weights_stage_3 = np.array(
        bbox_inside_weights_stage_3 > 0, dtype=bbox_inside_weights_stage_3.dtype
    )

    sampled_rois_stage_3 = sampled_boxes_stage_3 * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((sampled_rois_stage_3.shape[0], 1))
    sampled_rois_stage_3 = np.hstack((repeated_batch_idx, sampled_rois_stage_3))

    blob_dict = dict(
        labels_stage_3=sampled_labels_stage_3.astype(np.int32, copy=False),
        rois_stage_3=sampled_rois_stage_3,
        bbox_targets_stage_3=bbox_targets_stage_3,
        bbox_inside_weights_stage_3=bbox_inside_weights_stage_3,
        bbox_outside_weights_stage_3=bbox_outside_weights_stage_3
    )
    return blob_dict


def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    bbox_target_data: N x 5K blob of class and bbox targets

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        num_bbox_reg_classes = 2  # bg and fg

    clss = bbox_target_data[:, 0]
    bbox_targets = blob_utils.zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = blob_utils.zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


def _add_multilevel_rois(blobs, i):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL

    def _distribute_rois_over_fpn_levels(rois_blob_name):
        """Distribute rois over the different FPN levels."""
        # Get target level for each roi
        # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
        # the box coordinates from columns 1:5
        target_lvls = fpn.map_rois_to_fpn_levels(
            blobs[rois_blob_name][:, 1:5], lvl_min, lvl_max
        )
        # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
        fpn.add_multilevel_roi_blobs(
            blobs, rois_blob_name, blobs[rois_blob_name], target_lvls, lvl_min,
            lvl_max
        )
    if i == 2:
        _distribute_rois_over_fpn_levels('rois_stage_2')
    elif i == 3:
        _distribute_rois_over_fpn_levels('rois_stage_3')
    else:
        raise ValueError("i > 3.")
