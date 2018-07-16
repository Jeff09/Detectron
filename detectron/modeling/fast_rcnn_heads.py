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

"""Various network "heads" for classification and bounding box prediction.

The design is as follows:

... -> RoI ----\                               /-> box cls output -> cls loss
                -> RoIFeatureXform -> box head
... -> Feature /                               \-> box reg output -> reg loss
       Map

The Fast R-CNN head produces a feature representation of the RoI for the purpose
of bounding box classification and regression. The box output module converts
the feature representation into classification and regression predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
from caffe2.python import core, workspace
import numpy as np
import detectron.modeling.FPN as fpn
import os

# ---------------------------------------------------------------------------- #
# Fast R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_fast_rcnn_outputs(model, blob_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    # Box classification layer
    model.FC(
        blob_in,
        'cls_score',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('cls_score', 'cls_prob', engine='CUDNN')
    # Box regression layer
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    model.FC(
        blob_in,
        'bbox_pred',
        dim,
        num_bbox_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )


def add_fast_rcnn_losses(model):
    """Add losses for RoI classification and bounding box regression. Only take post_nms_topN rois to calculate the loss."""
    
    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        ['cls_score', 'labels_int32'], ['cls_prob', 'loss_cls'],
        scale=model.GetLossScale()
    )
    loss_bbox = model.net.SmoothL1Loss(
        [
            'bbox_pred', 'bbox_targets', 'bbox_inside_weights',
            'bbox_outside_weights'
        ],
        'loss_bbox',
        scale=model.GetLossScale()
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
    model.Accuracy(['cls_prob', 'labels_int32'], 'accuracy_cls')
    model.AddLosses(['loss_cls', 'loss_bbox'])
    model.AddMetrics('accuracy_cls')
    return loss_gradients


def add_cascade_rcnn_outputs(model, blob_in, dim, i):
    # Box classification layer
    # blob_in: fc7
    assert i < 3    
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    if i == 0:
        model.FC(
            blob_in,
            'cls_score_stage_1',
            dim,
            model.num_classes,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        if not model.train:  # == if test
            # Only add softmax when testing; during training the softmax is combined
            # with the label cross entropy loss for numerical stability
            model.Softmax('cls_score_stage_1', 'cls_prob_stage_1', engine='CUDNN')
        # Box regression layer
        model.FC(
            blob_in,
            'bbox_pred_stage_1',
            dim,
            num_bbox_reg_classes * 4,
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
        )
    elif i == 1:
        model.FC(
            blob_in,
            'cls_score_stage_2',
            dim,
            model.num_classes,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        if not model.train:  # == if test
            # Only add softmax when testing; during training the softmax is combined
            # with the label cross entropy loss for numerical stability
            model.Softmax('cls_score_stage_2', 'cls_prob_stage_2', engine='CUDNN')
        model.FC(
            blob_in,
            'bbox_pred_stage_2',
            dim,
            num_bbox_reg_classes * 4,
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
        )
    elif i == 2:
        model.FC(
            blob_in,
            'cls_score_stage_3',
            dim,
            model.num_classes,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        if not model.train:  # == if test
            # Only add softmax when testing; during training the softmax is combined
            # with the label cross entropy loss for numerical stability
            model.Softmax('cls_score_stage_3', 'cls_prob_stage_3', engine='CUDNN')
        model.FC(
            blob_in,
            'bbox_pred_stage_3',
            dim,
            num_bbox_reg_classes * 4,
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
        )
    #workspace.RunNetOnce(model.param_init_net)
    


def add_cascade_rcnn_losses(model, thresh, i):
    assert i < 3   
    #print("Current blobs in the workspace: {}".format(workspace.Blobs()))
    #if not workspace.HasBlob(core.ScopedName('labels_int32')):
    #    print("donot have blob labels_int32")
    #    print(model.net.Proto())
    #get_labels(model, i) 
    #print(model.net.Proto())
    if i == 0:
        cls_prob_stage_1, loss_cls_stage_1 = model.net.SoftmaxWithLoss(
            ['cls_score_stage_1', 'labels_int32'], ['cls_prob_stage_1', 'loss_cls_stage_1'],
            scale=model.GetLossScale()
        )
        loss_bbox_stage_1 = model.net.SmoothL1Loss(
            [
                'bbox_pred_stage_1', 'bbox_targets', 'bbox_inside_weights',
                'bbox_outside_weights'
            ],
            'loss_bbox_stage_1',
            scale=model.GetLossScale()
        )
        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls_stage_1, loss_bbox_stage_1])
        model.Accuracy(['cls_prob_stage_1', 'labels_int32'], 'accuracy_cls_stage_1')
        model.AddLosses(['loss_cls_stage_1', 'loss_bbox_stage_1'])
        model.AddMetrics('accuracy_cls_stage_1')
    elif i == 1:             
        #_sample_labels(model, i) 
        cls_prob_stage_2, loss_cls_stage_2 = model.net.SoftmaxWithLoss(
            ['cls_score_stage_2', 'labels_stage_2'], ['cls_prob_stage_2', 'loss_cls_stage_2'],
            scale=model.GetLossScale()
        )
        loss_bbox_stage_2 = model.net.SmoothL1Loss(
            [
                'bbox_pred_stage_2', 'bbox_targets_stage_2', 'bbox_inside_weights_stage_2',
                'bbox_outside_weights_stage_2'
            ],
            'loss_bbox_stage_2',
            scale=model.GetLossScale()
        )
        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls_stage_2, loss_bbox_stage_2])
        model.Accuracy(['cls_prob_stage_2', 'labels_stage_2'], 'accuracy_cls_stage_2')
        model.AddLosses(['loss_cls_stage_2', 'loss_bbox_stage_2'])
        model.AddMetrics('accuracy_cls_stage_2')
    elif i == 2:
        #_sample_labels(model, i)
        cls_prob_stage_3, loss_cls_stage_3 = model.net.SoftmaxWithLoss(
            ['cls_score_stage_3', 'labels_stage_3'], ['cls_prob_stage_3', 'loss_cls_stage_3'],
            scale=model.GetLossScale()
        )
        loss_bbox_stage_3 = model.net.SmoothL1Loss(
            [
                'bbox_pred_stage_3', 'bbox_targets_stage_3', 'bbox_inside_weights_stage_3',
                'bbox_outside_weights_stage_3'
            ],
            'loss_bbox_stage_3',
            scale=model.GetLossScale()
        )
        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls_stage_3, loss_bbox_stage_3])
        model.Accuracy(['cls_prob_stage_3', 'labels_stage_3'], 'accuracy_cls_stage_3')
        model.AddLosses(['loss_cls_stage_3', 'loss_bbox_stage_3'])
        model.AddMetrics('accuracy_cls_stage_3')        

    return loss_gradients


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

def add_cascade_rcnn_head(model, blob_in, dim_in, spatial_scale, i):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    assert i < 3
    if i == 0:
        roi_feat_stage_1 = model.RoIFeatureTransform(
            blob_in,
            'roi_feat_stage_1',
            blob_rois='rois', # post_nms_topN 
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=roi_size,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
            spatial_scale=spatial_scale
        )
        model.FC(roi_feat_stage_1, 'fc6_stage_1', dim_in * roi_size * roi_size, hidden_dim, weight_init=gauss_fill(0.01), bias_init=const_fill(0.0))
        model.Relu('fc6_stage_1', 'fc6_stage_1')
        model.FC('fc6_stage_1', 'fc7_stage_1', hidden_dim, hidden_dim, weight_init=gauss_fill(0.01), bias_init=const_fill(0.0))
        model.Relu('fc7_stage_1', 'fc7_stage_1')
        output = 'fc7_stage_1'
    elif i == 1:
        # map bbox_pred_stage_1 to fpn conv f roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        #add_multilevel_pred_box_blob(model, blob_in, "bbox_pred_stage_1")
        if model.train:
            # Add op that generates training labels for in-network RPN proposals
            model.GenerateProposalLabels_cascade_rcnn(['bbox_pred_stage_1', 'roidb', 'im_info'], i)
        else:
            # Alias rois to rpn_rois for inference
            model.net.Alias('bbox_pred_stage_1', 'rois')
        roi_feat_stage_2 = model.RoIFeatureTransform(
            blob_in,
            'roi_feat_stage_2',
            blob_rois='bbox_pred_stage_1', # post_nms_topN 
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=roi_size,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
            spatial_scale=spatial_scale
        )
        model.FC(roi_feat_stage_2, 'fc6_stage_2', dim_in * roi_size * roi_size, hidden_dim, weight_init=gauss_fill(0.01), bias_init=const_fill(0.0))
        model.Relu('fc6_stage_2', 'fc6_stage_2')
        model.FC('fc6_stage_2', 'fc7_stage_2', hidden_dim, hidden_dim, weight_init=gauss_fill(0.01), bias_init=const_fill(0.0))
        model.Relu('fc7_stage_2', 'fc7_stage_2')
        output = 'fc7_stage_2'
    elif i == 2:
        #add_multilevel_pred_box_blob(model, blob_in, 'bbox_pred_stage_2')
        if model.train:
            # Add op that generates training labels for in-network RPN proposals
            model.GenerateProposalLabels_cascade_rcnn(['bbox_pred_stage_2', 'roidb', 'im_info'], i)
        else:
            # Alias rois to rpn_rois for inference
            model.net.Alias('bbox_pred_stage_2', 'rois')
        roi_feat_stage_3 = model.RoIFeatureTransform(
            blob_in,
            'roi_feat_stage_3',
            blob_rois='bbox_pred_stage_2',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=roi_size,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
            spatial_scale=spatial_scale
        )
        model.FC(roi_feat_stage_3, 'fc6_stage_3', dim_in * roi_size * roi_size, hidden_dim, weight_init=gauss_fill(0.01), bias_init=const_fill(0.0))
        model.Relu('fc6_stage_3', 'fc6_stage_3')
        model.FC('fc6_stage_3', 'fc7_stage_3', hidden_dim, hidden_dim, weight_init=gauss_fill(0.01), bias_init=const_fill(0.0))
        model.Relu('fc7_stage_3', 'fc7_stage_3')
        output = 'fc7_stage_3'
    return output, hidden_dim

def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois', # post_nms_topN
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC(roi_feat, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', hidden_dim, hidden_dim)
    model.Relu('fc7', 'fc7')
    return 'fc7', hidden_dim

def add_roi_Xconv2fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 2fc head"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim
    
    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim, weight_init=gauss_fill(0.01), bias_init=const_fill(0.0))
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', fc_dim, fc_dim, weight_init=gauss_fill(0.01), bias_init=const_fill(0.0))
    model.Relu('fc7', 'fc7')
    return 'fc7', fc_dim


def add_roi_Xconv1fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim

def add_roi_Xconv2fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 2fc head"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            stride=1, pad=1,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim
    
    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim, weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0))
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', fc_dim, fc_dim, weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0))
    model.Relu('fc7', 'fc7')
    return 'fc7', fc_dim


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, with GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in, 'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            group_gn=get_group_gn(hidden_dim),
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim
