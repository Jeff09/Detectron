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
#
# modified by Wenhe Jia of priv-lab
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from detectron.core.config import cfg
from detectron.datasets import json_dataset
from detectron.datasets import roidb as roidb_utils
import detectron.modeling.FPN as fpn
import detectron.roi_data.fast_rcnn as fast_rcnn_roi_data
import detectron.utils.blob as blob_utils


class DistributeFpnRpnProposalsOp(object):
    def __init__(self, train, stage_num):
        self._train = train
        self._stage_num = stage_num

    def forward(self, inputs, outputs):
        """See modeling.detector.DistributeFpnRpnProposals for
        inputs/outputs documentation.
        """
        # inputs is [rois] out from decode_bbox operator
        # If training with Faster R-CNN, then inputs will additionally include
        #  + [roidb, im_info]
        _rois = inputs[0].data
        rois = remove_invalid_boxes(_rois, self._stage_num)
        # print('++++++++++++++++ DFRP Op of RCNN stage {} ++++++++++++++++++'.format(self._stage_num))
        if self._train:
            # During training we reuse the data loader code. We populate roidb
            # entries on the fly using the rois generated by RPN.
            # im_info: [[im_height, im_width, im_scale], ...]
            im_info = inputs[2].data
            im_scales = im_info[:, 2]
            roidb = blob_utils.deserialize(inputs[1].data)
            json_dataset.add_proposals(roidb, rois, im_scales, crowd_thresh=0)
            roidb_utils.add_bbox_regression_targets(roidb, self._stage_num)

            # Compute training labels for the RPN proposals; also handles
            # distributing the proposals over FPN levels
            output_blob_names = fast_rcnn_roi_data.get_cascade_fast_rcnn_blob_names(is_training=True, stage_num=self._stage_num)
            blobs = {k: [] for k in output_blob_names}
            fast_rcnn_roi_data.add_fast_rcnn_blobs(blobs, im_scales, roidb, self._stage_num)
            for i, k in enumerate(output_blob_names):
                blob_utils.py_op_copy_blob(blobs[k], outputs[i])

            # reset roidb for next rcnn stage, remove 'max_overlaps', 'max_classes', 'bbox_targets' in each roidb,
            # intialize 'boxes', 'seg_areas', 'gt_classes', 'gt_overlaps', 'box_to_gt_ind_map' only contain gt infos
            # if self._stage_num == 2:
            #     json_dataset.reset_roidb_for_next_stage(roidb)
        else:
            # For inference we have a special code path that avoids some data
            # loader overhead
            distribute(rois, None, outputs, self._train, self._stage_num)


def distribute(rois, label_blobs, outputs, train, stage_num):
    """To understand the output blob order see return value of
    detectron.roi_data.fast_rcnn.get_fast_rcnn_blob_names(is_training=False)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)

    outputs[0].reshape(rois.shape)
    outputs[0].data[...] = rois
    # Create new roi blobs for each FPN level
    # (See: modeling.FPN.add_multilevel_roi_blobs which is similar but annoying
    # to generalize to support this particular case.)
    rois_idx_order = np.empty((0, ))
    for output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)):
        idx_lvl = np.where(lvls == lvl)[0]
        blob_roi_level = rois[idx_lvl, :]
        # print('number rois of fpn{} :'.format(lvl), blob_roi_level.shape)
        outputs[output_idx + 1].reshape(blob_roi_level.shape)
        outputs[output_idx + 1].data[...] = blob_roi_level
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
    rois_idx_restore = np.argsort(rois_idx_order)
    blob_utils.py_op_copy_blob(rois_idx_restore.astype(np.int32), outputs[-1])


def remove_invalid_boxes(rois, stage_num):
    # rois with shape (N, 5), contain (batch_idx, x1, y1, x2, y2)
    ws = rois[:, 3] - rois[:, 1] + 1
    hs = rois[:, 4] - rois[:, 2] + 1
    invalid_idx_w = np.where(ws < 0)[0]
    invalid_idx_h = np.where(hs < 0)[0]
    _invalid_idx = np.append(invalid_idx_w, invalid_idx_h)
    invalid_idx = np.unique(_invalid_idx)

    if invalid_idx.shape[0] != 0:
        print('RCNN stage {} --- Distrubute And Fpn Rpn Proposals Op: input rois contain {} invalid boxes, they are:'.format(
            stage_num, invalid_idx.shape[0])
        )
        print(rois[invalid_idx, :])

    new_rois = np.delete(rois, invalid_idx, axis=0)

    return new_rois

