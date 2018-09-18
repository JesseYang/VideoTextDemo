import cv2
from scipy import misc
from easydict import EasyDict as edict
from tensorpack import *
import numpy as np
# list split utils
import random
from matplotlib import pyplot as plt
import math
from operator import itemgetter
import uuid
from itertools import *
import time
import functools
import os
from copy import deepcopy
import pdb
from PIL import Image
from tensorpack import *
from classify_frames.train import Model as Model_classify_frames
from detect_table.train import Model as Model_detect_table
from detect.train import VGGFSSD as Model_detect_text_area
from segment_lines.train import Model as Model_segment_lines
# from recognize_sequences.train import Model as Model_recognize_sequences
from recognize_english.train import Model as Model_recognize_english
from recognize_chinese.train import Model as Model_recognize_chinese
from recognize_korean.train import Model as Model_recognize_korean
from recognize_japanese.train import Model as Model_recognize_japanese
# Mapper  Model_recognize_sequences cfg_recognize_sequences
# import configs
from classify_frames.cfgs.config import cfg as cfg_classify_frames
from detect_table.cfgs.config import cfg as cfg_detect_table
from detect.cfgs.config import cfg as cfg_detect_text_area
from segment_lines.cfgs.config import cfg as cfg_segment_lines
# from recognize_sequences.cfgs.config import cfg as cfg_eng_recognize_sequences
from recognize_english.cfgs.config import cfg as cfg_recognize_english
from recognize_chinese.cfgs.config import cfg as cfg_recognize_chinese
from recognize_korean.cfgs.config import cfg as cfg_recognize_korean
from recognize_japanese.cfgs.config import cfg as cfg_recognize_japanese
# from recognize_sequences.mapper import Mapper
from recognize_english.mapper import Mapper as Mapper_english
from recognize_chinese.mapper import Mapper as Mapper_chinese
from recognize_korean.mapper import Mapper as Mapper_korean
from recognize_japanese.mapper import Mapper as Mapper_japanese

from cfgs.config import cfg
from detect.utils_bak import postprocess as detect_postprocess
## lm
# from lm_prefix_beam_search import prefix_beam_search
# import kenlm



time_record = {}
def timethis(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        t_start = time.time()
        res = func(*args, **kw)
        t_end = time.time()
        time_record[func.__name__]= t_end - t_start
        return res
    return wrapper

def cap_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("total_frame num ", str(total_frame))
    if total_frame <= 0:
        return []
    # return [cap.read()[1] for _ in range(total_frame)]

    frame_list = []
    
    # try:
    #     for i in range(total_frame):
    #         fr = cap.read()[1]
    #         frame_list.append(fr)
    #         # print(len(frame_list))
    # except Exception as e:
    #     print("video file error")
    #     frame_list = []
    # finally:
    #     print(len(frame_list))
    #     return frame_list
    try:
        ret, fr = cap.read()
        while ret:
            # print(fr)
            frame_list.append(fr)
            ret, fr = cap.read()
    except Exception as e:
        print("video file error")
        frame_list = []
    finally:
        print(len(frame_list))
        return frame_list

def batch_data(data, batch_size):
    len_data = len(data)
    batch_num = len_data // batch_size + 1 if len_data % batch_size else len_data // batch_size
    print('data will be splitted into {} batches'.format(batch_num))
    batched_data = np.array_split(data, batch_num)
    return batched_data

def classify_frames(inputs, pred_func):
    def preprocess(inputs):
        total_frame = len(inputs)

        assert len(inputs) >= len(cfg_classify_frames.frame_extract_pattern)
        # get resized gray-level frames
        # for i, frame in enumerate(inputs):
            # print(frame == None)
            # cv2.imwrite(os.path.join('output', str(i)+".jpg"), frame)
        resized_frames = [cv2.cvtColor(cv2.resize(i, (224, 224)), cv2.COLOR_BGR2GRAY) for i in inputs]
        # generate tensors in shape of (224, 224, c)
        tensors = []
      
        for frame_idx in range(total_frame):
            if frame_idx - margin < 0 or frame_idx + margin >= total_frame:
                continue
            selected_frames = resized_frames[frame_idx - margin:frame_idx + margin + 1]
            # select frames within margins, with shape(pattern_length, 224, 224)
            tensor = np.asarray(selected_frames)
            tensor = tensor.swapaxes(0,2)
            tensors.append(tensor)
        # generate tensors with shape (224, 224, pattern_length)

        return tensors

    def postprocess(preds):
        # 2-class probabilities to predictions
        label_pred = np.argmax(preds, axis = 1)
        # pad head and tail with `edge` mode
        label_pred = np.pad(label_pred, (margin, margin), mode='edge')
        # fill gaps smaller than `max_gap`
        label = [[idx, lb] for idx, lb in enumerate(label_pred)]
        label = [[f,list(g)] for f,g in groupby(label, lambda x:x[1])]
        label = [[i[0], len(i[1])] for i in label]

        chip_l = -1
        chip_r = -1
        inchip = False
        for i in range(len(label)):
            if inchip and label[i][1] >= 10:
                chip_r = i - 1
                inchip = False
                left, right = None, None
                if chip_l - 1 >= 0:
                    left = label[chip_l - 1][0]
                if chip_r + 1 < len(label):
                    right = label[chip_r + 1][0]
                if left is not None and right is not None:
                    if left==right==1:
                        for j in range(chip_l, chip_r + 1):
                            label[j][0] = 1
                    else:
                        for j in range(chip_l, chip_r + 1):
                            label[j][0] = 0
                elif left is not None and right is None:
                    for j in range(chip_l, chip_r + 1):
                        label[j][0] = left
                elif left is None and right is not None:
                    for j in range(chip_l, chip_r + 1):
                        label[j][0] = right
            elif not inchip and label[i][1] < 10:
                chip_l = i
                inchip = True
        new_label = []
        for i in label:
            new_label.extend([i[0]]*i[1])
        return np.array(new_label)
    
    margin = len(cfg_classify_frames.frame_extract_pattern) // 2
    batch_size = cfg_classify_frames.batch_size
    preprocessed = preprocess(inputs)
    batches = batch_data(preprocessed, batch_size)
    batched_preds = [pred_func([i])[0] for i in batches]
    preds = list(np.vstack(batched_preds))
    postprocessed = postprocess(preds)
    
    return postprocessed

def extract_frames(inputs, label):
    
    # extract valid frame indexes
    # frame_idx = [idx for idx, lb in enumerate(label) if lb]
    # print(len(frame_idx))
    # print("----")
    frame_idx = []
    list_1 = []
    list_0 = []
    for idx, lb in enumerate(label):
        if lb:
            list_1.append(idx)
        else:
            if len(list_1) <=5:
                list_1.clear()
                continue
            else:
                corr_num = math.floor(len(list_1)*0.6)
                frame_idx.extend(list_1[(len(list_1)//2-corr_num//2):(len(list_1)//2+corr_num//2+1)])
                list_1.clear()
    if len(list_1) >= 5:
        corr_num = math.floor(len(list_1)*0.6)
        frame_idx.extend(list_1[(len(list_1)//2-corr_num//2):(len(list_1)//2+corr_num//2+1)])
        list_1.clear()
    # print(len(label))
    # print(frame_idx)

    frame_idxss = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(frame_idx), lambda x: x[0]-x[1])]
    max_blurry_idxs = []
    # for each piece, find max-blurry index
    for frame_idxs in frame_idxss:
        max_blurry = 0
        max_blurry_idx = None
        for i in frame_idxs:
            img_h, img_w = inputs[i].shape[0:2]
            img_blur = inputs[i][img_h//2-img_h//4:img_h//2+img_h//4, img_w//2-img_w//4:img_w//2+img_w//4,:]
            # blurry = cv2.Laplacian(inputs[i], cv2.CV_64F).var()
            blurry = cv2.Laplacian(img_blur, cv2.CV_64F).var()
            # print(i, blurry)
            if max_blurry < blurry:
                max_blurry = blurry
                max_blurry_idx = i

        max_blurry_idxs.append(max_blurry_idx)
    # collect max-blurry frames and their indexes
    outputs = [[inputs[i], {'frame_idx': i}] for i in max_blurry_idxs]

    return outputs

def detect_table(inputs, pred_func, enlarge_ratio=1):
    def preprocess(inputs):
        # resize images and convert BGR to RGB
        rgb_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in inputs]
        resized_imgs = [cv2.resize(i, (img_w, img_h)) for i in rgb_imgs]
        # spec_mask = [np.zeros((cfg_detect_table.n_boxes, img_w // 32, img_h // 32), dtype=float) == 0 for _ in rgb_imgs]
        return resized_imgs


    def postprocess(predictions, img, det_th=None):
        def non_maximum_suppression(boxes, overlapThresh):
            # if there are no boxes, return an empty list
            if len(boxes) == 0:
                return []
            boxes = np.asarray(boxes).astype("float")

            # initialize the list of picked indexes 
            pick = []

            # grab the coordinates of the bounding boxes
            conf = boxes[:,0]
            x1 = boxes[:,1]
            y1 = boxes[:,2]
            x2 = boxes[:,3]
            y2 = boxes[:,4]

            # compute the area of the bounding boxes and sort the bounding
            # boxes by the bottom-right y-coordinate of the bounding box
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            idxs = np.argsort(conf)

            # keep looping while some indexes still remain in the indexes
            # list
            while len(idxs) > 0:
                # grab the last index in the indexes list and add the
                # index value to the list of picked indexes
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)

                # find the largest (x, y) coordinates for the start of
                # the bounding box and the smallest (x, y) coordinates
                # for the end of the bounding box
                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])

                # compute the width and height of the bounding box
                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)

                intersection = w * h
                union = area[idxs[:last]] + area[idxs[last]] - intersection

                # compute the ratio of overlap
                # overlap = (w * h) / area[idxs[:last]]
                overlap = intersection / union

                # delete all indexes from the index list that have
                idxs = np.delete(idxs, np.concatenate(([last],
                    np.where(overlap > overlapThresh)[0])))

            # return only the bounding boxes that were picked using the
            # integer data type
            return boxes[pick].astype("float")
        ori_height, ori_width = img.shape[:2]
        cfg = cfg_detect_table
        [pred_x, pred_y, pred_w, pred_h, pred_conf, pred_prob] = predictions

        _, box_n, klass_num, grid_h, grid_w = pred_prob.shape

        pred_conf_tile = np.tile(pred_conf, (1, 1, klass_num, 1, 1))
        klass_conf = pred_prob * pred_conf_tile

        width_rate = ori_width / float(cfg.img_w)
        height_rate = ori_height / float(cfg.img_h)

        boxes = {}
        for n in range(box_n):
            for gh in range(grid_h):
                for gw in range(grid_w):

                    k = np.argmax(klass_conf[0, n, :, gh, gw])
                    if klass_conf[0, n, k, gh, gw] < (det_th or cfg.det_th):
                        continue

                    anchor = cfg.anchors[n]
                    w = pred_w[0, n, 0, gh, gw]
                    h = pred_h[0, n, 0, gh, gw]
                    x = pred_x[0, n, 0, gh, gw]
                    y = pred_y[0, n, 0, gh, gw]

                    center_w_cell = gw + x
                    center_h_cell = gh + y
                    box_w_cell = np.exp(w) * anchor[0]
                    box_h_cell = np.exp(h) * anchor[1]

                    center_w_pixel = center_w_cell * 32
                    center_h_pixel = center_h_cell * 32
                    box_w_pixel = box_w_cell * 32
                    box_h_pixel = box_h_cell * 32

                    xmin = float(center_w_pixel - box_w_pixel // 2)
                    ymin = float(center_h_pixel - box_h_pixel // 2)
                    xmax = float(center_w_pixel + box_w_pixel // 2)
                    ymax = float(center_h_pixel + box_h_pixel // 2)
                    xmin = np.max([xmin, 0]) * width_rate
                    ymin = np.max([ymin, 0]) * height_rate
                    xmax = np.min([xmax, float(cfg.img_w)]) * width_rate
                    ymax = np.min([ymax, float(cfg.img_h)]) * height_rate

                    klass = cfg.classes_name[k]
                    if klass not in boxes.keys():
                        boxes[klass] = []

                    box = [klass_conf[0, n, k, gh, gw], xmin, ymin, xmax, ymax]

                    boxes[klass].append(box)

        # do non-maximum-suppresion
        nms_boxes = {}
        if cfg.nms == True:
            for klass, k_boxes in boxes.items():
                k_boxes = non_maximum_suppression(k_boxes, cfg.nms_th)
                nms_boxes[klass] = k_boxes
        else:
            nms_boxes = boxes

        output = []
        for klass, k_boxes in nms_boxes.items():
            for box_idx, each_box in enumerate(k_boxes):
                [conf, xmin, ymin, xmax, ymax] = each_box
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                coor = [xmin, ymin, xmax, ymax]

                xcenter = (xmin + xmax) / 2
                ycenter = (ymin + ymax) / 2
                width = (xmax - xmin) * enlarge_ratio
                height = (ymax - ymin) * enlarge_ratio
                xmin = np.max([0, int(xcenter - width / 2)])
                ymin = np.max([0, int(ycenter - height / 2)])
                xmax = np.min([ori_width - 1, int(xcenter + width / 2)])
                ymax = np.min([ori_height - 1, int(ycenter + height / 2)])

                cropped_img = img[ymin:ymax, xmin:xmax]
                det_area = [xmin, ymin, xmax, ymax]
                predicted_coors = [int(coor[0] - xmin), int(coor[1] - ymin), int(coor[2] - xmin), int(coor[3] - ymin)]
                # print(klass)
                output.append([cropped_img, {'detect_table': det_area, 'table_type': klass, 'table_raw_img': img, 'table_conf': conf, 'table_predicted_coor':predicted_coors}])
        return output

    def _batch_data(data, batch_size):
        batched_data = batch_data(data, batch_size)
        spec_mask = [np.zeros((i.shape[0], cfg_detect_table.n_boxes, img_h // 32, img_w // 32), dtype=float) == 0 for i in batched_data]
        return list(zip(batched_data, spec_mask))

    img_h, img_w = cfg_detect_table.img_h, cfg_detect_table.img_w
    batch_size = cfg_detect_table.batch_size
   
    preprocessed = preprocess(inputs)
    batches = _batch_data(preprocessed, batch_size = batch_size)
    
    # output of model is in order of `['pred_x', 'pred_y', 'pred_w', 'pred_h', 'pred_conf', 'pred_prob']`
    # as feed `batch_size` minibatch into graph, each element will be shape of (batch_size, n_boxes, value, grid_h, grid_w)
    # for `pred_prob`, value is n_classes, for others, value is 1
    # to split (batch_size, n_boxes, value, grid_h, grid_w) tensor into [x] * batch_size list
    # ---WRONG WAY---
    # use `np.asarray()` to create (6, batch_size, n_boxes, value, grid_h, grid_w) tensor and then split on axis 1
    # ---RIGHT WAY---
    # as n_classes > 1, `np.array()` or `np.vstack()` can't concatenate arrays with different size
    # just split each output and then put them into one list
    # batched_preds = [pred_func(i) for i in batches]
    batched_preds = []
    for i in batches:
        table_result = pred_func(i)
        batched_preds.append(table_result)
    # ---WRONG WAY---
    # batched_preds = [np.split(np.array(pred_func(i)), len(i[0]), axis = 0) for i in batches]
    # preds = list(np.vstack(batched_preds))

    # --RIGHT WAY---
    preds = []
    # print("predicted num", len(batched_preds[0]))
    for each_batch in batched_preds:
       
        size = each_batch[0].shape[0]
        # `['pred_x', 'pred_y', 'pred_w', 'pred_h', 'pred_conf', 'pred_prob']`, and each element is a list of (1, batch_size, n_boxes, value, grid_h, grid_w) tensors
        out = [np.split(i, size) for i in each_batch]
        preds.extend([[out[j][i] for j in range(6)] for i in range(size)])
    postprocessed = [postprocess(preds[i], inputs[i]) for i in range(len(inputs))]
    return postprocessed

def detect_text_area(inputs, pred_func, enlarge_ratio=1.1):
    def preprocess(inputs):
        # resize images and convert BGR to RGB
        resized_imgs = []
        for i in inputs:
            img_color = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(img_color, (img_w, img_h))
            # image_exp = np.expand_dims(resized_img, axis=0)
            resized_imgs.append(resized_img)
        
        return resized_imgs

    def _batch_data(data, batch_size):
        
        spec_mask = [np.ones((i.shape[0], cfg_detect_text_area.n_boxes, img_h // 32, img_w // 32), dtype=bool) for i in batched_data]
        return list(zip(batched_data, spec_mask))

    img_h, img_w = cfg_detect_text_area.img_h, cfg_detect_text_area.img_w
   
    preprocessed = preprocess(inputs)
    # batches = batch_data(preprocessed, cfg_detect_text_area.batch_size)
    
 
    batched_preds = []
    for i in preprocessed:

        text_area_result = pred_func(np.expand_dims(i, axis=0))
        # pdb.set_trace()
        batched_preds.append(text_area_result)
    # print(len(batched_preds))
    # ---WRONG WAY---
    # batched_preds = [np.split(np.array(pred_func(i)), len(i[0]), axis = 0) for i in batches]
    # preds = list(np.vstack(batched_preds))

    # --RIGHT WAY---
    output = []
   
    for each_idx, each_batch in enumerate(batched_preds):
        img = inputs[each_idx]
        ori_height, ori_width = img.shape[:2]
        boxes = detect_postprocess(each_batch, image_shape=img.shape[0:2], det_th=None)
        
        conf_value = 0
        max_idx = 0
        for klass, k_boxes in boxes.items():
            for box_idx, each_box in enumerate(k_boxes):
                [xmin, ymin, xmax, ymax, conf] = each_box
                if conf < conf_value:
                    continue
                conf_value = conf
                max_idx = box_idx
            # pdb.set_trace()
            for klass, k_boxes in boxes.items():
                if (len(k_boxes)==0):
                    continue
                [xmin, ymin, xmax, ymax, conf] = k_boxes[max_idx]
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
                xcenter = (xmin + xmax) / 2
                ycenter = (ymin + ymax) / 2

                width = (xmax - xmin) * enlarge_ratio
                # threshold_w = (xmax - xmin) * threshold

                height = (ymax - ymin) * enlarge_ratio
                # threshold_h = (ymax - ymin) * threshold

                xmin = np.max([0, int(xcenter - width / 2)])
                # threshold_xmin = np.max([0, int(xcenter - threshold_w / 2)])

                ymin = np.max([0, int(ycenter - height / 2)])
                # threshold_ymin = np.max([0, int(ycenter - threshold_h / 2)])


                xmax = np.min([ori_width - 1, int(xcenter + width / 2)])
                # threshold_xmax = np.min([ori_width - 1, int(xcenter + threshold_w / 2)])

                ymax = np.min([ori_height - 1, int(ycenter + height / 2)])
                # threshold_ymax = np.min([ori_height - 1, int(ycenter + threshold_h / 2)])


                cropped_img = img[ymin:ymax, xmin:xmax]
                # cropped_img = img[threshold_ymin:threshold_ymax, threshold_xmin:threshold_xmax]

                show_area = [xmin, ymin, xmax, ymax]
                # det_area = [int(xmin - threshold_xmin), int(ymin - threshold_ymin), int(xmax - threshold_xmin), int(ymax - threshold_ymin)]
                # predicted_coors = [threshold_xmin, threshold_ymin, threshold_xmax, threshold_ymax]
                # print(klass)
                output.append([cropped_img, {'detect_area': show_area, 'type': klass, 'raw_img': img, 'conf': conf, 'last_frame_id':each_idx}])
        
    return output
def segment_lines(inputs, pred_func):
    def preprocess(inputs):
        def split(img, img_idx):
            top, bottom, left, right = 0,1,2,3
            h_per, w_per = cfg_segment_lines.h, cfg_segment_lines.w
            overlap_top, overlap_bottom, overlap_left, overlap_right  = cfg_segment_lines.overlap
            h, w = img.shape[:2]
            res_imgs = []
            res_informations = []
            h_idx, w_idx = 0, 0
            # split
            for i in range(0, h, h_per):
                for j in range(0, w, w_per):
                    padding_shape = [0,0,0,0]
                    has_pad = False
                    h_idx_start, h_idx_end = i, i + h_per
                    # do padding
                    if h_idx_start - overlap_top < 0:
                        padding_shape[top] = overlap_top - h_idx_start
                        has_pad = True
                        h_idx_start = 0
                    else:
                        h_idx_start -= overlap_top
                    if h_idx_end + overlap_bottom > h:
                        padding_shape[bottom] = h_idx_end + overlap_bottom - h
                        has_pad = True
                        h_idx_end = h
                    else:
                        h_idx_end += overlap_bottom
                    w_idx_start, w_idx_end = j, j + w_per
                    if w_idx_start - overlap_left < 0:
                        padding_shape[left] = overlap_left - w_idx_start
                        has_pad = True
                        w_idx_start = 0
                    else:
                        w_idx_start -= overlap_left
                    if w_idx_end + overlap_right > w:
                        padding_shape[right] = w_idx_end + overlap_right - w
                        has_pad = True
                        w_idx_end = w
                    else:
                        w_idx_end += overlap_right
                    res_img = img[h_idx_start:h_idx_end,w_idx_start:w_idx_end]
                    if has_pad:
                        res_img = np.pad(res_img, ((padding_shape[top], padding_shape[bottom]), (padding_shape[left], padding_shape[right])), 'edge')
                    res_imgs.append(np.expand_dims(res_img, axis = -1))
                    res_informations.append(
                        {
                            'img_idx': img_idx,
                            'h_idx': h_idx,
                            'w_idx': w_idx,
                            'padding_shape': padding_shape
                        })
                    w_idx += 1
                h_idx += 1
                w_idx = 0
            return res_imgs, res_informations

        res_imgs = []
        res_informations = []
        for idx, img in enumerate(inputs):
            if len(img.shape) == 3:
                reshaped = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                reshaped = img
            reshaped_a = cv2.adaptiveThreshold(reshaped,200,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,5)
            reshaped = cv2.medianBlur(reshaped_a, 3)
            # cv2.imwrite(str(idx)+".jpg", reshaped)
            res_img, res_information = split(reshaped, idx)
            res_imgs.extend(res_img)
            res_informations.extend(res_information)
        return res_imgs, res_informations

    def postprocess(preds, informations):
        def concat(inputs):
            res = []
            # group by h_idx
            grouped = [list(g) for k, g in groupby(inputs, lambda x: x[1]['h_idx'])]
            all_row = []
            for each_row in grouped:
                preds_per_row = [i[0] for i in each_row]
                all_row.append(np.concatenate(preds_per_row, axis = 1))
            preds_per_det_area = np.concatenate(all_row)
            return preds_per_det_area
    
        # cut off overlap part
        overlap_top, overlap_bottom, overlap_left, overlap_right = cfg_segment_lines.overlap
        cropped_preds = []
        for i in zip(preds, informations):
            h, w = i[0].shape[:2]
            padding_top, padding_bottom, padding_left, padding_right = i[1]['padding_shape']
            h_idx_start = max(overlap_top, padding_top)
            h_idx_end = h - max(overlap_bottom, padding_bottom)
            w_idx_start = max(overlap_left, padding_left)
            w_idx_end = w - max(overlap_right, padding_right)
            cropped_preds.append(i[0][h_idx_start:h_idx_end, w_idx_start:w_idx_end])

        # sort // maybe no need
        # and group by img_idx
        zipped = zip(cropped_preds, informations)
        grouped = [list(g) for k, g in groupby(zipped, lambda x:x[1]['img_idx'])]
        res = [np.argmax(concat(i), axis=2) for i in grouped]
        return res
    batch_size = cfg_segment_lines.batch_size

    preprocessed_tensors, preprocessed_informations = preprocess(inputs)
    batches = batch_data(preprocessed_tensors, batch_size)
    batched_preds = [pred_func([i])[0] for i in batches]
    preds = list(np.vstack(batched_preds))
    postprocessed = postprocess(preds, preprocessed_informations)

    return postprocessed

def new_extract_lines(self, inputs):
    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        # from: http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
        """
        import numpy as np
        from math import factorial
        
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')

    print("new_extract_lines ====", len(inputs))
    res = []
    right_idx = []
    for img_idx, each_input in enumerate(inputs):
       
        img, mask = each_input
        if len(img.shape) == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        H, W = img.shape

        # find all contours
        im2, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        tem_w = self.output_detect_text_area[img_idx][1]['detect_area'][2] - self.output_detect_text_area[img_idx][1]['detect_area'][0]
        tem_h = self.output_detect_text_area[img_idx][1]['detect_area'][3] - self.output_detect_text_area[img_idx][1]['detect_area'][1]
        for each_id, each_contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(each_contour)
 


            # if (self.output_detect_text_area[img_idx][1]['show_coor'][0] - self.output_detect_text_area[img_idx][1]['detect_area'][0] <= 1) or \
            # (self.output_detect_text_area[img_idx][1]['detect_area'][2] - self.output_detect_text_area[img_idx][1]['show_coor'][2] <= 1):
            #     if w*h < 210:
            #         continue

            # elif x < ori_coors[img_idx][0] or (x+w) > ori_coors[img_idx][2] or y <= ori_coors[img_idx][1] or (y+h) >= ori_coors[img_idx][3]  or w*h <210:
            #     continue

            if (x <= 10 and w < tem_w//5) or ((x+w)>= (tem_w-10) and w < tem_w//5) or (x+h)<=10 or x>=(tem_h-10) or w*h <=1000:
                continue

            flage_count = 0
            if len(self.output_detect_table[img_idx][1]) > 0:
                
                for table_coor in self.output_detect_table[img_idx][1]:
                    # for table_sub_coor in table_coor:
                    # print(table_coor[1]['detect_table'])
                    coor = self.output_detect_text_area[img_idx][1]['detect_area']
                    # if (x+coor[0])  >= (table_coor[1]['detect_table'][0]-5) and (x+w+coor[0]) <= (table_coor[1]['detect_table'][2]+5) and \
                    # (y+coor[1]) >= (table_coor[1]['detect_table'][1]-5) and (y+h+coor[1]) <= (table_coor[1]['detect_table'][3]+5):
                    if (x+coor[0])  >= (int(table_coor[1][0])-5) and (x+w+coor[0]) <= (int(table_coor[1][2])+5) and \
                    (y+coor[1]) >= (int(table_coor[1][1])-5) and (y+h+coor[1]) <= (int(table_coor[1][3])+5):
                          
                        flage_count += 1
                        # continue
            if flage_count > 0:
                continue

            isolated = np.zeros((H, W), np.uint8)
            cv2.fillPoly(isolated, pts = [each_contour], color = 255)
            # dilate
            kernel = np.ones((20, 20), np.uint8)
            kernel[:11] = 1
            isolated = cv2.dilate(isolated, kernel)
            # isolated = cv2.erode(isolated, kernel, iterations=1)
            # print(img_idx, len(isolated))
            isolated_im2, isolated_contours, isolated_hierarchy = cv2.findContours(isolated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for each_isolated_contour in isolated_contours:
            
                x, y, w, h = cv2.boundingRect(each_isolated_contour)
                if w * h <=200:
                    continue
                    # cv2.fillPoly(img, [each_isolated_contour], (255,0,0))
                h_idx_start = max(y, 0)
                h_idx_end = min(y + h, H - 1)
                w_idx_start = max(x, 0)
                w_idx_end = min(x + w, W - 1)
                if h_idx_start >= h_idx_end or w_idx_start >= w_idx_end:
                    continue

                # 从原图多取上下各20像素，不够的靠`mode`='edge'的pad
                if h_idx_start <= 20:
                    new_pad_top = 20 - h_idx_start
                    new_h_idx_start =  0
                else:
                    new_pad_top = 0
                    new_h_idx_start = h_idx_start - 20
                if h_idx_end >= H - 1 - 20:
                    new_pad_bottom = h_idx_end + 20 - H
                    new_h_idx_end = H - 1
                else:
                    new_pad_bottom = 0
                    new_h_idx_end = h_idx_end + 20
                raw_canvas = img[h_idx_start:h_idx_end, w_idx_start:w_idx_end]
                canvas = img[new_h_idx_start:new_h_idx_end, w_idx_start:w_idx_end]
                if w_idx_end == 485 and h_idx_end == 1214:
                    print(new_pad_top, new_pad_bottom, H-1-20)
                if new_pad_top > 0 or new_pad_bottom > 0:
                    canvas = np.pad(canvas, ((new_pad_top, new_pad_bottom),(0, 0)), mode = 'edge')
                canvas_h, canvas_w = canvas.shape[:2]
                # 创建大的空画布
                new_canvas = np.zeros((3*canvas_h,canvas_w))
                # collect all coords in contour
                centers = []
                ceilings = []
                floors = []
                # get contour's ceiling and floor
                for x in range(canvas_w):
                    max_coord = -1
                    min_coord = -1
                    for y in range(canvas_h):
                        if cv2.pointPolygonTest(each_isolated_contour,(x+w_idx_start, y+new_h_idx_start+new_pad_top),False) >= 0:
                            min_coord = y
                            break
                    for y in range(canvas_h-1, -1, -1):
                        if cv2.pointPolygonTest(each_isolated_contour,(x+w_idx_start, y+new_h_idx_start+new_pad_top),False) >= 0:
                            max_coord = y + 1
                            break
                    centers.append((max_coord+min_coord)//2)
                    floors.append(max_coord)
                    ceilings.append(min_coord)
                window_size = canvas_w//4
                if window_size % 2 == 0:
                    window_size -= 1
                window_size = max(window_size, 3)
                smoothed = list(np.round(savitzky_golay(np.array(centers), window_size, 1)))
                ceiling, floor = canvas_h, 0
                # 将整个旧的列移到空画布上
                for i in range(canvas_w):
                    new_canvas[int(canvas_h - smoothed[i]):int(2*canvas_h - smoothed[i]), i] = canvas[:, i]
                    ceiling = min(ceiling, int(canvas_h - smoothed[i] + ceilings[i]))
                    floor = max(floor, int(canvas_h - smoothed[i] + floors[i]))

                if ceiling >= floor:
                    continue
                
                res_canvas = np.copy(new_canvas[ceiling:floor])
                new_canvas[ceiling] = 0
                new_canvas[floor] = 0
                # 裁出需要的区域
                data = res_canvas
                # data = new_canvas
                information = {
                    'img_idx': img_idx,
                    'line_area': [w_idx_start, h_idx_start, w_idx_end, h_idx_end],
                    'before_align': new_canvas,
                    'binary': mask[h_idx_start:h_idx_end, w_idx_start:w_idx_end]
                }
                res.append([data, information])
                right_idx.append(img_idx)
    return res, right_idx

def recognize_sequences(inputs, pred_func, cfg_recognize_sequences, lm_model):
    def preprocess(inputs):
        # resize inputs
        resized = []
        for each_img in inputs:
            if each_img.shape[0] != input_height:
                if input_width != None:
                    resized_img = cv2.resize(each_img, (input_width, input_height))
                else:
                    scale = input_height / each_img.shape[0]
                    resized_img = cv2.resize(each_img, None, fx=scale, fy=scale)
                resized.append(resized_img)
            else:
                resized.append(each_img)
        # pad 1 channel
        imgs = [np.expand_dims(i, -1) for i in resized]
        
        return imgs
    def _batch_data(data, batch_size):
        len_data = len(data)
        batch_num = len_data // batch_size + 1 if len_data % batch_size else len_data // batch_size
        print('data will be splitted into {} batches'.format(batch_num))

        batched_data = []

        for i in range(batch_num):
            batch = []
            maxlen = max(j.shape[1] for j in data[i*batch_size:(i+1)*batch_size])
            seqlen = []
            for j in data[i*batch_size:(i+1)*batch_size]:
                seqlen.append(j.shape[1])
                if j.shape[1] == maxlen:
                    batch.append(j)
                else:
                    batch.append(np.pad(j, ((0,0), (0,maxlen - j.shape[1]), (0,0)), 'constant',constant_values=0))
            batch = np.array(batch)
            seqlen = np.array(seqlen)
            batched_data.append([batch, seqlen])
        
        return batched_data

    def postprocess(preds, cfg_recognize_sequences = cfg_recognize_sequences):
        # mapper = Mapper()
        mapper = lm_model()
        return [mapper.decode_output(i) for i in preds]
        # pdb.set_trace()
        # print("total", str(len(preds)), "sentences for language model ...")
        # return [prefix_beam_search(i, cfg=cfg_recognize_sequences, lm=lm_model) for i in preds]

    input_height = cfg_recognize_sequences.input_height
    input_width = cfg_recognize_sequences.input_width
    batch_size = cfg_recognize_sequences.batch_size

    preprocessed = preprocess(inputs)
    batches = _batch_data(preprocessed, batch_size)
   
    batched_preds = [pred_func(i)[1] for i in batches]
    print(batched_preds[0].shape)
    preds = [j for i in batched_preds for j in i]
    # start_time_rec = time.time()
    postprocessed = postprocess(preds)
    # print("language model time: ", str(time.time() - start_time_rec))
    return postprocessed


class Extractor():
    def __init__(self):
        def _init_models():
            # Load weights
            self.video_path = ''
            weights_classify_frames = SaverRestore('models/classify_frames')
            weights_detect_table = SaverRestore('models/detect_table')
            weights_detect_text_area = SaverRestore('models/detect')
            weights_segment_lines = SaverRestore('models/segment_lines')
            weights_recognize_english = SaverRestore('models/recognize_english')
            weights_recognize_chinese = SaverRestore('models/recognize_chinese')
            weights_recognize_korean = SaverRestore('models/recognize_korean')
            weights_recognize_japanese = SaverRestore('models/recognize_japanese')
            
            # Build graphs
            model_classify_frames = Model_classify_frames()
            model_detect_table = Model_detect_table()
            model_detect_text_area = Model_detect_text_area()
            model_segment_lines = Model_segment_lines()
            model_recognize_english = Model_recognize_english()
            model_recognize_chinese = Model_recognize_chinese()
            model_recognize_korean = Model_recognize_korean()
            model_recognize_japanese = Model_recognize_japanese()

            # Build predict configs
            
            config_classify_frames = PredictConfig(session_init = weights_classify_frames, model = model_classify_frames, input_names = ['input'], output_names = ['output'])
            config_detect_table = PredictConfig(session_init = weights_detect_table, model = model_detect_table, input_names = ['input', 'spec_mask'], output_names = ['pred_x', 'pred_y', 'pred_w', 'pred_h', 'pred_conf', 'pred_prob'])
            config_detect_text_area = PredictConfig(session_init = weights_detect_text_area, model = model_detect_text_area, input_names=["input"], output_names=["loc_pred", "cls_pred"])           
            config_segment_lines = PredictConfig(session_init = weights_segment_lines, model = model_segment_lines, input_names = ['input'], output_names = ['softmax_output'])
            # config_recognize_sequences = PredictConfig(session_init = weights_recognize_sequences, model = model_recognize_sequences, input_names = ['feat', 'seqlen'], output_names = ['prediction'])
            config_recognize_english = PredictConfig(session_init = weights_recognize_english, model = model_recognize_english, input_names = ['feat', 'seqlen'], output_names = ['prediction_prob', 'prediction'])
            config_recognize_chinese = PredictConfig(session_init = weights_recognize_chinese, model = model_recognize_chinese, input_names = ['feat', 'seqlen'], output_names = ['prediction_prob', 'prediction'])
            config_recognize_korean = PredictConfig(session_init = weights_recognize_korean, model = model_recognize_korean, input_names = ['feat', 'seqlen'], output_names = ['prediction_prob', 'prediction'])
            config_recognize_japanese = PredictConfig(session_init = weights_recognize_japanese, model = model_recognize_japanese, input_names = ['feat', 'seqlen'], output_names = ['prediction_prob', 'prediction'])
            

            # Build predictors
            print('start restore classify_frames-----------------------')
            self.predictor_classify_frames = OfflinePredictor(config_classify_frames)

            print('start restore detect_table-----------------------')
            self.predictor_detect_table = OfflinePredictor(config_detect_table)

            print('start restore detect_text_area-----------------------')
            self.predictor_detect_text_area = OfflinePredictor(config_detect_text_area)

            print('start restore segment_lines -----------------------')
            self.predictor_segment_lines = OfflinePredictor(config_segment_lines)
            # self.predictor_recognize_sequences = OfflinePredictor(config_recognize_sequences)

            print('start restore recognize_english-----------------------')
            self.predictor_recognize_english = OfflinePredictor(config_recognize_english)

            print('start restore recognize_chinese-----------------------')
            self.predictor_recognize_chinese = OfflinePredictor(config_recognize_chinese)

            print('start restore recognize_korean-----------------------')
            self.predictor_recognize_korean = OfflinePredictor(config_recognize_korean)

            print('start restore recognize_japanese-----------------------')
            self.predictor_recognize_japanese = OfflinePredictor(config_recognize_japanese)
            
            
            # print('start restore english language model-----------------------')
            # self.english_lm_model = kenlm.LanguageModel('models/english_lm.arpa')

            # print('start restore chinese language model-----------------------')
            # self.chinese_lm_model = kenlm.LanguageModel('models/chinese_lm.arpa')
            # self.chinese_lm_model = self.english_lm_model

        _init_models()

    def _cap_video(self,video_path):
        def rotate(img):
            # im = Image.fromarray(img)
            # im = im.transpose(Image.ROTATE_270)
            # return np.array(im)

            img = cv2.transpose(img)  
            img = cv2.flip(img,1)
            return img

        if not os.path.exists(video_path):
            print(video_path, "not exists")
            self.total_frame = 0
            return
        self.video_path = video_path
        start_time = time.time()
        frames = cap_video(video_path)
        self.frames = frames
        if len(self.frames) <= 0:
            self.total_frame = 0
            return

        
        start_time4=time.time()
        print("read video ", str(start_time4-start_time))
        height, width = frames[0].shape[:2]
        if width > height:
            self.frames = [rotate(i) for i in frames]
            self.width = height
            self.height = width
        else:
            self.frames = frames
            self.width = width
            self.height = height
        self.total_frame = len(self.frames)
        print(self.total_frame)
        print("transfer frame", str(time.time()-start_time4))
        
    def _classify_frames(self):
        print('classifing frames...')
        frames = self.frames
        if len(frames) == 0:
            self.output_classify_frames = []
            return
        if len(frames) <= len(cfg_classify_frames.frame_extract_pattern):
            self.output_classify_frames = []
            return
        pred_func = self.predictor_classify_frames
        self.output_classify_frames = classify_frames(frames, pred_func)

    def _extract_frames(self):
        if len(self.output_classify_frames) <= 0:
            self.output_extract_frames = []
            return
        print('extracting frames...')
        frames = self.frames
        label = self.output_classify_frames
        self.output_extract_frames = extract_frames(frames, label)

    def _detect_text_area(self):
        if len(self.output_extract_frames) <= 0:
            self.output_detect_text_area = []
            self.output_extract_frames_save = self.output_extract_frames.copy()
            return
        print('detecting text area...')
        inputs = []
        informations = []
        print("output_extract_frames", len(self.output_extract_frames))
        for i in self.output_extract_frames:
            inputs.append(i[0])
            informations.append(i[1])
            print(i[1]['frame_idx'])
        pred_func = self.predictor_detect_text_area
        pure_outputs = detect_text_area(inputs, pred_func)
        print("after area detect", len(pure_outputs))
        # pdb.set_trace()
        # text_area_list = []
        outputs = []
        for i_idx, i in enumerate(pure_outputs):
           
            # if len(i) <= 0:
            #     # pdb.set_trace()
            #     text_area_list.append(i_idx)
            #     continue
            # max_area = []
            # max_area_indx = []
            # for j in range(len(i)):
              
            #     if i[j][1]['type'] == 'text_area':
            #         max_area.append(i[j][1]['conf'])
            #         max_area_indx.append(j)
            # if len(max_area) <=0:
            #     text_area_list.append(i_idx)
            #     continue

            # max_index = max_area_indx[np.argmax(max_area)]
            # data = i[max_index][0]
            # added_information = i[max_index][1]

            information = deepcopy(informations[i[1]['last_frame_id']])

            information.update(i[1])
            outputs.append([i[0], information])
        # print("no text_area_list ", text_area_list)   
        self.output_detect_text_area = outputs

        self.output_extract_frames_save = self.output_extract_frames.copy()
        # if len(text_area_list) >= 1:
        #     for i_idx, i in enumerate(text_area_list):
        #         if i_idx >= 1:
        #             self.output_extract_frames.pop(i-i_idx)
        #         else:
        #             self.output_extract_frames.pop(i)
        print("text area detect num ", len(self.output_detect_text_area), len(self.output_extract_frames))
        # quit()

    def _detect_table(self):
        if len(self.output_detect_text_area) <= 0:
            self.output_detect_table = []
            return
        print('detecting table...')
        inputs = []
        informations = []
        
        for i in self.output_extract_frames:
            inputs.append(i[0])
            informations.append(i[1])
        pred_func = self.predictor_detect_table
        pure_outputs = detect_table(inputs, pred_func)
        
        outputs = []
        # pdb.set_trace()
        print(len(pure_outputs) == len(inputs))
        # quit()

        # for i_idx, i in enumerate(pure_outputs):
            # table_max_conf = 0.001
            # table_max_index = 0
            # table_num = 0

            # figure_max_conf = 0.001
            # figure_max_index = 0
            # figure_num = 0
            # print("----", len(i))
            # tem = []
            # if len(i) <= 0:
            #     outputs.append([i_idx, tem])
            #     continue
            # print(i_idx, len(i))
            # for j in range(len(i)):
            #     if i[j][1]['table_type'] == 'table' and i[j][1]['table_conf'] > table_max_conf:
            #         table_num += 1
            #         table_max_conf = i[j][1]['table_conf']
            #         table_max_index = j
            #     elif i[j][1]['table_type'] == 'figure' and i[j][1]['table_conf'] > figure_max_conf:
            #         figure_num += 1
            #         figure_max_conf = i[j][1]['table_conf']
            #         figure_max_index = j
            
            # if table_num > 0:
            #     tem.append(["table", i[table_max_index][1]])
            # if figure_num > 0:
            #     tem.append(["figure", i[figure_max_index][1]])
            # outputs.append([i_idx, tem])
        
        for i_idx, i in enumerate(pure_outputs):
            tem = []
            if len(i) <= 0:
                outputs.append([i_idx, tem])
                continue
            # print(i_idx, len(i))
            for j in range(len(i)):
                # if i[j][1]['table_type'] == 'table':
                #     tem.append(["table", i[j][1]])
                # elif i[j][1]['table_type'] == 'figure':
                #     tem.append(["figure", i[j][1]])
                tem.append([i[j][1]['table_type'], i[j][1]['detect_table']])
            outputs.append([i_idx, tem])
        # pdb.set_trace()

        self.output_detect_table = outputs
        print("table detect num ", len(self.output_detect_table), len(self.output_extract_frames))
        
    def _segment_lines(self):
        print('segmenting lines ...', len(self.output_detect_text_area))
        if len(self.output_detect_text_area) <= 0:
            self.output_segment_lines = []
            return
        inputs = []
        outputs = []
        informations = []
        tem_output_detect_text_area = self.output_detect_text_area
        # self.output_detect_text_area = []
        for i in tem_output_detect_text_area:
           
            if i[1]['type'] == 'text_area':
                inputs.append(i[0])
                informations.append(i[1])

        if len(inputs) <= 0:
            self.output_segment_lines = []
            return
        self.output_detect_text_area = []
        for i in range(len(inputs)):
            self.output_detect_text_area.append([inputs[i], informations[i]])

        print(len(inputs) == len(informations))
        print(len(self.output_detect_text_area))
  
        pred_func = self.predictor_segment_lines
        pure_outputs = segment_lines(inputs, pred_func)
        for i in range(len(pure_outputs)):
            data = pure_outputs[i]
            information = deepcopy(informations[i])
            information['img'] = inputs[i]
            outputs.append([data, information])
        # pdb.set_trace()
        self.output_segment_lines = outputs
      
    def _extract_lines(self):
        print('extracting lines ...')
        if len(self.output_segment_lines) <= 0:
            self.output_extract_lines = []
            return

        print(len(self.output_detect_text_area), len(self.output_segment_lines))
        inputs = []
        informations = []
        # ori_coors = []
        for i,j in zip(self.output_detect_text_area, self.output_segment_lines):
            inputs.append([i[0], j[0]])
            informations.append(j[1])
            # ori_coors.append(j[1]['predicted_coor'])
        pure_outputs, right_idx_lists = new_extract_lines(self,inputs)
        print("after new extractt_lines", len(pure_outputs), "after new extract_lines right_list num", len(right_idx_lists))
        right_list = list(set(right_idx_lists))
        print("last, after new extract_lines", len(right_list))
        print(right_list)
        error_list = []
        if len(self.output_detect_text_area) != len(right_list):
            error_list = list(set([i for i in range(len(self.output_detect_text_area))]).difference(set(right_list)))
            print("error_list", error_list)
            print(error_list)

        error_list_frame_idx = []
        if len(error_list) > 0:
            
            for err_x in range(len(self.output_detect_text_area)):
                if err_x in error_list:
                    error_list_frame_idx.append(self.output_detect_text_area[err_x][1]['frame_idx'])
        
        print(error_list_frame_idx)


        grouped = [list(g) for k,g in groupby(pure_outputs, lambda x: x[1]['img_idx'])]
        outputs = []
        print("grouped", len(grouped))
        for i in range(len(grouped)):
            ii = right_list[i]
            added_information = deepcopy(informations[ii]) 
            for j in grouped[i]:
                data = j[0]
                information = j[1]
                information.pop('img_idx')
                information.update(added_information)
                outputs.append([data, information])
        self.output_extract_lines = outputs
        self.output_error_list = error_list_frame_idx
      
    def _recognize_sequences(self, pred_func, cfg_recognize_sequences, lm_model):
        print('recognizing sequences...')
        if len(self.output_extract_lines) <= 0:
            self.output_recognize_sequences = []
            return
        data = []
        informations = []
        for i in self.output_extract_lines:
            data.append(i[0])
            informations.append(i[1])


        pure_outputs = recognize_sequences(data, pred_func, cfg_recognize_sequences, lm_model)
        outputs = []
        for i in range(len(pure_outputs)):
            data = pure_outputs[i]

            information = informations[i]
            outputs.append([data, information])
        self.output_recognize_sequences = outputs


    def from_video(self,video_path,  language_name='english'):
        self._cap_video(video_path)

        self._classify_frames()

        self._extract_frames()
       
        self._detect_text_area()
        self._detect_table()
        self._segment_lines()
        self._extract_lines()
        # Mapper  Model_recognize_sequences cfg_recognize_sequences
        if language_name == 'english':
            pred_func = self.predictor_recognize_english
            cfg_recognize_sequences = cfg_recognize_english
            Mapper  = Mapper_english
            # lm_model = self.english_lm_model
        elif language_name == 'chinese':
            pred_func = self.predictor_recognize_chinese
            cfg_recognize_sequences = cfg_recognize_chinese
            Mapper  = Mapper_chinese
            # lm_model = self.chinese_lm_model
        elif language_name == 'korean':
            pred_func = self.predictor_recognize_korean
            cfg_recognize_sequences = cfg_recognize_korean
            Mapper  = Mapper_korean
        else:
            pred_func = self.predictor_recognize_japanese
            cfg_recognize_sequences = cfg_recognize_japanese
            Mapper  = Mapper_japanese

        self._recognize_sequences(pred_func, cfg_recognize_sequences, Mapper)
        self.output_type = 'video'
    def from_image(self, img_paths):
        self.output_extract_frames = [[cv2.imread(img_path), idx] for idx, img_path in enumerate(img_paths)]
        self._detect_text_area()
        self._segment_lines()
        self._extract_lines()
        self._recognize_sequences()
        self.output_type = 'images'

    def from_txt(self, txt_path):
        img_paths = []
        lines = open(txt_path, 'r').readlines()
        for i in lines:
            img_paths.append(i[:-1])
        self.from_image(img_paths)
    
    def save(self):
        import os, shutil
        from datetime import datetime
        video_name  = self.video_path.split("/")[-1]
        self.filename = 'output-' + video_name + "-" + datetime.now().strftime('%Y%m%d%H%M%S')
        filename = self.filename
        if os.path.isdir(filename):
            shutil.rmtree(filename)
        os.mkdir(filename)


        # grouped = [list(g) for f,g in groupby(self.output_detect_text_area, lambda x: x[1]['frame_idx'])]
        # table_count = 0
        # print("grouped ", len(grouped))
        # for group in grouped:
        #     img = group[0][1]['raw_img']
        #     for idx, data in enumerate(group):
        #         x, y, x_end, y_end = data[1]['detect_area']
        #         cv2.rectangle(img,
        #                 (x, y),
        #                 (x_end, y_end),
        #                 (0, 0, 255),
        #                 4)
        #         if len(self.output_detect_table[table_count][1]) > 0:
        #             for table_coor in self.output_detect_table[table_count][1]:
        #                 cv2.rectangle(img,
        #                 (table_coor[1]['detect_table'][0], table_coor[1]['detect_table'][1]),
        #                 (table_coor[1]['detect_table'][2], table_coor[1]['detect_table'][3]),
        #                 (255, 0, 145),
        #                 4)
        
                # cv2.imwrite('{}/detect_text_area/{}-{}.png'.format(self.filename,data[1]['frame_idx'], idx), data[0])
            # cv2.imwrite('{}/detect_text_area/{}.png'.format(self.filename,data[1]['frame_idx']), img)
            # table_count += 1

        if not cfg.isSaveResult:
            dirs = ['gui_frames', 'gui_preds']
            for i in dirs:
                os.mkdir('{}/{}'.format(self.filename, i))
            return
            # grouped = [list(g) for f,g in groupby(self.output_detect_text_area, lambda x: x[1]['frame_idx'])]
            # table_count = 0
            # print("grouped ", len(grouped))
            # for group in grouped:
            #     img = group[0][1]['raw_img']
            #     for idx, data in enumerate(group):
            #         x, y, x_end, y_end = data[1]['detect_area']
            #         cv2.rectangle(img,
            #                 (x, y),
            #                 (x_end, y_end),
            #                 (0, 0, 255),
            #                 4)
            #         if len(self.output_detect_table[table_count][1]) > 0:
            #             for table_coor in self.output_detect_table[table_count][1]:
            #                 cv2.rectangle(img,
            #                 (table_coor[1]['detect_table'][0], table_coor[1]['detect_table'][1]),
            #                 (table_coor[1]['detect_table'][2], table_coor[1]['detect_table'][3]),
            #                 (255, 0, 145),
            #                 4)
            
            #         # cv2.imwrite('{}/detect_text_area/{}-{}.png'.format(self.filename,data[1]['frame_idx'], idx), data[0])
            #     # cv2.imwrite('{}/detect_text_area/{}.png'.format(self.filename,data[1]['frame_idx']), img)
            #     table_count += 1

            

        dirs = ['extract_frames', 'detect_text_area', 'segment_lines', 'extract_lines', 'recognize_sequences','gui_frames', 'gui_preds']
        if self.output_type == 'video':
            dirs = ['frames', 'classify_frames'] + dirs

        for i in dirs:
            os.mkdir('{}/{}'.format(self.filename, i))

        if self.output_type == 'video':
            # save frames
            for idx, data in enumerate(self.frames):
                cv2.imwrite('{}/frames/{}.png'.format(self.filename, idx), data)

            # save output of classify_frames
            f = open(self.filename + '/classify_frames/prediction.txt', 'w')
            for i in list(self.output_classify_frames):
                f.write(str(i))

        # save output of extract_frames
        for data in self.output_extract_frames_save:
            cv2.imwrite('{}/extract_frames/{}.png'.format(self.filename, data[1]['frame_idx']), data[0])

        # save output of detect_text_area
        grouped = [list(g) for f,g in groupby(self.output_detect_text_area, lambda x: x[1]['frame_idx'])]
        
        table_count = 0
        print("grouped ", len(grouped))
        for group in grouped:
            img = group[0][1]['raw_img'].copy()
            for idx, data in enumerate(group):
                x, y, x_end, y_end = data[1]['detect_area']
                cv2.rectangle(img,
                        (x, y),
                        (x_end, y_end),
                        (0, 0, 255),
                        4)
        #         if len(self.output_detect_table[table_count][1]) > 0:
        #             for table_coor in self.output_detect_table[table_count][1]:
        #                 cv2.rectangle(img,
        #                 (table_coor[1]['detect_table'][0], table_coor[1]['detect_table'][1]),
        #                 (table_coor[1]['detect_table'][2], table_coor[1]['detect_table'][3]),
        #                 (255, 0, 145),
        #                 4)
        
                cv2.imwrite('{}/detect_text_area/{}-{}.png'.format(self.filename,data[1]['frame_idx'], idx), data[0])
            cv2.imwrite('{}/detect_text_area/{}.png'.format(self.filename,data[1]['frame_idx']), img)
        #     table_count += 1

        # save output of segment_lines
        
        for i,j in zip(self.output_detect_text_area, self.output_segment_lines):
            img = i[0].copy()
            print(img.shape)
            img = j[1]['img']
            mask = j[0]
            print(img.shape)
            # mask = np.expand_dims(j[0], -1)
            # mask = np.concatenate((mask, np.zeros_like(mask), np.zeros_like(mask)), axis=2)
            # mask = mask
            boolean = mask == 255
            information = j[1]
            mask = mask == 1
            mask_img = np.zeros(img.shape)
            mask_img[:,:,2][mask] = 255
            canvas = img * 0.7 + mask_img * 0.3

            detect_area_coor = [j[1]['detect_area'][1], j[1]['detect_area'][0], j[1]['detect_area'][3], j[1]['detect_area'][2]]
            print(detect_area_coor)

            # cv2.imwrite('{}/segment_lines/{}-({},{})({},{}).png'.format(self.filename,j[1]['frame_idx'], *(j[1]['detect_area'])), canvas)
            # misc.imsave('{}/segment_lines/{}-({},{})({},{})-mask.png'.format(self.filename,j[1]['frame_idx'], *(j[1]['detect_area'])), mask)
            cv2.imwrite('{}/segment_lines/{}-({},{})({},{}).png'.format(self.filename,j[1]['frame_idx'], *(detect_area_coor)), canvas)
            misc.imsave('{}/segment_lines/{}-({},{})({},{})-mask.png'.format(self.filename,j[1]['frame_idx'], *(detect_area_coor)), mask_img)

        # save output of extract_lines
        for data in self.output_extract_lines:
            # line_img = cv2.cvtColor(data[0].astype(np.uint8), cv2.COLOR_BGR2GRAY)
            # cv2.imwrite('{}/extract_lines/{}-({},{})({},{})-({},{})({},{}).png'.format(self.filename, data[1]['frame_idx'], *data[1]['detect_area'], *data[1]['line_area']), data[0])
            # cv2.imwrite('{}/extract_lines/{}-({},{})({},{})-({},{})({},{}) - before_align.png'.format(self.filename, data[1]['frame_idx'], *data[1]['detect_area'], *data[1]['line_area']), data[1]['before_align'])
            detect_area_coor = [data[1]['detect_area'][1], data[1]['detect_area'][0], data[1]['detect_area'][3], data[1]['detect_area'][2]]
            line_area_coor = [data[1]['line_area'][1], data[1]['line_area'][0], data[1]['line_area'][3], data[1]['line_area'][2]]

            cv2.imwrite('{}/extract_lines/{}-({},{})({},{})-({},{})({},{}).png'.format(self.filename, data[1]['frame_idx'],\
            detect_area_coor[0], detect_area_coor[1], detect_area_coor[2], detect_area_coor[3], line_area_coor[0], line_area_coor[1], line_area_coor[2], line_area_coor[3]), data[0])
            
            cv2.imwrite('{}/extract_lines/{}-({},{})({},{})-({},{})({},{}) - before_align.png'.format(self.filename, data[1]['frame_idx'],\
            detect_area_coor[0], detect_area_coor[1], detect_area_coor[2], detect_area_coor[3], line_area_coor[0], line_area_coor[1], line_area_coor[2], line_area_coor[3]), data[1]['before_align'])
        
        # save output of recognize_sequences
        for data in self.output_recognize_sequences:
            detect_area_coor = [data[1]['detect_area'][1], data[1]['detect_area'][0], data[1]['detect_area'][3], data[1]['detect_area'][2]]
            line_area_coor = [data[1]['line_area'][1], data[1]['line_area'][0], data[1]['line_area'][3], data[1]['line_area'][2]]
            # f = open('{}/recognize_sequences/{}-({},{})({},{})-({},{})({},{}).txt'.format(self.filename, data[1]['frame_idx'], *data[1]['detect_area'], *data[1]['line_area']), 'w')
            f = open('{}/recognize_sequences/{}-({},{})({},{})-({},{})({},{}).txt'.format(self.filename, data[1]['frame_idx'],\
            detect_area_coor[0], detect_area_coor[1], detect_area_coor[2], detect_area_coor[3], line_area_coor[0], line_area_coor[1], line_area_coor[2], line_area_coor[3]), 'w')
            f.write(data[0])

    def gui(self, language_name='english'):
        def sort_areas(l):
            # TODO
            # 文本区域/图/表， 应返回文本区域的，排序后结果
            if len(l) == 1:
                return l
            
        def sort_lines(l):
            # l = sorted(l, key = lambda x: x[3][1])
            l = sorted(l, key = lambda x: (x[1]['line_area'][1], x[1]['line_area'][0]))
            # l = sorted(l, key = lambda x: (x[1]['line_area'][1]))
            return l
        self.gui_frames = []
        if len(self.frames) <= 0:
            return 9
        if len(self.output_recognize_sequences) <= 0:
            return 0
        # pdb.set_trace()
        pre_frame_id = []
        for idx in range(len(self.output_segment_lines)):
            # img = np.copy(self.output_extract_frames[idx][0])
            ids_ = self.output_segment_lines[idx][1]['frame_idx']
            img = np.copy(self.frames[ids_])
            pre_frame_id.append(ids_)
            origin_h, origin_w = img.shape[:2]
            mask = np.copy(self.output_segment_lines[idx][0])
            x, y, x_end, y_end = self.output_segment_lines[idx][1]['detect_area']
            # x1, y1, x_end1, y_end1 = self.output_segment_lines[idx][1]['show_coor']
            filled = mask.astype(np.uint8).copy()
            # for line in self.output_recognize_sequences:
                # same frame and same area
                # if line[1] == self.output_extract_frames[idx][1] and line[2] == [x, y, x_end, y_end]:
                #     txt = ''.join(line[0].split())
                #     if len(txt) == 0:
                #         cv2.fillPoly(filled, pts =[line[4]], color = 0)
            
            mask = np.pad(filled, ((y, origin_h - y_end), (x, origin_w - x_end)), 'constant', constant_values = 0)
            mask = mask == 1
            mask_img = np.zeros(img.shape)
            mask_img[:,:,2][mask] = 255
            img = img * 0.7 + mask_img * 0.3
            cv2.rectangle(img, (x,y), (x_end, y_end), (0, 0, 255), 4)
            # cv2.rectangle(img, (x1,y1), (x_end1, y_end1), (255, 255, 0), 2)

            if len(self.output_detect_table[idx][1]) > 0:
                for table_coor in self.output_detect_table[idx][1]:
                    cv2.rectangle(img,
                        # (table_coor[1]['detect_table'][0], table_coor[1]['detect_table'][1]),
                        # (table_coor[1]['detect_table'][2], table_coor[1]['detect_table'][3]),
                        # (255, 0, 145),
                        # 4)
                        (table_coor[1][0], table_coor[1][1]),
                        (table_coor[1][2], table_coor[1][3]),
                        (255, 0, 145),4)
            self.gui_frames.append(img)
        self.gui_preds = []
        # sort by frame_idx, det_area, and line_area
        # pdb.set_trace()
        predictions = self.output_recognize_sequences
        predictions = sorted(predictions, key = lambda pre_ : (pre_[1]['frame_idx'], pre_[1]['detect_area'], pre_[1]['line_area']))

        # predictions = sorted(predictions, key = itemgetter(1,2,3))
        # split by frame_idx
        # predictions_per_frame = [list(g) for k, g in groupby(predictions, lambda x: x[1])]
        predictions_per_frame = [list(g) for k, g in groupby(predictions, lambda x: x[1]['frame_idx'])]
        for i in predictions_per_frame:
            for j in i:
                if j[1]['frame_idx'] != i[0][1]['frame_idx']:
                    print(i)
                    print('---')
                    print(j)
        # 同帧不同的det_area排序，先上后下，先左后右
        # 上下：比较y，左右：比较x 
        # 每帧分别排序，每个det_area分别排序，根据x, y, x_end, y_end来排序
        # pdb.set_trace()

        ##filter the repeat frames by recognize result
        repeat_frame_list = []
        repeat_buffer = []
        cmp_buffer = []
        repeat_value = []

        mid = []
        if language_name == 'english':
            cfg_recognize_sequences = cfg_recognize_english
            min_loc = 12
            max_loc = 81
        elif language_name == 'chinese':
            cfg_recognize_sequences = cfg_recognize_chinese
            min_loc = 185
            max_loc = 3683
        elif language_name == 'korean':
            cfg_recognize_sequences = cfg_recognize_korean
            min_loc = 76
            max_loc = 1621
            mid = [266, 970, 1262, 855, 209, 1245, 668]
        else:
            cfg_recognize_sequences = cfg_recognize_japanese
            min_loc = 69
            max_loc = 6025
            # mid = [266, 970, 1262, 855, 209, 1245, 668]

        output_per_frame = []
        for idx, pred_each_frame in enumerate(predictions_per_frame):
            # if idx >=4:
            #     continue
            # # predictions_per_det_area = [list(g) for k, g in groupby(pred_each_frame, lambda x: x[2])]
            # predictions_per_det_area = [list(g) for k, g in groupby(pred_each_frame, lambda x: x[1]['detect_area'])]
            print("predictions_per_det_area num must be 1 ", len(pred_each_frame))
            # 按先上后下，先左后右排序
            # pdb.set_trace()
            sorted_lines = sort_lines(pred_each_frame)
            # 对每一个det_area
            
            buffer_list = []
            output_lines = []
            text_area_coor = pred_each_frame[0][1]['detect_area']
            x = text_area_coor[0]
            y = text_area_coor[1]
            w = text_area_coor[2] - text_area_coor[0]
            h = text_area_coor[3] - text_area_coor[1]
            middle_left = x + w // 2 + int(w*0.1)
            middle_right = x + w // 2 - int(w*0.1)
            # middle_right = middle_left
            # print()
            # print("middle_left ",middle_left, middle_right)
            # print("sorted_areas num must be 1 ", len(sorted_areas))
            cmp_list = sorted(random.sample(list(np.arange(len(sorted_lines))), math.floor(len(sorted_lines)*0.9)))
            # cmp_list = sorted(list(np.arange(len(sorted_lines))))
                   
            repeat_buffer = cmp_buffer
            cmp_buffer = []
            # print(repeat_buffer)
        
            repeat_frame_list.clear()

            for det_area_idx, single_line in enumerate(sorted_lines):
                # print(len(single_line[0]))
                if idx >= 1:
                    # repeat_frame_list = [1 for e in repeat_buffer if e in single_line[0]]
                    for e in repeat_buffer:
                        if e in single_line[0].strip(' '):
                            # print(idx)
                            # print(e)
                            # print(single_line[0])
                            repeat_frame_list.append(idx)
                            pre_ = list(set(list(e)) & set(list(single_line[0])))
                            repeat_tem = 0
                            for sub_pre in pre_:
                                if min_loc <= cfg_recognize_sequences.dictionary.index(sub_pre) <= max_loc and cfg_recognize_sequences.dictionary.index(sub_pre) not in mid:
                                    repeat_tem += 1
                            if repeat_tem == 0:
                                continue
                            # print(idx)
                            # print(e)
                            # print(single_line[0])
                            # repeat_frame_list.append(idx)
                            # print(repeat_frame_list)
                           
                if det_area_idx in cmp_list:

                    if len(single_line[0]) >= 10 and len(single_line[0]) <= 20:
                        if len(single_line[0].strip(' ')) >15:
                            cmp_buffer.append(single_line[0].strip(' '))
                    elif len(single_line[0]) > 21 and len(single_line[0]) <= 50:
                        if len(single_line[0][21:50].strip(' ')) >5:
                            cmp_buffer.append(single_line[0][15:50].strip(' '))
                    elif len(single_line[0]) > 51 and len(single_line[0]) <= 100:
                        if len(single_line[0][55:90].strip(' ')) > 5 and len(single_line[0][10:50].strip()) > 5:
                            cmp_buffer.append(single_line[0][20:90].strip(' '))
                            cmp_buffer.append(single_line[0][4:50].strip(' '))
                    else:
                        if len(single_line[0][55:90].strip(' ')) > 0 and len(single_line[0][10:50].strip())>5:
                            cmp_buffer.append(single_line[0][20:47].strip(' '))
                            cmp_buffer.append(single_line[0][40:99].strip(' '))
                    # if len(single_line[0].strip(' ')) >= 5:
                    #   cmp_buffer.append(single_line[0].strip(' '))

                
            # for singe_line_idx, single_line in enumerate(pred_each_frame):
                if single_line[0] == '':
                    continue
                if single_line[1]['line_area'][2] <= middle_left:
                    output_lines.append(single_line[0])

                elif single_line[1]['line_area'][0] >= middle_right:
                    buffer_list.append(single_line[0])

                else:
                    if len(buffer_list) > 0:
                        output_lines.extend(buffer_list)
                        buffer_list.clear()
                    output_lines.append(single_line[0])
            # pdb.set_trace()

            if len(repeat_frame_list) >= 2:
                repeat_value.append(repeat_frame_list[0])
                repeat_frame_list.clear()
                continue


            if len(buffer_list) > 0:
                output_lines.extend(buffer_list)
            output_per_frame.append('\n'.join(output_lines))
            
            # print("repeat frame list:")
            # print(repeat_frame_list)
            # if len(repeat_frame_list) >= 2:
            #     repeat_value.append(repeat_frame_list[0])
            #     repeat_frame_list.clear()

            # print(repeat_value)
                # 先上后下，先左后右（合并）
            #     sorted_lines = sort_lines(pred_each_det_area)
            #     output_lines.append('\n'.join(i[0] for i in sorted_lines if i[0] != ''))
                # output_per_frame.append('\n'.join(output_lines))
            self.gui_preds = output_per_frame



        print(len(self.gui_preds), len(self.gui_frames))
        print("repeat frame num:")
        print(repeat_value)
        repeat_value_frame_idx = []

        for repeat_idx in range(len(predictions_per_frame)):
            if repeat_idx in repeat_value:
                repeat_value_frame_idx.append(predictions_per_frame[repeat_idx][0][1]['frame_idx'])

        print("last frame idx", self.output_error_list, repeat_value_frame_idx)

        gui_frame_tem = []
        gui_preds_tem = []
        save_id = 0
        for idx, img in enumerate(self.gui_frames):
            if pre_frame_id[idx] in repeat_value_frame_idx:
                continue
            gui_frame_tem.append(img)
            cv2.imwrite('{}/gui_frames/{}.png'.format(self.filename, save_id), img)
            save_id += 1
        self.gui_frames = gui_frame_tem
        # pdb.set_trace()
        if len(self.output_error_list) == 0:
            save_id = 0
            for idx, pred in enumerate(self.gui_preds):
                # if idx in repeat_value:
                #     continue
                gui_preds_tem.append(pred)
                open('{}/gui_preds/{}.txt'.format(self.filename, save_id), 'w').write(pred)
                save_id += 1
            self.gui_preds = gui_preds_tem
        else:
            self.new_gui_freds = []
            count_index = 0
            save_id = 0
            for idx in range(len(self.output_detect_text_area)):
                if self.output_detect_text_area[idx][1]['frame_idx'] in repeat_value_frame_idx:
                    continue
                if self.output_detect_text_area[idx][1]['frame_idx'] in self.output_error_list:
                    open('{}/gui_preds/{}.txt'.format(self.filename, save_id), 'w').write("                 ")
                    save_id += 1
                    self.new_gui_freds.append("                ")
                else:
                    open('{}/gui_preds/{}.txt'.format(self.filename, save_id), 'w').write(self.gui_preds[count_index])
                    save_id += 1
                    self.new_gui_freds.append(self.gui_preds[count_index])
                    count_index += 1
            self.gui_preds = self.new_gui_freds

            # for idx, pred in enumerate(self.gui_preds):
            #     if idx in [self.output_error_list]:
            #         open('{}/gui_preds/{}.txt'.format(self.filename, idx), 'w').write(pred)
            #     else:
            #         open('{}/gui_preds/{}.txt'.format(self.filename, idx), 'w').write(pred)

if __name__ == '__main__':

    ext = Extractor()
    # ext.from_video('/home/user/VideoText/DEMO/classify_frames/label_tool/raw_videos/data_20180109/VID_20180109_095217.mp4')
    ext.from_video('/home/user/VideoText/DEMO/test2.mp4')
    # if len(self.total_frame) != 0:
    #   ext.save()
