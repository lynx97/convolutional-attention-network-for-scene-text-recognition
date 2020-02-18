import numpy as np
import tensorflow as tf

from imageio  import imread,  imsave
from PIL import Image
import numpy as np
import cv2

import config

def resize_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (400,128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print("error with read file: ", image_path)
        return None
    return img

def resize(img):
    img = cv2.resize(img, (400,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def check_character_level(gt_string, pred_string):
    gt_string = gt_string.lower()
    pred_string = pred_string.lower()
    s = 0
    for i in range(len(gt_string)):
        if gt_string[i] == '*' and i > 0:
            break
        if gt_string[i] == pred_string[i]:
            s+= 1
    return s/i

def cal_accuracy(GT, PRED, files, file_log):
    word_acc = 0
    char_acc = 0
    for i, gt in enumerate(GT):
        gt_string = ground_truth_to_word(gt)
        pred_string = ground_truth_to_word(PRED[i])
        if gt_string.upper() == pred_string.upper():
            word_acc += 1
        else:
            with open(file_log, "a+") as f:
                str_w = files[i] + " " + gt_string + " " + pred_string
                f.write(str_w)
                f.write("\n")
    return word_acc/GT.shape[0], char_acc/GT.shape[0]

def label_to_array_2(label, length):
    try:
        flag = True
        label_array = np.zeros((length))
        for i in range(0, len(label)):
            try:
                label_array[i] = config.CHAR_VECTOR.index(label[i])
            except Exception as ex:
                label_array[i] = 0
                flag = False
                return label_array, flag
        label_array[len(label)] = config.NUM_CLASSES - 1
        return label_array, flag
    except Exception as ex:
        print(label)
        raise ex

def ground_truth_to_word(ground_truth):
    """
        Return the word string based on the input ground_truth
    """
    result = ""
    try:
        for arr in ground_truth:
            if int(arr) == (config.NUM_CLASSES-1):
                break
            else:
                result += config.CHAR_VECTOR[int(arr)]
        return result
    except Exception as ex:
        print(ground_truth)
        print(ex)
        input()
