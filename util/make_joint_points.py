import json
from sys import implementation
from typing import List
import pdb
import os
import numpy as np

# alphapose_json_path = "alphapose-results_test.json"
# json_open = open(alphapose_json_path, 'r')
# alphapose = json.load(json_open)[0]['keypoints']
# alphapose = None

speech2gesture_txt_path = "intermediate/test_decode.txt"
COCO_17 = {"Nose": 0, "LEye": 1, "REye": 2, "LEar": 3, "REar": 4, "LShoulder": 5,
           "RShoulder": 6, "LElbow": 7, "RElbow": 8, "LWrist": 9, "RWrist": 10, "LHip": 11,
           "RHip": 12, "LKnee": 13, "RKnee": 14, "LAnkle": 15, "RAnkle": 16}
Speech_Gen = {"Head": 5, "Nose": 4, "Neck": 3, "Hips": 0, "RShoulder": 30, "RArm": 31, "Spine1": 2,
              "RForeArm": 32, "RHand": 33, "LShoulder": 6, "LArm": 7, "LForeArm": 8, "LHand": 9, "LUpLeg": 54, "LLeg": 55, "LFoot": 56, "RUpLeg": 59, "RLeg": 60, "RFoot": 61}
LSP_14 = {"RAnkle": 0, "RKnee": 1, "RHip": 2, "LHip": 3, "LKnee": 4, "LAnkle": 5, "RWrist": 6,
          "RElbow": 7, "RShoulder": 8, "LShoulder": 9, "LElbow": 10, "LWrist": 11, "Neck": 12, "Head": 13}
OPENPOSE = {"Nose": 0, "Neck": 1, "LShoulder": 2, "LElbow": 3,
            "LWrist": 4, "RShoulder": 5, "RElbow": 6, "RWrist": 7, "LHip": 8, "LKnee": 9, "LAnkle": 10, "RHip": 11, "RKnee": 12, "RAnkle": 13, "LEye": 14, "REye": 15, "LEar": 16, "REar": 17}

SAVE_DIR = "intermediate/location_adgan/"
os.makedirs(SAVE_DIR, exist_ok=True)


def arr_to_point(npy_file):
    result = []
    npy_arr = np.load(npy_file)
    for i in range(npy_arr.shape[-1]):
        point_idxes = np.nonzero(npy_arr[:, :, i] == 1)
        result.extend([point_idxes[1][0], point_idxes[0][0], 0])
    return result


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result


def get_pos_sppecg_gen(joints, idx):
    return [joints[idx*3], joints[idx*3+1]]


def get_pos_alphapose(joints: List, idx: int):
    return [joints[idx*3], joints[idx*3+1]]


alphapose = arr_to_point('util/fashionWOMENTees_Tanksid0000660217_4full.jpg.npy')
width_ratio, height_ratio = get_pos_alphapose(alphapose, OPENPOSE["RWrist"])[0]-get_pos_alphapose(alphapose, OPENPOSE["LWrist"])[
    0], max(get_pos_alphapose(alphapose, OPENPOSE["RAnkle"])[1], get_pos_alphapose(alphapose, OPENPOSE["LAnkle"])[1])-min(get_pos_alphapose(alphapose, OPENPOSE["REye"])[1], get_pos_alphapose(alphapose, OPENPOSE["LEye"])[1])
width_l, height_u = get_pos_alphapose(alphapose, OPENPOSE["LWrist"])[0], min(get_pos_alphapose(alphapose, OPENPOSE["REye"])[
    1], get_pos_alphapose(alphapose, OPENPOSE["LEye"])[1])
alphapose = np.array(alphapose, dtype=np.float64)
alphapose[[i for i in range(len(alphapose)) if i % 3 == 0]] = min_max(
    alphapose[[i for i in range(len(alphapose)) if i % 3 == 0]])
alphapose[[i for i in range(len(alphapose)) if i % 3 == 1]] = min_max(
    alphapose[[i for i in range(len(alphapose)) if i % 3 == 1]])
alphapose = list(alphapose)


def get_pos_3d(joints, idx):
    return [joints[idx*3], joints[idx*3+1], joints[idx*3+2]]


def ratio_from_nose(nose_pos: List, other_pos: List):
    def cal_ratio(x, y):
        return (y-x)/(x+1e-10)
    return [cal_ratio(nose_pos_, other_pos_) for nose_pos_, other_pos_ in zip(nose_pos, other_pos)]


def cal_ratio_from_nose(mode="openpose"):
    if mode == "openpose":
        pose_dict = OPENPOSE
    elif mode == "coco17":
        pose_dict = COCO_17
    idxes = [pose_dict["Nose"], pose_dict["LEye"],
             pose_dict["REye"], pose_dict["LEar"], pose_dict["REar"]]
    poses = [get_pos_alphapose(alphapose, idx) for idx in idxes]
    nose_pos, other_poses = poses[0], poses[1:]
    ratios = [ratio_from_nose(nose_pos, other_pos)
              for other_pos in other_poses]
    return ratios


def cal_pos_fromratio(joints, ratios):
    return [joint+joint*ratio for joint, ratio in zip(joints, ratios)]


def cal_bdbox(w1, w2, h1, h2):
    return abs(w1-w2)*abs(h1-h2)


def npy_point(name, npy_arr, mode="openpose"):
    if mode == "openpose":
        idx = OPENPOSE[name]
        return np.nonzero(npy_arr[:, :, idx] == 1)


def to_img_arr(width, height, result, mode="openpose"):
    if mode == "openpose":
        point_dict = OPENPOSE
    img_arr = [[] for _ in range(len(point_dict))]
    for key, value in point_dict.items():
        arr = [[0 for _ in range(width)] for _ in range(height)]
        point = result[key]
        #print(key, height-(int(point[1]*height_ratio +
        #                       height_u)), int(point[0]*width_ratio)+width_l)
        arr[height-(int(point[1]*height_ratio+height_u))
            ][width-(int(point[0]*width_ratio)+width_l)] = 1
        img_arr[value] = arr
    return np.array(img_arr)


def cal_relative_pos(A, B, C):
    return [A[0]+(-B[0]+C[0]), A[1]+(-B[1]+C[1])]


def change_to_openpose(joints, idx, ratios, npy_file):
    os.makedirs(SAVE_DIR, exist_ok=True)
    result = {}
    joints = np.array(joints)
    joints[[i for i in range(len(joints)) if i % 3 == 0]] = min_max(
        joints[[i for i in range(len(joints)) if i % 3 == 0]])
    joints[[i for i in range(len(joints)) if i % 3 == 1]] = min_max(
        joints[[i for i in range(len(joints)) if i % 3 == 1]])
    joints = list(joints)
    # 鼻は頭と首の位置の平均とする

    result["Nose"] = [(head*1.5+nose)/2.5 for head, nose in zip(get_pos_sppecg_gen(
        joints, Speech_Gen["Head"]), get_pos_sppecg_gen(joints, Speech_Gen["Nose"]))]
    # 出力されたものに無いやつを計算する
    alphapose_nose = get_pos_alphapose(alphapose, OPENPOSE["Nose"])
    result["LEye"] = cal_relative_pos(result["Nose"], get_pos_alphapose(
        alphapose, OPENPOSE["LEye"]), alphapose_nose)
    result["REye"] = cal_relative_pos(result["Nose"], get_pos_alphapose(
        alphapose, OPENPOSE["REye"]), alphapose_nose)
    result["LEar"] = cal_relative_pos(result["Nose"], get_pos_alphapose(
        alphapose, OPENPOSE["LEar"]), alphapose_nose)
    result["REar"] = cal_relative_pos(result["Nose"], get_pos_alphapose(
        alphapose, OPENPOSE["REar"]), alphapose_nose)
    result["Neck"] = get_pos_sppecg_gen(joints, Speech_Gen["Neck"])
    result["LShoulder"] = get_pos_sppecg_gen(joints, Speech_Gen["LArm"])
    result["RShoulder"] = get_pos_sppecg_gen(joints, Speech_Gen["RArm"])
    result["LElbow"] = get_pos_sppecg_gen(joints, Speech_Gen["LForeArm"])
    result["RElbow"] = get_pos_sppecg_gen(joints, Speech_Gen["RForeArm"])
    result["LWrist"] = get_pos_sppecg_gen(joints, Speech_Gen["LHand"])
    result["RWrist"] = get_pos_sppecg_gen(joints, Speech_Gen["RHand"])
    result["LHip"] = get_pos_sppecg_gen(joints, Speech_Gen["LUpLeg"])
    result["RHip"] = get_pos_sppecg_gen(joints, Speech_Gen["RUpLeg"])
    result["LKnee"] = get_pos_sppecg_gen(joints, Speech_Gen["LLeg"])
    result["RKnee"] = get_pos_sppecg_gen(joints, Speech_Gen["RLeg"])
    result["LAnkle"] = get_pos_sppecg_gen(joints, Speech_Gen["LFoot"])
    result["RAnkle"] = get_pos_sppecg_gen(joints, Speech_Gen["RFoot"])

    result_arr = to_img_arr(176, 256,  result)	
    np.save(f'{SAVE_DIR}/{str(idx).zfill(8)}.npy',
            result_arr.transpose(1, 2, 0))


def change_to_lsp(joints, idx):
    print(joints)
    result = {}
    convert_tables = {lsp: speech for lsp, speech in zip(LSP_14.keys(
    ), ["RFoot", "RLeg", "RUpLeg", "LUpLeg", "LLeg", "LFoot", "RHand", "RForeArm", "RArm", "LArm", "LForeArm", "LHand", "Neck", "Head"])}
    for convert_table in convert_tables.items():
        lsp, speech = convert_table
        result[lsp] = get_pos_alphapose(joints, Speech_Gen[speech])
        # if lsp == "Head":
        #     import pdb
        #     pdb.set_trace()
        #     result[lsp] = [result[lsp][0]*0.1, result[lsp][1]*0.1]
    result_lists = []
    for key, value in result.items():
        result_lists += [value]
    print(result_lists)
    # import pdb
    # pdb.set_trace()
    json_result = {"poses": result_lists}
    with open(f'{SAVE_DIR}/{str(idx).zfill(8)}.json', 'wt', encoding='utf-8') as f:
        json.dump(json_result, f)


def change_to_alphapose(joints, ratios, idx):
    os.makedirs(SAVE_DIR, exist_ok=True)
    result = {}
    # 鼻は頭と首の位置の平均とする
    result["Nose"] = [(head+neck)/2 for head, neck in zip(get_pos_sppecg_gen(joints,
                                                                             Speech_Gen["Head"]), get_pos_sppecg_gen(joints, Speech_Gen["Neck"]))]
    # 出力されたものに無いやつを計算する
    result["LEye"] = cal_pos_fromratio(result["Nose"], ratios[0])
    result["REye"] = cal_pos_fromratio(result["Nose"], ratios[1])
    result["LEar"] = cal_pos_fromratio(result["Nose"], ratios[2])
    result["REar"] = cal_pos_fromratio(result["Nose"], ratios[3])
    # 他のものの位置を取得
    result["LShoulder"] = get_pos_sppecg_gen(joints, Speech_Gen["LArm"])
    result["RShoulder"] = get_pos_sppecg_gen(joints, Speech_Gen["RArm"])
    result["LElbow"] = get_pos_sppecg_gen(joints, Speech_Gen["LForeArm"])
    result["RElbow"] = get_pos_sppecg_gen(joints, Speech_Gen["RForeArm"])
    result["LWrist"] = get_pos_sppecg_gen(joints, Speech_Gen["LHand"])
    result["RWrist"] = get_pos_sppecg_gen(joints, Speech_Gen["RHand"])
    result["LHip"] = get_pos_sppecg_gen(joints, Speech_Gen["LUpLeg"])
    result["RHip"] = get_pos_sppecg_gen(joints, Speech_Gen["RUpLeg"])
    result["LKnee"] = get_pos_sppecg_gen(joints, Speech_Gen["LLeg"])
    result["RKnee"] = get_pos_sppecg_gen(joints, Speech_Gen["RLeg"])
    result["LAnkle"] = get_pos_sppecg_gen(joints, Speech_Gen["LFoot"])
    result["RAnkle"] = get_pos_sppecg_gen(joints, Speech_Gen["RFoot"])
    result_lists = []
    for key, value in result.items():
        result_lists += value
        result_lists += [0.8]
    json_result = {"version": "AlphaPose v0.3",
                   "people": [{"pose_keypoints_2d": result_lists}]}
    with open(f'{SAVE_DIR}/{str(idx).zfill(8)}.json', 'wt', encoding='utf-8') as f:
        json.dump(json_result, f)


with open(speech2gesture_txt_path, "r") as f:
    lines = list(map(lambda x: list(map(float, x.split())), f.readlines()))
    ratios = cal_ratio_from_nose()

    for idx, line in enumerate(lines):
        # change_to_lsp(line, idx)
        # change_to_alphapose(line, ratios, idx)
        change_to_openpose(line, idx, ratios,
                           'fashionWOMENTees_Tanksid0000660217_4full.jpg.npy')
    # json_open.close()
