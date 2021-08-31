# refering https://qiita.com/mczkzk/items/fda37ac4f9ddab2d7f45
import argparse
import os
from pdb import set_trace
import cv2
from glob import glob
import face_alignment
import collections
import numpy as np
from face_swap import face_swap
import torch
from matplotlib import pyplot as plt
# from cv2 import dnn_superres
import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageDraw
from CFA import CFA

classifier = cv2.CascadeClassifier('util/lbpcascade_animeface.xml')
# base_body_img = 'intermediate/input_img.png'
landmark_detector = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D, flip_input=True, device='cpu')
# landmark_detector.face_detector.detect_from_image = classifier.detectMultiScale


def frame_imgs(cap, is_body=False):
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            return frames


def detect_from_img(img_path):
    image = cv2.imread(img_path)
    # グレースケールで処理を高速化
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray_image)
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y+h, x:x+w]
        cv2.imwrite('intermediate/input_img_face.jpg',
                    cv2.resize(face, (256, 256)))


def crop_from_movie(movie_path):
    cap = cv2.VideoCapture(movie_path)
    frames = frame_imgs(cap)
    os.makedirs('tmp')
    for i, frame in enumerate(frames):
        cv2.imwrite(f'tmp/{str(i).zfill(5)}.png', frame[:, 256:512, :])
    num_frames = len(glob('tmp/*'))
    os.system(f'rm {movie_path}')
    os.system(
        f'ffmpeg -r {num_frames} -i tmp/%05d.png -pix_fmt yuv420p -vcodec libx264 {movie_path}')
    os.system('rm -rf tmp')


def gamma_correction(img, gamma):
    # テーブルを作成する。
    table = (np.arange(256) / 255) ** gamma * 255
    # [0, 255] でクリップし、uint8 型にする。
    table = np.clip(table, 0, 255).astype(np.uint8)

    return cv2.LUT(img, table)


def detect_landmarks_anime(image, do_gamma=False, do_plot=False):
    if do_gamma:
        image = gamma_correction(image, 2.5)
    # param
    num_landmark = 24
    img_width = 128
    checkpoint_name = 'util/checkpoint_landmark_191116.pth.tar'

    # detector
    face_detector = cv2.CascadeClassifier('util/lbpcascade_animeface.xml')
    landmark_detector = CFA(output_channel_num=num_landmark + 1,
                            checkpoint_name=checkpoint_name).cuda()

    # transform
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    train_transform = [transforms.ToTensor(), normalize]
    train_transform = transforms.Compose(train_transform)

    # input image & detect face

    faces = face_detector.detectMultiScale(image)
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks = []
    for x_, y_, w_, h_ in faces:

        # adjust face size
        x = max(x_ - w_ / 8, 0)
        rx = min(x_ + w_ * 9 / 8, img.width)
        y = max(y_ - h_ / 4, 0)
        by = y_ + h_
        w = rx - x
        h = by - y

        # transform image
        img_tmp = img.crop((x, y, x+w, y+h))
        img_tmp = img_tmp.resize((img_width, img_width), Image.BICUBIC)
        img_tmp = train_transform(img_tmp)
        img_tmp = img_tmp.unsqueeze(0).cuda()

        # estimate heatmap
        heatmaps = landmark_detector(img_tmp)
        heatmaps = heatmaps[-1].cpu().detach().numpy()[0]
        landmark_idxes = [i for i in range(20)]
        # calculate landmark position
        draw = ImageDraw.Draw(img)
        for i in range(num_landmark):
            heatmaps_tmp = cv2.resize(
                heatmaps[i], (img_width, img_width), interpolation=cv2.INTER_CUBIC)
            landmark = np.unravel_index(
                np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
            landmark_y = landmark[0] * h / img_width
            landmark_x = landmark[1] * w / img_width
            if i in landmark_idxes:
                landmarks.append([landmark_x, landmark_y])
            if do_plot:
                # draw landmarks
                draw.ellipse((x + landmark_x - 2, y + landmark_y - 2, x +
                              landmark_x + 2, y + landmark_y + 2), fill=(255, 0, 0))
        if do_plot:
            # output image
            img.save('output.bmp')
    return np.array(landmarks).astype(np.int32)


def detect_landmarks(image, names, do_plot=False):
    # refering from https://qiita.com/T_keigo_wwk/items/cf56ec0bc53570105b0d
    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                  'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                  'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                  'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                  'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                  'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                  'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                  'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                  'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                  }
    preds = landmark_detector.get_landmarks(image)[-1]
    if do_plot:
        fig, ax = plt.subplots(facecolor='w', figsize=(
            4.50, 4.50), dpi=100, frameon=False)  # 入力サイズ×DPIの大きさになります。
        # 2D-Plot
        plot_style = dict(marker='None',
                          markersize=4,
                          linestyle='-',
                          lw=2)
        fig.tight_layout(pad=0)
        ax.imshow(np.ones(shape=image.shape))
        ax.imshow(image)

        for pred_type in pred_types.values():
            ax.plot(preds[pred_type.slice, 0],
                    preds[pred_type.slice, 1],
                    color=pred_type.color, **plot_style)

        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)

        # Save figure
        plt.savefig('land.jpg', dpi=100, pad_inches=0)
        plt.close()

    landmarks = []
    for key, pred_type in pred_types.items():
        # if not key in names:
        #     continue
        landmarks += preds[pred_type.slice].tolist()
    return np.array(landmarks).astype(np.int32)


def swap_faces(face_src, face_dst):
    # refering from https://github.com/wuhuikai/FaceSwap
    def get_shape(image, points):
        r = 0
        im_w, im_h = image.shape[:2]
        left, top = np.min(points, 0)
        right, bottom = np.max(points, 0)

        x, y = max(0, left - r), max(0, top - r)
        w, h = min(right + r, im_h) - x, min(bottom + r, im_w) - y
        return points, (x, y, w, h)

    names = ['face', 'eyebrow1', 'eyebrow2']
    landmarks_face_src, landmarks_face_dst = detect_landmarks_anime(
        face_src, True), detect_landmarks_anime(face_dst, True, True)
    landmarks_face_src, shape_src = get_shape(face_src, landmarks_face_src)
    landmarks_face_dst, shape_dst = get_shape(face_dst, landmarks_face_dst)
    output = face_swap(face_src, face_dst, landmarks_face_src,
                       landmarks_face_dst, shape_dst, end=9)
    return output


def convert_from_movie(body_movie_path, face_movie_path):
    cap_body = cv2.VideoCapture(body_movie_path)
    cap_face = cv2.VideoCapture(face_movie_path)

    body_frames, face_frames = frame_imgs(
        cap_body, is_body=True), frame_imgs(cap_face)
    result_movies = []
    print(body_movie_path, face_movie_path)
    for idx, body_frame in enumerate(body_frames):
        gray_image = cv2.cvtColor(body_frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray_image)
        for i, (x, y, w, h) in enumerate(faces):
            body_face, face_face = body_frame[y:y+h, x:x +
                                              w, :], face_frames[int(len(face_frames)*(idx/len(body_frames)))]

        # cv2.imwrite('body_face.png', body_face)
        # # Read the desired model
        # sr = dnn_superres.DnnSuperResImpl_create()
        # path = "LapSRN_x8.pb"

        # sr.readModel(path)

        # # Set the desired model and scale to get correct pre- and post-processing
        # sr.setModel("lapsrn", 8)

        # # Upscale the image
        # body_face = sr.upsample(body_face)

        # try:
        swapped_face = swap_faces(cv2.resize(
            face_face, (300, 300)), cv2.resize(body_face, (300, 300)))
        cv2.imwrite('test.png', swapped_face)
        body_frame[y:y+h, x:x+w, :] = cv2.resize(swapped_face, (w, h))

        result_movies.append(body_frame)
        # except:
        #     print(idx)
    os.makedirs('tmp')
    num_frames = len(result_movies)
    for i, result_img in enumerate(result_movies):
        cv2.imwrite(f'tmp/{str(i).zfill(5)}.png', result_img)
    os.system(
        f'ffmpeg -r {num_frames} -i tmp/%05d.png -pix_fmt yuv420p -vcodec libx264 aligned.mp4')
    os.system('rm -rf tmp')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('function_name',
                        type=str,
                        help='set fuction name in this file')
    parser.add_argument('-i', '--func_args',
                        nargs='*',
                        help='args in function',
                        default=[])
    args = parser.parse_args()

    # このファイル内の関数を取得
    func_dict = {k: v for k, v in locals().items() if callable(v)}
    # 引数のうち，数値として解釈できる要素はfloatにcastする
    func_args = [float(x) if x.isnumeric() else x for x in args.func_args]
    # 関数実行
    ret = func_dict[args.function_name](*func_args)
