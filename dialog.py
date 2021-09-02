import argparse
import subprocess
from subprocess import PIPE
from glob import glob


def dialog_nlp(input_txt, version):
    if version == 't5':
        DIALOG_NLP_CONTAINER_NAME = "0e3490a65e84"
        proc = subprocess.run(
            f"docker start {DIALOG_NLP_CONTAINER_NAME}", shell=True)
        with open("dialogue/t5/question/question.txt", "w") as question_txt:
            question_txt.write(input_txt)
        with open("intermediate/nlp_out.txt", "w") as output_txt:
            # import os
            # os.system(
            #     f"docker exec -w /t5 {DIALOG_NLP_CONTAINER_NAME}  python3 test.py")
            proc_1 = subprocess.run(
                f"docker exec -w /t5 {DIALOG_NLP_CONTAINER_NAME}  python3 test.py", shell=True, stdout=output_txt, text=True)
            print(proc_1.stdout)

        # TODO: text outprocess(extract only answer in english)
        from pykakasi import kakasi
        kakasi = kakasi()

        kakasi.setMode('H', 'a')
        kakasi.setMode('K', 'a')
        kakasi.setMode('J', 'a')

        conv = kakasi.getConverter()
        with open("intermediate/nlp_out.txt", "r") as f:
            responce = f.readlines()[1].replace(
                "<pad>", '').replace('</s>', '')
            print(responce)
            responce = conv.do(responce)
            print(responce)
        with open("intermediate/nlp_out_fixed.txt", "w") as f:
            f.write(
                "jsut_ver1.1/onomatopee300/wav/ONOMATOPEE300_300.wav|"+responce[1:].replace('\n', '')+".|1")


def dialog_tts(version):
    if version == 'mellotron':
        DIALOG_TTS_CONTAINER_NAME = "62f3c46f5ee3"
        subprocess.run(
            f"cp intermediate/nlp_out_fixed.txt tts/mellotron_forked/filelists/tts_dataset", shell=True)
        proc = subprocess.run(
            f"docker start {DIALOG_TTS_CONTAINER_NAME}", shell=True)
        proc_1 = subprocess.run(
            f"docker exec -w /mellotron {DIALOG_TTS_CONTAINER_NAME} python3 infer.py", shell=True)
        subprocess.run(
            f"cp tts/mellotron_forked/wavfile/test.wav intermediate/", shell=True)


def dialog_talkinghead(version):
    if version == "makeittalk":
        DIALOG_TALKINGHEAD_CONTAINER_NAME = "makeittalk"
        # TODO: cp img and audio
        subprocess.run(
            f'python3 util/face.py detect_from_img -i intermediate/input_img.png', shell=True)
        subprocess.run(
            f"cp intermediate/test.wav gesture_face/MakeItTalk/examples/", shell=True)
        subprocess.run(
            f"docker cp intermediate/input_img_face.jpg {DIALOG_TALKINGHEAD_CONTAINER_NAME}:makeittalk/examples/", shell=True)
        proc = subprocess.run(
            f"docker start {DIALOG_TALKINGHEAD_CONTAINER_NAME}", shell=True)
        proc_1 = subprocess.run(
            f"docker exec -w /makeittalk {DIALOG_TALKINGHEAD_CONTAINER_NAME} python3 test.py --jpg input_img_face.jpg", shell=True)
        proc_2 = subprocess.run(
            f"docker cp {DIALOG_TALKINGHEAD_CONTAINER_NAME}:/makeittalk/out.mp4 intermediate/face.mp4", shell=True)
        proc_3 = subprocess.run(
            f"python3 util/face.py crop_from_movie -i intermediate/face.mp4", shell=True)
        proc_4 = subprocess.run(
            f'ffmpeg -i intermediate/face.mp4 -i intermediate/test.wav -c:v copy -shortest -map 0:v:0 -map 1:a:0 "face.mp4"', shell=True)


def gesture_fromspeech(version):
    if version == "Speech_driven_gesture_generation_with_autoencoder":
        GESTURE_FROMSPEECH_CONTAINER_NAME = "gesture_location"
        proc = subprocess.run(
            f"docker start {GESTURE_FROMSPEECH_CONTAINER_NAME}", shell=True)
        proc_1 = subprocess.run(
            f"docker cp intermediate/test.wav {GESTURE_FROMSPEECH_CONTAINER_NAME}:/gesture_location/", shell=True)
        proc_2 = subprocess.run(
            f"docker exec -w /gesture_location {GESTURE_FROMSPEECH_CONTAINER_NAME} python3 preprocess_own_data.py test.wav", shell=True)
        proc_3 = subprocess.run(
            f"docker exec -w /gesture_location {GESTURE_FROMSPEECH_CONTAINER_NAME} python3 predict.py model_500ep_posvel_60.hdf5 test.npy test.txt", shell=True)
        proc_4 = subprocess.run(
            f"docker exec -w /gesture_location {GESTURE_FROMSPEECH_CONTAINER_NAME} python3 motion_repr_learning/ae/decode.py dataset test.txt output/test_decode.txt -restore=True -pretrain=False -layer1_width=325 -chkpt_dir checkpoints -batch_size=8 ", shell=True)
        proc_5 = subprocess.run(
            f"docker exec -w /gesture_location {GESTURE_FROMSPEECH_CONTAINER_NAME} python3 helpers/remove_velocity.py -g output", shell=True)
        proc_6 = subprocess.run(
            f"docker cp {GESTURE_FROMSPEECH_CONTAINER_NAME}:/gesture_location/output/no_vel/test_decode.txt ./intermediate/", shell=True)


def gesture_location_to_img(version):
    if version == "adgan":
        GESTURE_LOCATION_TO_IMG_CONTAINER_NAME = "adgan"
        proc = subprocess.run(
            f"docker start {GESTURE_LOCATION_TO_IMG_CONTAINER_NAME}", shell=True)
        proc_1 = subprocess.run(
            "python3 util/make_joint_points.py", shell=True)
        proc_2 = subprocess.run(
            f"docker cp intermediate/location_adgan  {GESTURE_LOCATION_TO_IMG_CONTAINER_NAME}:/mounted/deepfashion", shell=True)
        proc_3 = subprocess.run(
            f"docker exec -w /mounted/deepfashion {GESTURE_LOCATION_TO_IMG_CONTAINER_NAME} python3 make_pair_csv.py", shell=True)
        proc_4 = subprocess.run(
            f"docker exec -w /mounted {GESTURE_LOCATION_TO_IMG_CONTAINER_NAME} bash ./scripts/test.sh", shell=True)
        proc_5 = subprocess.run(
            f"docker cp {GESTURE_LOCATION_TO_IMG_CONTAINER_NAME}:/mounted/results intermediate/", shell=True)
        for mp4_file in glob('intermediate/*.mp4'):
            if 'face.mp4' in mp4_file:
                continue
            subprocess.run(f'rm {mp4_file}', shell=True)
        num_img = len(glob(
            'intermediate/results/fashion_AdaGen_sty512_nres8_lre3_SS_fc_vgg_cxloss_ss_merge3/test_800/images/*'))
        proc_6 = subprocess.run(
            f"ffmpeg -r {num_img} -i intermediate/results/fashion_AdaGen_sty512_nres8_lre3_SS_fc_vgg_cxloss_ss_merge3/test_800/images/fashionWOMENTees_Tanksid0000660217_4full.jpg___fashionWOMENTees_Tanksid0000660217_4full_%08d.jpg_vis.jpg -pix_fmt yuv420p -vcodec libx264 intermediate/fashion.mp4", shell=True)
        proc_7 = subprocess.run(
            f'ffmpeg -i intermediate/fashion.mp4 -vf crop=176:150:0:0 intermediate/fashion_cropped.mp4', shell=True)


def gesture_img_to_img(version):
    if version == "articulated_animation":
        GESTURE_IMG_TO_IMG_CONTAINER_NAME = 'gesture_img2img'
        proc = subprocess.run(
            f"docker start {GESTURE_IMG_TO_IMG_CONTAINER_NAME}", shell=True)
        proc_1 = subprocess.run(
            f"docker cp intermediate/fashion_cropped.mp4 {GESTURE_IMG_TO_IMG_CONTAINER_NAME}:fomm/sup-mat/", shell=True)
        proc_2 = subprocess.run(
            f"docker cp intermediate/input_img.png {GESTURE_IMG_TO_IMG_CONTAINER_NAME}:fomm/sup-mat/", shell=True)
        proc_3 = subprocess.run(
            f"docker exec -w /fomm {GESTURE_IMG_TO_IMG_CONTAINER_NAME} python3 demo.py --config config/ted-youtube384.yaml --checkpoint checkpoints/ted-youtube384.pth --source_image sup-mat/input_img.png --driving_video sup-mat/fashion_cropped.mp4 --mode relative", shell=True)
        proc_4 = subprocess.run(
            f"docker cp {GESTURE_IMG_TO_IMG_CONTAINER_NAME}:/fomm/result.mp4 intermediate/body.mp4", shell=True)


def align_face_body():
    subprocess.run(f'rm -rf out.mp4', shell=True)
    subprocess.run(f'rm -rf aligned.mp4', shell=True)
    proc = subprocess.run(f'docker start face_util', shell=True)
    proc_1 = subprocess.run(
        f'docker exec -w /mounted face_util python3 util/face.py convert_from_movie -i intermediate/body.mp4 intermediate/face.mp4', shell=True)
    proc_2 = subprocess.run(
        f'ffmpeg -i aligned.mp4 -i intermediate/test.wav -c:v copy -shortest -map 0:v:0 -map 1:a:0 "out.mp4"', shell=True)
    # subprocess.run(
    #     f'python3 util/face.py convert_from_movie -i intermediate/body.mp4 intermediate/face.mp4', shell=True)


if __name__ == "__main__":
    # refered from https://qiita.com/kenichi-hamaguchi/items/dda5532f3b218142e7c9
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
