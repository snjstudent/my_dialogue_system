import argparse
import subprocess
from subprocess import PIPE


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
        DIALOG_TALKINGHEAD_CONTAINER_NAME = "356b7494c33c"
        # TODO: cp img and audio
        subprocess.run(
            f"cp intermediate/test.wav gesture_face/MakeItTalk/examples/", shell=True)
        proc = subprocess.run(
            f"docker start {DIALOG_TALKINGHEAD_CONTAINER_NAME}", shell=True)
        proc_1 = subprocess.run(
            f"docker exec -w /makeittalk {DIALOG_TALKINGHEAD_CONTAINER_NAME} python3 test.py --jpg harry.jpg", shell=True)


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
