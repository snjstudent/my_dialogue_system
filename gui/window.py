from datetime import time
from typing import Text
from PySimpleGUI.PySimpleGUI import Window, theme
import cv2
import PySimpleGUI as sg
import os

sg.theme('Dark Blue 3')


def select_window():
    select_modes = ['face', 'body']
    layout = [[sg.Button(mode,  size=(10, 1), key=mode)]
              for mode in select_modes]
    window = sg.Window('表示部分選択', layout, size=(
        200, 200), element_justification='c')
    while True:
        event, values = window.read()
        # 選択されてないときはeventがnullを返すかわからないので、保険でリスト内判定
        if event in select_modes:
            break
    window.close()
    return event


def dialog_window(mode):
    def read_reply_fromfile(txt_file):
        with open(txt_file, 'r') as f:
            for i, line in enumerate(f):
                if i != 1:
                    continue
                return line.replace('<pad>', '').replace('</s>\n', '')
    cap = cv2.VideoCapture(f'gui/hello_{mode}.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    movie_width, movie_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    layout = [[sg.Image(filename='',  key='-IMAGE-')], [
        sg.Text('', font=('HGゴシックE', 15), justification='center', size=(80, 1), key='-reply-')], [sg.Input('', justification='center', size=(80, 1), key='input_text')]]
    window = sg.Window('DiaLog', layout, size=(
        movie_width+100, movie_height+150), return_keyboard_events=True, element_justification='c')
    while True:
        event, values = window.read(timeout=1000//fps)
        ret, frame = cap.read()
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        # フレーム切れ、なおかつ文が入力されている
        if event == 'Return:36' and not ret:
            input_text = values['input_text']
            print(input_text)
            os.system(f'sh pipeline_all.sh {input_text} {mode}')
            reply = read_reply_fromfile('intermediate/nlp_out.txt')
            window['-reply-'].update(reply)
            cap = cv2.VideoCapture(f'out.mp4') if mode == 'body' else cv2.VideoCapture(
                f'intermediate/face.mp4')
            fps = cap.get(cv2.CAP_PROP_FPS)
            continue
        # フレーム切れ
        elif not ret:
            continue
        # その他
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['-IMAGE-'].update(data=imgbytes)
    window.close()


if __name__ == '__main__':
    mode = select_window()
    dialog_window(mode)
