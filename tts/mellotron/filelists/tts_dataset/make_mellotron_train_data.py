import glob
from pykakasi import kakasi
import os
import pdb
import soundfile as sf
start_num = 1
kakasi = kakasi()

kakasi.setMode('H', 'a')
kakasi.setMode('K', 'a')
kakasi.setMode('J', 'a')
kakasi.setMode('E', 'a')
kakasi.setMode('s', False)

conv = kakasi.getConverter()


def make_jsut():
    global start_num
    files = glob.glob('jsut_ver1.1/*')
    with open('info_jsut.txt', 'w') as info:
        for file in files:
            if "." in file.split("/")[-1]:
                continue
            with open(file+'/transcript_utf8.txt', 'r') as f:
                for line in f.readlines():
                    filename, word = line.split(":")
                    if not os.path.exists(file+"/wav/"+filename + ".wav"):
                        continue
                    data, fs = sf.read(f'{file}/wav/{filename}.wav')
                    sf.write(f'{file}/wav/{filename}.wav',
                             data, fs, subtype='PCM_16')
                    info.write(file+"/wav/"+filename + ".wav|"+conv.do(
                        word.replace('\n', "")).replace(" ", "")+"|"+str(start_num)+"\n")
            f.close()
    info.close()
    start_num += 1


def make_tsukuyomi():
    global start_num
    files = 'つくよみちゃんコーパスVol.1声優統計コーパス/台本と補足資料/台本テキスト/01 補足なし台本（JSUTコーパス・JVSコーパス版）.txt'
    with open('info_tsukuyomi.txt', 'w') as info:
        with open(files, 'r') as f:
            for line in f.readlines():
                filename, word = line.split(":")
                if not os.path.exists('つくよみちゃんコーパスVol.1声優統計コーパス/02WAV/' +
                                      filename + ".wav"):
                    continue
                data, fs = sf.read(
                    f'つくよみちゃんコーパスVol.1声優統計コーパス/02WAV/{filename}.wav')
                sf.write(f'つくよみちゃんコーパスVol.1声優統計コーパス/02WAV/{filename}".wav',
                         data, fs, subtype='PCM_16')
                info.write("つくよみちゃんコーパスVol.1声優統計コーパス/02WAV/" +
                           filename+".wav|"+conv.do(word.replace('\n', "")).replace(" ", "")+"|"+str(start_num)+"\n")
        f.close()
    info.close()
    start_num += 1


def make_jvc():
    global start_num
    files = glob.glob('jvs_ver1/*')
    with open('info_jvc.txt', 'w') as info:
        for file in files:
            if "." in file.split("/")[-1]:
                continue
            file_files = glob.glob(file+"/*")
            for file_file in file_files:
                with open(file_file+"/transcripts_utf8.txt", "r") as f:
                    for line in f.readlines():
                        filename, word = line.split(":")
                        if not os.path.exists(file_file+"/wav24kHz16bit/"+filename + ".wav"):
                            continue
                        data, fs = sf.read(
                            f'{file_file}/wav24kHz16bit/{filename}.wav')
                        sf.write(f'{file_file}/wav24kHz16bit/{filename}.wav',
                                 data, fs, subtype='PCM_16')
                        info.write(file_file+"/wav24kHz16bit/" +
                                   filename+".wav|"+conv.do(word.replace('\n', "")).replace(" ", "")+"|"+str(start_num)+"\n")
                f.close()
            start_num += 1
    info.close()


make_jsut()
make_jvc()
make_tsukuyomi()
