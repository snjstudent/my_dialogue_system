import torch
import glob
import pdb
import re
import neologdn
from tokenizers import NormalizedString, PreTokenizedString, Tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer
import textspan
from typing import List, Optional
import pathlib
import numpy as np


# データセットの定義
class TweetReplyDataSet(torch.utils.data.Dataset):
    def __init__(self, mode, do_preprocess: bool = True) -> None:
        super().__init__()
        reply_txtfiles = glob.glob(f'../../../tweet/{mode}/*')
        reply_txtfiles = sorted(reply_txtfiles)
        self.tokenizer = T5Tokenizer.from_pretrained(
            "sonoisa/t5-base-japanese")
        self.do_preprocess = do_preprocess
        self.dialog = self._sep_req_res(reply_txtfiles)

    def _sep_req_res(self, reply_txtfiles: list) -> dict:
        dialog_dict = {'REQ': [], 'RES': []}
        for txt_file in reply_txtfiles:
            with open(txt_file, "r", errors='ignore') as f:
                l = f.readlines()
                for line in l:
                    try:
                        speaker, speak = line.split(":", 1)
                        if self.do_preprocess:
                            speak = self._preprocess(speak)
                        dialog_dict[speaker].append(speak)
                    except:
                        print("Error : ", line)
        assert len(dialog_dict['REQ']) == len(dialog_dict['RES']), print(
            "something wrong with dialogue dataset")
        return dialog_dict

    @classmethod
    def _preprocess(self, sentence: str) -> str:
        return neologdn.normalize(re.sub(r'\d+', '0', sentence)).replace("\n", "")

    def __len__(self):
        return len(self.dialog['REQ'])

    # トークンの挙動がどう変わるかをチェックしておく
    def tokenize(self, speak):
        speak = speak + "</s>"
        speak = self.tokenizer(speak, max_length=200, truncation=True,
                               padding="max_length", return_tensors="pt")
        return speak

    def _reshape_tokens(self, intput_token):
        intput_token["input_ids"] = intput_token["input_ids"][0]
        intput_token["attention_mask"] = intput_token["attention_mask"][0]
        return intput_token

    def __getitem__(self, idx):

        speak, responce = self.dialog['REQ'][idx], self.dialog['RES'][idx]
        speak, responce = self.tokenize(speak), self.tokenize(responce)
        speak, responce = self._reshape_tokens(
            speak), self._reshape_tokens(responce)
        # one-hot torch.nn.functional.one_hot(torch.LongTensor(responce), num_classes=50000
        return speak, responce


class MeidaiReplyDataSet(torch.utils.data.Dataset):
    def __init__(self, mode, do_preprocess: bool = True) -> None:
        super().__init__()
        reply_txtfiles = glob.glob(f'../../../meidai/{mode}/*')
        reply_txtfiles = sorted(reply_txtfiles)
        self.tokenizer = T5Tokenizer.from_pretrained(
            "sonoisa/t5-base-japanese")
        self.do_preprocess = do_preprocess
        self.dialog = self._sep_req_res(reply_txtfiles)

    def _sep_req_res(self, reply_txtfiles: list) -> dict:
        dialog_dict = {'REQ': [], 'RES': []}
        convert_speaker_dict = {'input': 'REQ', 'output': 'RES'}
        for txt_file in reply_txtfiles:
            with open(txt_file, "r", errors='ignore') as f:
                l = f.readlines()
                for line in l:
                    try:
                        speaker, speak = line.split(":", 1)
                        speaker, speak = convert_speaker_dict[speaker], speak[1:]
                        if self.do_preprocess:
                            speak = self._preprocess(speak)
                        dialog_dict[speaker].append(speak)
                    except:
                        print("Error : ", line)
        assert len(dialog_dict['REQ']) == len(dialog_dict['RES']), print(
            "something wrong with dialogue dataset")
        return dialog_dict

    @classmethod
    def _preprocess(self, sentence: str) -> str:
        sentence = sentence.replace('***', '<UNK>')
        sentence = sentence.replace('*', '')
        s = sentence
        # 参考元：https://qiita.com/gacky01/items/26cd642731e3eddde60d
        while s.find("（") != -1:
            start_1 = s.find("（")
            if s.find("）") != -1:
                end_1 = s.find("）")
                if start_1 >= end_1:
                    s = s.replace(s[end_1], "")
                else:
                    s = s.replace(s[start_1:end_1+1], "")
                if len(s) == 0:
                    continue
            else:
                s = s[0:start_1]

        while s.find("［") != -1:
            start_2 = s.find("［")
            if s.find("］") != -1:
                end_2 = s.find("］")
                s = s.replace(s[start_2:end_2+1], "")
            else:
                s = s[0:start_2]

        while s.find("＜") != -1:
            start_3 = s.find("＜")
            if s.find("＞") != -1:
                end_3 = s.find("＞")
                s = s.replace(s[start_3:end_3+1], "")
            else:
                s = s[0:start_3]

        while s.find("【") != -1:
            start_4 = s.find("【")
            if s.find("】") != -1:
                end_4 = s.find("】")
                s = s.replace(s[start_4:end_4+1], "")
            else:
                s = s[0:start_4]
        sentence = s
        return neologdn.normalize(re.sub(r'\d+', '0', sentence)).replace("\n", "")

    def __len__(self):
        return len(self.dialog['REQ'])

    # トークンの挙動がどう変わるかをチェックしておく
    def tokenize(self, speak):
        speak = speak + "</s>"
        speak = self.tokenizer(speak, max_length=200, truncation=True,
                               padding="max_length", return_tensors="pt")
        return speak

    def _reshape_tokens(self, intput_token):
        intput_token["input_ids"] = intput_token["input_ids"][0]
        intput_token["attention_mask"] = intput_token["attention_mask"][0]
        return intput_token

    def __getitem__(self, idx):
        speak, responce = self.dialog['REQ'][idx], self.dialog['RES'][idx]
        speak, responce = self.tokenize(speak), self.tokenize(responce)
        speak, responce = self._reshape_tokens(
            speak), self._reshape_tokens(responce)
        # one-hot torch.nn.functional.one_hot(torch.LongTensor(responce), num_classes=50000
        return speak, responce


class J_ToccDataSet(torch.utils.data.Dataset):
    def __init__(self, mode, do_preprocess: bool = True) -> None:
        super().__init__()
        reply_txtfiles = glob.glob(f'../../../j_tocc/{mode}/*')
        reply_txtfiles = sorted(reply_txtfiles)
        self.tokenizer = T5Tokenizer.from_pretrained(
            "sonoisa/t5-base-japanese")
        self.do_preprocess = do_preprocess
        self.dialog = self._sep_req_res(reply_txtfiles)

    def _sep_req_res(self, reply_txtfiles: list) -> dict:
        dialog_dict = {'REQ': [], 'RES': []}
        for txt_file in reply_txtfiles:
            with open(txt_file, "r", errors='ignore') as f:
                l = f.readlines()
                for line in l:
                    try:
                        speaker, speak = line.split("：", 1)
                        if speaker[-2] == '1':
                            speaker = 'REQ'
                        elif speaker[-2] == '2':
                            speaker = 'RES'
                        if self.do_preprocess:
                            speak = self._preprocess(speak)
                        dialog_dict[speaker].append(speak)
                    except:
                        print("Error : ", line)
            if not len(dialog_dict['REQ']) == len(dialog_dict['RES']):
                del dialog_dict['REQ' if len(
                    dialog_dict['REQ']) > len(dialog_dict['RES']) else 'RES'][-1]
        assert len(dialog_dict['REQ']) == len(dialog_dict['RES']), print(
            "something wrong with dialogue dataset")
        return dialog_dict

    @classmethod
    def _preprocess(self, sentence: str) -> str:
        sentence = sentence.replace('●', '')
        s = sentence
        # 参考元：https://qiita.com/gacky01/items/26cd642731e3eddde60d
        while s.find("（") != -1:
            start_1 = s.find("（")
            if s.find("）") != -1:
                end_1 = s.find("）")
                if start_1 >= end_1:
                    s = s.replace(s[end_1], "")
                else:
                    s = s.replace(s[start_1:end_1+1], "")
                if len(s) == 0:
                    continue
            else:
                s = s[0:start_1]
        while s.find("【") != -1:
            start_4 = s.find("【")
            if s.find("】") != -1:
                end_4 = s.find("】")
                s = s.replace(s[start_4:end_4+1], "")
            else:
                s = s[0:start_4]
        sentence = s
        return neologdn.normalize(re.sub(r'\d+', '0', sentence)).replace("\n", "")

    def __len__(self):
        return len(self.dialog['REQ'])

    # トークンの挙動がどう変わるかをチェックしておく
    def tokenize(self, speak):
        speak = speak + "</s>"
        speak = self.tokenizer(speak, max_length=200, truncation=True,
                               padding="max_length", return_tensors="pt")
        return speak

    def _reshape_tokens(self, intput_token):
        intput_token["input_ids"] = intput_token["input_ids"][0]
        intput_token["attention_mask"] = intput_token["attention_mask"][0]
        return intput_token

    def __getitem__(self, idx):
        speak, responce = self.dialog['REQ'][idx], self.dialog['RES'][idx]
        speak, responce = self.tokenize(speak), self.tokenize(responce)
        speak, responce = self._reshape_tokens(
            speak), self._reshape_tokens(responce)
        # one-hot torch.nn.functional.one_hot(torch.LongTensor(responce), num_classes=50000
        return speak, responce


def get_dataloader(loader_category: str, mode: str, batch_size: int, do_preprocess: bool = True) -> torch.utils.data.DataLoader:
    if loader_category == "tweet_reply":
        tweetreply_dataset = TweetReplyDataSet(mode, do_preprocess)
        TweetReplyDataLoader = torch.utils.data.DataLoader(
            tweetreply_dataset, batch_size=batch_size, shuffle=True)
        return TweetReplyDataLoader
    elif loader_category == "meidai":
        meidaireply_dataset = MeidaiReplyDataSet(mode, do_preprocess)
        MeidaiReplyDataLoader = torch.utils.data.DataLoader(
            meidaireply_dataset, batch_size=batch_size, shuffle=True)
        return MeidaiReplyDataLoader

    elif loader_category == "j-tocc":
        j_tocc_datataSet = J_ToccDataSet(mode, do_preprocess)
        J_ToccDataLoader = torch.utils.data.DataLoader(
            j_tocc_datataSet, batch_size=batch_size, shuffle=True)
        return J_ToccDataLoader

    else:
        print("No matching category!")
