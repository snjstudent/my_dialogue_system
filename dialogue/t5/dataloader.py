import torch
import glob
import pdb
import re
import neologdn
from tokenizers import NormalizedString, PreTokenizedString, Tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer
import MeCab
import textspan
from typing import List, Optional
import pathlib
import numpy as np


# データセットの定義
class TweetReplyDataSet(torch.utils.data.Dataset):
    def __init__(self, do_preprocess: bool = True) -> None:
        super().__init__()
        reply_txtfiles = glob.glob('./tweet/*')
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
                        dialog_dict[speaker].append(speak)
                    except:
                        print("Error : ", line)
                        del dialog_dict['REQ' if len(dialog_dict['REQ']) > len(
                            dialog_dict['RES']) else 'RES'][-1]

        assert len(dialog_dict['REQ']) == len(dialog_dict['RES']), print(
            "something wrong with dialogue dataset")
        return dialog_dict

    @classmethod
    def _preprocess(self, sentence: str) -> str:
        return neologdn.normalize(re.sub(r'\d+', '0', sentence))

    def __len__(self):
        return len(self.dialog['REQ'])

    # トークンの挙動がどう変わるかをチェックしておく
    def tokenize(self, speak):
        speak = self._preprocess(speak)
        speak = "[CLS]"+speak+"[SEP]"
        speak = self.tokenizer.encode(
            speak).ids
        speak = speak + [0 for _ in range(200 - len(speak))]
        return speak

    def __getitem__(self, idx):
        speak, responce = self.dialog['REQ'][idx], self.dialog['RES'][idx]
        speak, responce = self.tokenize(speak), self.tokenize(responce)
        # one-hot torch.nn.functional.one_hot(torch.LongTensor(responce), num_classes=50000)
        return np.array(speak), np.array(responce)


def get_dataloader(loader_category: str, batch_size: int, do_preprocess: bool = True) -> torch.utils.data.DataLoader:
    if loader_category == "tweet_reply":
        tweetreply_dataset = TweetReplyDataSet(do_preprocess=do_preprocess)
        TweetReplyDataLoader = torch.utils.data.DataLoader(
            tweetreply_dataset, batch_size=batch_size, shuffle=True)
        return TweetReplyDataLoader

    elif loader_category == "persona":
        return NotImplementedError

    else:
        print("No matching category!")
