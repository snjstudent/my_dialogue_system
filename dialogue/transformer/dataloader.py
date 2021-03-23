import torch
import glob
import pdb
import re
import neologdn
from tokenizers import NormalizedString, PreTokenizedString, Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer
import MeCab
import textspan
from typing import List, Optional
import pathlib
import numpy as np
# トークナイザの定義
# 引用元 https://tech.mntsq.co.jp/entry/2021/02/26/120013


class MecabPreTokenizer:
    def __init__(
        self,
        mecab_dict_path: Optional[str] = None,
    ):
        """Construct a custom PreTokenizer with MeCab for huggingface tokenizers."""

        mecab_option = (
            f"-Owakati -d {mecab_dict_path}"
            if mecab_dict_path is not None
            else "-Owakati"
        )
        self.mecab = MeCab.Tagger(mecab_option)

    def tokenize(self, sequence: str) -> List[str]:
        return self.mecab.parse(sequence).strip().split(" ")

    def custom_split(
        self, i: int, normalized_string: NormalizedString
    ) -> List[NormalizedString]:
        """See. https://github.com/huggingface/tokenizers/blob/b24a2fc/bindings/python/examples/custom_components.py"""
        text = str(normalized_string)
        tokens = self.tokenize(text)
        tokens_spans = textspan.get_original_spans(tokens, text)
        return [
            normalized_string[st:ed]
            for char_spans in tokens_spans
            for st, ed in char_spans
        ]

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.custom_split)


def train_custom_tokenizer(
    files: List[str], tokenizer_file: str, **kwargs
) -> BertWordPieceTokenizer:
    """ Tokenizerの学習・保存処理：custom PreTokenizer付きのTokenizerを学習・保存する。
    """

    tokenizer = BertWordPieceTokenizer(
        handle_chinese_chars=False,  # for japanese
        strip_accents=False,  # for japanese
    )
    tokenizer._tokenizer.pre_tokenizer = PreTokenizer.custom(
        MecabPreTokenizer())

    # 与えられたコーパスファイル集合からサブワード分割を学習
    tokenizer.train(files, **kwargs)

    # vocab情報に加えて、前処理等パラメータ情報を含んだトークナイザ設定のJSONを保存
    # NOTE: Pythonで書かれたcustom PreTokenizerはシリアライズできないので、RustベースのPreTokenizerをダミー注入してシリアライズ
    # JSONにはダミーのPreTokenizerが記録されるので、ロード時にcustom PreTokenizerを再設定する必要がある。
    tokenizer._tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.save(tokenizer_file)

    # (Optional) .txt形式のvocabファイルは f"vocab-{filename}.txt" で保存される（外部の処理で欲しい場合）
    filename = "wordpiece"
    model_files = tokenizer._tokenizer.model.save(
        str(pathlib.Path(tokenizer_file).parent), filename
    )

    return tokenizer


def load_custom_tokenizer(tokenizer_file: str) -> Tokenizer:
    """ Tokenizerのロード処理：tokenizer.json からTokenizerをロードし、custome PreTokenizerをセットする。
    """
    tokenizer = Tokenizer.from_file(tokenizer_file)
    # ダミー注入したRustベースのPreTokenizerを、custom PreTokenizerで上書き。
    tokenizer.pre_tokenizer = PreTokenizer.custom(MecabPreTokenizer())
    return tokenizer


settings = dict(
    vocab_size=50000,
    min_frequency=1,
    limit_alphabet=1000,
)
# データセットの定義


class TweetReplyDataSet(torch.utils.data.Dataset):
    def __init__(self, do_preprocess: bool = True) -> None:
        super().__init__()
        reply_txtfiles = glob.glob('../../../tweet/*')
        reply_txtfiles = sorted(reply_txtfiles)
        train_custom_tokenizer(reply_txtfiles, "tokenizer.json")
        self.tokenizer = load_custom_tokenizer("tokenizer.json")
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
                        speak = "[CLS]"+speak+"[SEP]"
                        speak = self.tokenizer.encode(
                            speak).ids
                        speak = speak + [0 for _ in range(200 - len(speak))]
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

    def __getitem__(self, idx):
        speak, responce = self.dialog['REQ'][idx], self.dialog['RES'][idx]
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
