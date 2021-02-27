import torch.nn as nn
from torch import optim
import torch
from torch.nn.modules.module import Module
from transformers import BartModel, BartConfig,PreTrainedTokenizer
import torch.nn.functional as F
import MeCab
import textspan
from tokenizers import NormalizedString, PreTokenizedString, Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer

def train_custom_tokenizer(
    files: List[str], tokenizer_file: str, **kwargs
) -> BertWordPieceTokenizer:
    """ Tokenizerの学習・保存処理：custom PreTokenizer付きのTokenizerを学習・保存する。
    """

    tokenizer = BertWordPieceTokenizer(
        handle_chinese_chars=False,  # for japanese
        strip_accents=False,  # for japanese
    )
    tokenizer._tokenizer.pre_tokenizer = PreTokenizer.custom(MecabPreTokenizer())

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
        str(Path(tokenizer_file).parent), filename
    )

    return tokenizer

def load_custom_tokenizer(tokenizer_file: str) -> Tokenizer:
    """ Tokenizerのロード処理：tokenizer.json からTokenizerをロードし、custome PreTokenizerをセットする。
    """
    tokenizer = Tokenizer.from_file(tokenizer_file)
    # ダミー注入したRustベースのPreTokenizerを、custom PreTokenizerで上書き。
    tokenizer.pre_tokenizer = PreTokenizer.custom(MecabPreTokenizer())
    return tokenizer

class Model(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super(Model, self).__init__()
        self.tokenizer = PreTrainedTokenizer()#'cl-tohoku/bert-base-japanese-whole-word-masking')
        configuration = BartConfig(
            max_position_embeddings=1920, encoder_layers=2, decoder_layers=24)
        self.encoder_decoder = BartModel(configuration)
        self.linear = nn.Linear(
            1024, vocab_size)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.tokenizer(x)
        x = self.encoder_decoder(x)
        return F.log_softmax(self.linear(x))
