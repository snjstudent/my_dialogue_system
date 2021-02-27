import torch
import glob
import pdb
import re
import neologdn

# データセットの定義


class TweetReplyDataSet(torch.utils.data.Dataset):
    def __init__(self, do_preprocess: bool = True) -> None:
        super().__init__()
        reply_txtfiles = glob.glob('../../../tweet/*')
        # import pdb
        # pdb.set_trace()
        reply_txtfiles = sorted(reply_txtfiles)
        self.dialog = self._sep_req_res(reply_txtfiles)
        print(len(self.dialog['REQ']))
        self.do_preprocess = do_preprocess

    @classmethod
    def _sep_req_res(self, reply_txtfiles: list) -> dict:
        dialog_dict = {'REQ': [], 'RES': []}
        for txt_file in reply_txtfiles:
            with open(txt_file,"r",errors='ignore') as f:
                l = f.readlines()
                for line in l:
                    speaker, speak = line.split(":", 1)
                    dialog_dict[speaker].append(speak)
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
        if self.do_preprocess:
            speak, responce = self._preprocess(
                speak), self._preprocess(responce)
        return speak, responce


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
