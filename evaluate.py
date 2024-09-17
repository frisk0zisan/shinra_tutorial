import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForTokenClassification, BertJapaneseTokenizer

from code.data import ShinraDataset, my_collate_fn
from code.util import decode_output, print_shinra_format
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available else "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./bert.model", help="path for saving model")
    parser.add_argument("--input_path", type=str, help="path for input datsaet")
    parser.add_argument("--output_path", type=str, help="path for output datsaet")

    args = parser.parse_args()
    return args


def predict(model, dataset):
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=16, collate_fn=my_collate_fn)

        losses = []
        preds = []
        test_infos = []
        for tokens, _, infos in tqdm(dataloader):
            input_x = pad_sequence([torch.tensor(token)
                                    for token in tokens], batch_first=True, padding_value=0).to(device)

            mask = input_x > 0
            output = model(input_x, attention_mask=mask)

            output = output[0][:,1:,:]
            mask = mask[:, 1:]

            scores, idxs = torch.max(output, dim=-1)

            labels = [idxs[i][mask[i]].tolist() for i in range(idxs.size(0))]
            labels = [[dataset.id2label[l] for l in label] for label in labels]
            preds.extend(labels)

            test_infos.extend(infos)

    return preds, test_infos


if __name__ == "__main__":
    # load argument
    args = parse_arguments()

    # load tokenizer
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

    # load dataset
    dataset = ShinraDataset(args.input_path, tokenizer, is_train=False)

    # load model
    model = BertForTokenClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=len(dataset.label_vocab)).to(device)
    model.load_state_dict(torch.load(args.model_path))

    preds, infos = predict(model, dataset)
    outputs = decode_output(preds, infos)
    print_shinra_format(outputs, args.output_path)
    #torch.save(model.state_dict(), args.model_path)

