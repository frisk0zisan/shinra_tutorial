import argparse

from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForTokenClassification, BertJapaneseTokenizer

from code.data import ShinraDataset, my_collate_fn
from code.util import decode_output
from shinra_jp_scorer import get_score

device = "cuda" if torch.cuda.is_available else "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16, help="batch size during training")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--epoch", type=int, default=10, help="epoch")
    parser.add_argument("--model_path", type=str, default="./bert.model", help="path for saving model")
    parser.add_argument("--input_path", type=str, help="path for input path")
    parser.add_argument('--valid', action='store_true')
    args = parser.parse_args()
    return args


def train(model, dataset, lr=5e-5, batch_size=16, epoch=10, is_valid=False):
    if is_valid is True:
        n_samples = len(dataset)
        train_size = int(len(dataset) * 0.8)
        val_size = n_samples - train_size 

        # shuffleしてから分割
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=my_collate_fn)
    else :
        train_dataset = dataset

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=my_collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    print("--- train ---")
    for e in range(epoch):
        print(f"epoch : {e+1} ")
        all_loss = 0
        for tokens, labels, infos in tqdm(train_dataloader):
            optimizer.zero_grad()

            input_x = pad_sequence([torch.tensor(token)
                                    for token in tokens], batch_first=True, padding_value=0).to(device)
            input_y = pad_sequence([torch.tensor(label)
                                    for label in labels], batch_first=True, padding_value=0).to(device)

            mask = input_x > 0
            output = model(input_x, labels=input_y, attention_mask=mask)
            loss = output[0]

            loss.backward()
            optimizer.step()
            all_loss += loss.item()

        losses.append(all_loss / len(train_dataloader))
    
    if is_valid is True:
        ans_labels = []
        ans_infos = []
        for tokens, labels, infos in val_dataloader:
            labels = [[val_dataset.dataset.id2label[l] for l in label] for label in labels]
            ans_labels.extend(labels)
            ans_infos.extend(infos)
        
        val_preds, val_infos  = predict(model, val_dataset)
        f = open('val_preds.txt', 'w')
        for x in val_preds:
            f.write(str(x) + "\n")
        f.close()

        f = open('val_infos.txt', 'w')
        for x in val_infos:
            f.write(str(x) + "\n")
        f.close()
        
        result = decode_output(val_preds, val_infos)
        answer = decode_output(ans_labels, ans_infos)
        get_score(answer, result, error_path = "error", score_path = "score")

    return losses

def predict(model, dataset):
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=16, collate_fn=my_collate_fn)

        losses = []
        preds = []
        test_infos = []
        print("--- valid ---")
        for tokens, _, infos in tqdm(dataloader):
            input_x = pad_sequence([torch.tensor(token)
                                    for token in tokens], batch_first=True, padding_value=0).to(device)

            mask = input_x > 0
            output = model(input_x, attention_mask=mask)

            scores, idxs = torch.max(output[0], dim=-1)

            labels = [idxs[i][mask[i]].tolist() for i in range(idxs.size(0))]
            labels = [[dataset.dataset.id2label[l] for l in label] for label in labels]
            preds.extend(labels)

            test_infos.extend(infos)

    return preds, test_infos

if __name__ == "__main__":
    # load argument
    args = parse_arguments()

    # load tokenizer
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

    # load dataset
    dataset = ShinraDataset(args.input_path, tokenizer)

    # load model
    model = BertForTokenClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=len(dataset.label_vocab)).to(device)

    # train model
    losses = train(model, dataset, lr=args.lr, batch_size=args.batch_size, epoch=args.epoch, is_valid=args.valid)
    
    # save model
    torch.save(model.state_dict(), args.model_path)
