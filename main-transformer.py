import time
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_numericalizer
import os
from torch.utils.data import DataLoader
from neural_models import CNNClassifier, TransformerClassifier,LSTMClassifier, ModelClassifier
import warnings

warnings.filterwarnings('ignore')

seed = 0
tokenizer_type="char" # the type of SentencePiece model, including unigram, bpe, char, word.
vocab_size = 30
Methods = ["random", "majority","deeplearning"]
method = Methods[2]
max_num_epochs = 100
log_interval = 1
eval_interval = 1
learning_rate=5e-5#0.5
batch_size=4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def read_data():
    xtrain_path = "./xtrain_obfuscated.txt"
    ytrain_path = "./ytrain.txt"
    xtest_path = "./xtest_obfuscated.txt"

    with open(xtrain_path,"r") as f:
        xtrain = f.read().strip().lower().split("\n")


    with open(ytrain_path,"r") as f:
        ytrain = [int(label) for label in f.read().strip().lower().split("\n")]
        

    with open(xtest_path,"r") as f:
        xtest = f.read().strip().lower().split("\n")

    assert (len(xtrain) ==len(ytrain))

    print(f"# train samples: {len(xtrain)}\n# test samples: {len(xtest)}")
    print()

    labels = dict(sorted(collections.Counter(ytrain).items()))
    print("checking the distribution of training labels (balanced vs imbalanced)?")
    plt.bar(range(len(labels)), list(labels.values()), align='center')
    plt.xticks(range(len(labels)), list(labels.keys()))
    plt.title('label distribution in xtrain.txt')
    plt.show()
    print(f"all: {labels}")
    print()

    print("split the train set into two sets: train, and dev with ratio 0.9/0.1 respectively.")
    print("why 0.9/0.1 ratio? because the size of dev set becomes similar to the test set (about 3000).")
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.1, train_size=0.9)
    y_train_dict = dict(sorted(collections.Counter(y_train).items()))
    plt.bar(range(len(y_train_dict)), list(y_train_dict.values()), align='center')
    plt.xticks(range(len(y_train_dict)), list(y_train_dict.keys()))
    plt.title('label distribution in y_train after train/dev split')
    plt.show()
    print(f"y_train: {y_train_dict}")

    y_dev_dict = dict(sorted(collections.Counter(y_dev).items()))
    plt.bar(range(len(y_dev_dict)), list(y_dev_dict.values()), align='center')
    plt.xticks(range(len(y_dev_dict)), list(y_dev_dict.keys()))
    plt.title('label distribution in y_dev after train/dev split')
    print(f"y_dev:{y_dev_dict}")
    plt.show()
    print()

    assert(len(x_train)==len(y_train))
    train = (x_train,y_train)
    assert (len(x_dev) == len(y_dev))
    dev = (x_dev,y_dev)
    test = (xtest,)

    print(f"# train samples: {len(train[0])}")
    print(f"# dev samples: {len(dev[0])}")
    print(f"# test samples: {len(test[0])}")
    print()


    return train, dev, test


from torch.utils.data import Dataset, DataLoader
class novel_classification(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(x, y, tokenizer, max_len, batch_size):
    ds = novel_classification(
        texts=x,
        targets=y,
        tokenizer=tokenizer,
        max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

from transformers import BertTokenizer, BertModel


from torch import nn
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
class BertNovelClassifier(nn.Module):
    def __init__(self, n_classes=12):
        super(BertNovelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        #self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids,attention_mask=attention_mask)

        output = output[0]

        output = torch.mean(output, dim=1)

        #output = self.drop(output)
        return self.out(output)

from transformers import  AdamW, get_linear_schedule_with_warmup

def deep_learning(train, dev, test):
    print("-"*5)
    MAX_LEN = 512
    x_train, y_train = train[0], train[1]
    x_dev, y_dev = dev[0], dev[1]
    x_test = test[0]
    pred_dev = []
    pred_test = []

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    train_data_loader = create_data_loader(x_train, y_train, tokenizer, MAX_LEN, batch_size)
    val_data_loader = create_data_loader(x_dev,y_dev, tokenizer, MAX_LEN, 1)
    y_test = [-1] * len(x_test)
    test_data_loader = create_data_loader(x_test, y_test, tokenizer, MAX_LEN, 1)

    model = BertNovelClassifier()

    model = model.to(device)

    # define optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

    total_steps = len(train_data_loader) * max_num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # train the model on training data
    best_acc = -1.0

    for epoch in range(1, max_num_epochs + 1):

        start_time = time.time()

        total_acc, total_count = 0, 0

        for idx, d in enumerate(train_data_loader):
            model.train()
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            logits = model(input_ids = input_ids,
                           attention_mask = attention_mask)

            _, preds = torch.max(logits, dim=1)

            loss = criterion(logits, targets).to(device)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad()

            total_acc += torch.sum(preds == targets).item()

            total_count += targets.size(0)

            # if idx % eval_interval == 0:
            #
            #     # evaluate the model on a dev set
            #     model.eval()
            #     dev_loss, dev_acc, dev_count = 0.0, 0.0, 0.0
            #
            #     with torch.no_grad():
            #         for dev_idx, (dev_label, dev_text, dev_lens) in enumerate(dev_dataloader):
            #             dev_logits = model(dev_text, dev_lens)
            #             dev_loss = criterion(dev_logits, dev_label)
            #             dev_acc += (dev_logits.argmax(1) == dev_label).sum().item()
            #             dev_count += label.size(0)
            #     dev_total_acc = (dev_acc * 100 / dev_count)
            #     if dev_total_acc > best_acc:
            #         best_acc = dev_total_acc
            #         torch.save(model.state_dict(), './best_model.pt')


            if idx % log_interval == 0:
                elapsed = time.time() - start_time

                print(f'| epoch {epoch:3d} | '
                      f'{idx:5d}/{len(train_data_loader):2d} batches |'
                      f' loss: {loss.item():3.2f} | '
                      f'train accuracy {total_acc * 100 / total_count:3.2f} | '
                      )
                total_acc, total_count = 0, 0
                start_time = time.time()
    exit(0)
    print("===")
    print(f"best accuracy on the dev set: {best_acc:3.2f}")

    dev_acc = best_acc

    # load best model
    model.load_state_dict(torch.load('./best_model.pt'))
    model.eval()
    # predict on test set.
    pred_test = None

    return dev_acc, pred_test

def main():
    print()
    set_seed(seed=seed)
    train, dev, test = read_data()


    dev_acc, pred_test = deep_learning(train, dev, test)


    print(f"method: {method} => Expected accuracy on the test set = accuracy score on the dev set: {dev_acc:.2f}%")
    print()


if __name__=="__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Running time: {end_time - start_time:5.2f} seconds")