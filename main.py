import time
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from neural_models import MLPClassifier, CNNClassifier, BiLSTMClassifier, CNNBiLSTMClassifier
from non_neural_models import (random_uniform_baseline, majority_baseline, logistic_regression)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0,  type=int
                        , help="random seed to ensure reproducibility of results")
    parser.add_argument("--method", default="random",  type=str
                        , help="random|majority|logreg|mlp|cnn|bilstm|cnnbilstm")
    args = parser.parse_args()
    return args

tokenizer_type="char" # the type of SentencePiece model, including unigram, bpe, char, word.
vocab_size = 30
max_num_epochs = 150 #200
log_interval = 2
eval_interval = 100
learning_rate = 0.2
batch_size = 64

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
    plt.title('label distribution in ytrain_obfuscated.txt')
    plt.savefig('./label-distribution-ytrain-obfuscated.png', bbox_inches='tight')
    print(f"all: {labels}")
    print()

    print("split the train set into two sets: train, and dev with ratio 0.9/0.1 respectively.")
    print("why 0.9/0.1 ratio? because the size of dev set becomes similar to the test set (about 3000).")
    x_train, x_dev, y_train, y_dev = train_test_split(xtrain, ytrain, test_size=0.1, train_size=0.9)
    y_train_dict = dict(sorted(collections.Counter(y_train).items()))
    plt.bar(range(len(y_train_dict)), list(y_train_dict.values()), align='center')
    plt.xticks(range(len(y_train_dict)), list(y_train_dict.keys()))
    plt.title('label distribution in y_train after train/dev split')
    plt.savefig('./label-distribution-ytrain-after-train-dev-split.png', bbox_inches='tight')
    print(f"y_train: {y_train_dict}")

    y_dev_dict = dict(sorted(collections.Counter(y_dev).items()))
    plt.bar(range(len(y_dev_dict)), list(y_dev_dict.values()), align='center')
    plt.xticks(range(len(y_dev_dict)), list(y_dev_dict.keys()))
    plt.title('label distribution in y_dev after train/dev split')
    print(f"y_dev:{y_dev_dict}")
    plt.savefig('./label-distribution-ydev-after-train-dev-split.png', bbox_inches='tight')
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

class LANG():
    def __init__(self, sentences):
        self.idx2chr = {0:'<pad>'}
        self.chr2idx = {'<pad>':0}
        self.len = 1
        for sent in sentences:
            for char in list(sent):
                if char not in self.chr2idx:
                    self.chr2idx[char] = self.len
                    self.idx2chr[self.len] = char
                    self.len += 1

    def sent_to_ids(self,x):
        return [self.chr2idx[chr] for chr in list(x)]

    def ids_to_sent(self,ids):
        return [self.idx2chr[id] for id in ids]

def neural_runner(args, model, train, dev, test):
    print("-"*5)

    x_train, y_train = train[0], train[1]
    x_dev, y_dev = dev[0], dev[1]
    x_test = test[0]
    pred_dev = []
    pred_test = []

    # learn tokenization model
    all_sentences = x_train[:]
    all_sentences.extend(x_dev[:])
    lang = LANG(all_sentences)


    def collate_batch(batch):
        label_list, text_list, lens = [], [], []
        for (_label, _text) in batch:
            label_list.append(_label)
            processed_text = lang.sent_to_ids(_text)
            #processed_text = list(token_id_generator([_text]))[0][1:]
            text_list.append(processed_text)
            lens.append(len(processed_text))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        max_length = max(lens)
        lens = torch.tensor(lens)
        padded_text_list = []
        for item in text_list:
            padded_item = [0] * max_length
            padded_item[-len(item):]= item
            padded_text_list.append(padded_item)
        padded_text_list = torch.tensor(padded_text_list, dtype=torch.int64)
        return label_list.to(device), padded_text_list.to(device), lens.to(device)

    train_dataloader = DataLoader(list(zip(y_train,x_train)), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    dev_dataloader = DataLoader(list(zip(y_dev, x_dev)), batch_size=1, shuffle=False, collate_fn=collate_batch)
    y_test = [-1]*len(x_test)
    test_dataloader = DataLoader(list(zip(y_test, x_test)), batch_size=1, shuffle=False, collate_fn=collate_batch)

    model = model.to(device)

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    print(model.parameters)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # train the model on training data
    best_acc = -1.0

    for epoch in range(1, max_num_epochs + 1):

        start_time = time.time()

        total_acc, total_count = 0, 0

        for idx, (label, text, lens) in enumerate(train_dataloader):

            model.train()

            logits = model(text, lens)

            loss = criterion(logits, label)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            _, preds = torch.max(logits, dim=1)

            total_acc += torch.sum(preds == label).item()

            total_count += label.size(0)

            if idx % eval_interval == 0:

                # evaluate the model on a dev set
                model.eval()

                dev_loss, dev_acc, dev_count = 0.0, 0.0, 0.0

                with torch.no_grad():

                    for dev_idx, (dev_label, dev_text, dev_lens) in enumerate(dev_dataloader):

                        dev_logits = model(dev_text, dev_lens)

                        dev_loss = criterion(dev_logits, dev_label)

                        dev_acc += (dev_logits.argmax(1) == dev_label).sum().item()

                        dev_count += dev_label.size(0)

                dev_total_acc = (dev_acc * 100 / dev_count)

                if dev_total_acc > best_acc:

                    best_acc = dev_total_acc

                    torch.save(model.state_dict(), f'./best-model-{args.method}-seed-{args.seed}.pt')

                    pred_test = []

                    with torch.no_grad():

                        for test_idx, (test_label, test_text, test_lens) in enumerate(test_dataloader):

                            test_logits = model(test_text, test_lens)

                            pred_test.append(test_logits.argmax(1).item())

                    assert(len(pred_test)== len(test_dataloader))

                    with open(f'./ytest-best-model-{args.method}-seed-{args.seed}.txt', "w") as fout:

                        fout.write('\n'.join([str(item) for item in pred_test]))

            if idx % log_interval == 0:
                elapsed = time.time() - start_time

                print(f'| epoch {epoch:3d} | '
                      f'{idx:5d}/{len(train_dataloader):2d} batches |'
                      f' loss: {loss.item():3.2f} | '
                      f' train accuracy {total_acc * 100 / total_count:3.2f} | '
                      f' dev loss: {dev_loss.item():3.2f} | '
                      f' dev accuracy: {dev_total_acc:3.2f} |'
                      f' best dev accuracy: {best_acc:3.2f}'
                      )
                total_acc, total_count = 0, 0
                start_time = time.time()
    print("===")
    print(f"best accuracy on the dev set: {best_acc:3.2f}")

    dev_acc = best_acc

    return dev_acc, pred_test

def main():
    print()
    args = parse_args()
    method = args.method
    set_seed(seed=args.seed)
    train, dev, test = read_data()

    # random baseline
    if method == "random":
        dev_acc, pred_test = random_uniform_baseline(args, train,dev,test)

    elif method == "majority":
        dev_acc, pred_test = majority_baseline(args, train, dev, test)
    elif method == "logreg":
        dev_acc, pred_test = logistic_regression(args, train, dev, test)
    elif method == "mlp":
        model = MLPClassifier(batch_size,vocab_size).to(device)
        dev_acc, pred_test = neural_runner(args, model, train, dev, test)
    elif method == "bilstm":
        model = BiLSTMClassifier(batch_size,vocab_size)
        dev_acc, pred_test = neural_runner(args, model, train, dev, test)
    elif method == "cnn":
        model = CNNClassifier(batch_size,vocab_size)
        dev_acc, pred_test = neural_runner(args, model, train, dev, test)
    elif method == "cnnbilstm":
        model = CNNBiLSTMClassifier(batch_size, vocab_size)
        dev_acc, pred_test = neural_runner(args, model, train, dev, test)
    else:
        raise NotImplementedError(method)

    assert (len(pred_test) == len(test[0]))
    with open(f'./ytest-best-model-{args.method}-seed-{args.seed}.txt', 'w') as fout:

        fout.write('\n'.join([str(item) for item in pred_test]))

    print(f"method: {method} => Expected accuracy on the test set = accuracy score on the dev set: {dev_acc:.2f}%")
    print(f"method: {method} => test predictions are saved in: ./ytest-best-model-{args.method}-seed-{args.seed}.txt")
    print()

if __name__=="__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Running time: {end_time - start_time:5.2f} seconds")