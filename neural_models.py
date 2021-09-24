import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class MLPClassifier(nn.Module):
    def __init__(self,batch_size, vocab_size):
        super(MLPClassifier, self).__init__()


        self.batch_size = batch_size
        self.output_size = 12
        self.hidden_size = 100
        self.vocab_size = vocab_size
        self.embedding_length = 50
        self.LSTM_layers = 1
        self.embeddings_path = "./glove.840B.300d-char.txt"

        print('Processing pretrained character embeds...')
        embedding_vectors = {}
        with open(self.embeddings_path, 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                vec = np.array(line_split[1:], dtype=float)
                char = line_split[0]
                embedding_vectors[char] = vec

        embedding_matrix = np.zeros((self.vocab_size, 300))


        with open("./tokenizer.vocab", "r") as f:
            lines = f.read().strip().split("\n")

        char_indices = {k:v.split()[0] for k,v in enumerate(lines)}

        for char, i in char_indices.items():
            embedding_vector = embedding_vectors.get(char)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        weight = torch.FloatTensor(embedding_matrix)

        #self.word_embeddings = nn.Embedding.from_pretrained(weight, padding_idx=0, freeze=False )
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_length, padding_idx=0)

        self.mlp = [nn.Linear(self.embedding_length, 512).to("cuda")]
        self.mlp.extend([nn.Linear(512, 512).to("cuda") for i in range(2)])

        self.label = nn.Linear(512, self.output_size)

    def forward(self, input_sentence, lens):

        batch_size = input_sentence.size(0)

        x = self.word_embeddings(input_sentence)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)

        sent_vec = torch.mean(x, dim=1)

        for i in range(len(self.mlp)):

            sent_vec = self.mlp[i](sent_vec)

            sent_vec = nn.functional.relu(sent_vec)

        logits = self.label(sent_vec)

        return logits

class CNNClassifier(nn.Module):
    def __init__(self, batch_size =None ,vocab_size =None,  number_of_classes=12):
        super(CNNClassifier, self).__init__()
        self.embed_num =30
        self.embed_dim = 8
        self.kernel_num =100
        self.kernel_sizes = [3,4,5,6,7,8,9,10,11]

        self.number_of_classes = number_of_classes

        V = self.embed_num
        D = self.embed_dim
        C = self.number_of_classes
        Ci = 1
        Co = self.kernel_num
        Ks = self.kernel_sizes

        self.embed = nn.Embedding(V, D, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self,input_sentence, lens):
        x = self.embed(input_sentence)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        #x = self.dropout(x)  # (N, len(Ks)*Co)

        logit = self.fc1(x)  # (N, C)

        return logit

class BiLSTMClassifier(nn.Module):
    def __init__(self,batch_size, vocab_size):
        super(BiLSTMClassifier, self).__init__()


        self.batch_size = batch_size
        self.output_size = 12
        self.hidden_size = 100
        self.vocab_size = vocab_size
        self.embedding_length = 50
        self.LSTM_layers = 1
        self.embeddings_path = "./glove.840B.300d-char.txt"

        print('Processing pretrained character embeds...')
        embedding_vectors = {}
        with open(self.embeddings_path, 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                vec = np.array(line_split[1:], dtype=float)
                char = line_split[0]
                embedding_vectors[char] = vec

        embedding_matrix = np.zeros((self.vocab_size, 300))


        with open("./tokenizer.vocab", "r") as f:
            lines = f.read().strip().split("\n")

        char_indices = {k:v.split()[0] for k,v in enumerate(lines)}

        for char, i in char_indices.items():
            embedding_vector = embedding_vectors.get(char)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        weight = torch.FloatTensor(embedding_matrix)

        #self.word_embeddings = nn.Embedding.from_pretrained(weight, padding_idx=0, freeze=False )
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_length, padding_idx=0)

        self.lstm = nn.LSTM(self.embedding_length,
                            num_layers=self.LSTM_layers,
                            hidden_size=self.hidden_size,
                            bidirectional = True,
                            batch_first=True)

        self.output_layer = nn.Linear(2*self.hidden_size, 512)

        self.label = nn.Linear(512, self.output_size)


    def forward(self, input_sentence, lens):

        batch_size = input_sentence.size(0)

        x = self.word_embeddings(input_sentence)

        x, (final_hidden_state, final_cell_state) = self.lstm(x)

        sent_vec = torch.mean(x, dim=1)

        sent_vec = torch.nn.functional.relu(self.output_layer(sent_vec))

        logits = self.label(sent_vec)

        return logits

class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, batch_size =None ,vocab_size =None,  number_of_classes=12):
        super(CNNBiLSTMClassifier, self).__init__()
        self.embed_num =30
        self.embed_dim = 8
        self.kernel_num =100
        self.kernel_sizes = [3,4,5]

        self.number_of_classes = number_of_classes

        V = self.embed_num
        D = self.embed_dim
        C = self.number_of_classes
        Ci = 1
        Co = self.kernel_num
        Ks = self.kernel_sizes

        self.embed = nn.Embedding(V, D, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.lstm = nn.LSTM(self.kernel_num,
                            num_layers=1,
                            hidden_size=self.kernel_num+50,
                            bidirectional = True,
                            batch_first=True)

        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(Ks) * 2 * (self.kernel_num+50), C)



    def forward(self,input_sentence, lens):
        x = self.embed(input_sentence)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        lstm_outputs = []
        for i in x:
            i = i.permute(0,2,1)
            i_out, _ = self.lstm(i)
            i_out = i_out.mean(dim=1)
            lstm_outputs.append(i_out)

        x = lstm_outputs

        x = torch.cat(x, dim=1)

        logit = self.fc1(x)  # (N, C)

        return logit





