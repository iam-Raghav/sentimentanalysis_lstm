from __future__ import print_function
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2out = nn.Linear(hidden_dim, output_size)



    def forward(self, sentence):
        embeds = self.word_embeddings(sentence) #dimensions length of sentence * embedding dimension
        # sin_embeds = t.sigmoid(embeds)
        lstm_out , _ = self.lstm(embeds.view(1,len(sentence), -1)) #dimensions 1 * length of sentence * embedding dimension
        # h_n = h_n.view(len(sentence),-1)
        embeds = embeds.view(len(sentence),-1)#dimensions length of sentence * embedding dimension
        length = t.tensor(len(sentence)-1, dtype= t.long) #will give length of the sentence
        embeds = embeds.index_select(0,length) #selects only first set of output entries with the dimension of sentence length
        out_space = self.hidden2out(embeds)#dimensions hidden_dim * output size
        out_scores = F.softmax(out_space, dim=1) #return softmax of the output size
        return out_scores

class Brain():
    def __init__(self,embedding_dim, hidden_dim, vocab_size, output_size):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.model = LSTMClassifier(embedding_dim, hidden_dim, vocab_size, output_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.loss = nn.NLLLoss()

    def prepare_sequence(self, seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return idxs

    def classify(self, sentence):
        with t.no_grad():
            output = self.model(sentence)
        return output.max(1)[1].view(-1,1)

    def train(self, sentence, labels):
        sentence = t.tensor(sentence, dtype= t.long)
        labels = t.tensor(labels, dtype= t.long)
        output = self.model(sentence)
        labels = labels.unsqueeze(0)
        loss_out = self.loss(output, t.max(labels,1)[1])
        loss_out.backward()
        self.optimizer.step()
        return loss_out.item()


    def pre_trainer(self, inputs, labels, word_to_ix, epochs):
        labels_class = np.zeros(shape=(len(labels), 2))
        labels = labels.values
        print_losses =[]
        for i in range(len(labels)):
            if labels[i] == 0:
                labels_class[i, 0] = 1
            else:
                labels_class[i, 1] = 1

        for epoch in range(epochs):
            losses = 0
            for i in np.random.permutation(len(inputs)):
                sentence = self.prepare_sequence(inputs[i].split(), word_to_ix)
                self.model.zero_grad()
                loss = self.train(sentence, labels_class[i])
                losses += loss
            print_losses.append(losses/len(inputs))
            if ((epoch + 1) % 50) == 0:
                print(sum(print_losses) / 50)
                print_losses = []






