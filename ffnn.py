import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.LogSoftmax(dim=-1) # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class
        

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hl=self.activation(self.W1(input_vector))
        hl = self.dropout(hl)
        # [to fill] obtain output layer representation
        op=self.W2(hl)
        # [to fill] obtain probability dist.
        predicted_vector=self.softmax(op)
        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data, test_data=None):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    tst = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))

    if test_data:
        with open(test_data) as test_f:
            testing = json.load(test_f)
        for elt in testing:
            tst.append((elt["text"].split(), int(elt["stars"] - 1)))

    if test_data:
        return tra, val, tst
    else:
        return tra, val



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data,test_data = load_data(args.train_data, args.val_data,args.test_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    test_data = convert_to_vector_representation(test_data, word2index)

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9)
    train_accuracies = []
    val_accuracies = []
    test_accuracies=[]
    train_losses=[]
    val_losses=[]
    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_accuracy= correct/total
        train_accuracies.append(train_accuracy)
        avg_epoch_loss = epoch_loss / (N // minibatch_size)
        train_losses.append(avg_epoch_loss)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))


        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        model.eval()
        avg_epoch_loss_val=0.0
        epoch_loss_val=0.0
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16 
        N = len(valid_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            with torch.no_grad():
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
                epoch_loss_val+=loss.item()

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        avg_epoch_loss_val =loss/minibatch_size
        val_losses.append(avg_epoch_loss)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

    model.eval()
    correct = 0
    total = 0
    loss = None
    start_time = time.time()
    #print("========== Testing started ==========")
    minibatch_size = 16
    N = len(test_data)

    for minibatch_index in tqdm(range(N // minibatch_size)):
        with torch.no_grad():
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = test_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size

  
    

    print("========== Testing completed ==========")
    print("Test accuracy: {}".format(correct / total))
    print("Testing time: {}".format(time.time() - start_time))
    test_accuracy = correct / total

    learning_rate=0.001

    dropout=0.25
    output_file_path = 'results/test1.out'

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'a') as f:
            f.write("="*65 + "\n")
            f.write(f"Epochs: {args.epochs}\t")
            f.write(f"Learning Rate: {learning_rate}\t")
            f.write(f"Dropout: {dropout}\t")
            f.write(f"Hidden Dim: {args.hidden_dim}\t")
            f.write("\n")
            f.write("="*65 + "\n")

            f.write(f"Testing Accuracy: {test_accuracy:.4f}\n")
            f.write(f"{'Epoch':<10}{'Training Accuracy':<20}{'Validation Accuracy':<20}\n")
            f.write("="*65 + "\n")
            for epoch in range(1, args.epochs + 1):
                train_acc = train_accuracies[epoch - 1]
                val_acc = val_accuracies[epoch - 1]
                f.write(f"{epoch:<10}{train_acc:<20.4f}{val_acc:<20.4f}\n")
  
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting Training Loss on the primary y-axis
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='blue')
    ax1.plot(range(1, args.epochs + 1), train_losses, label='Training Loss', color='blue', marker='o')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for Validation Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='orange')
    ax2.plot(range(1, args.epochs+1), val_accuracies, label='Validation Accuracy', color='orange', marker='o')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(0.1,1.0) 

    # Title and grid
    plt.title('Training Loss and Validation Accuracy vs. Epochs')
    fig.tight_layout()  # Adjust layout to accommodate two y-axes
    plt.grid()
    plt.show()

    # print(train_losses)
    # print(val_losses)


   
