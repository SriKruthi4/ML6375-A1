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
import string
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h,dropout=0.3):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.dropout=nn.Dropout(dropout)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        op, hidden = self.rnn(inputs)
       
        op = self.dropout(op)
        
        ol = self.W(op[-1])
        sumop=ol.sum(dim=0)
        
        predicted_vector = self.softmax(ol)
        return predicted_vector

        return predicted_vector


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

    print("========== Loading data ==========")
    train_data, valid_data,test_data = load_data(args.train_data, args.val_data,args.test_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0005,weight_decay=1e-4)
    word_embedding = pickle.load(open('./Data_Embedding/word_embedding.pkl', 'rb'))

    
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    best_val_acc = 0.0
    best_train_acc = 0.0
    cycles=3
    epochs_no_improve=0
    stop = False
    stopping_condition = False

    train_accuracies=[]
    val_accuracies=[]
    train_losses=[]
    val_losses=[]

    while not stopping_condition and epoch<args.epochs:
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 32
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]

                # Transform the input into required shape
                #vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                vectors = torch.tensor(np.array(vectors)).view(len(vectors), 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        train_acc=correct/total
        train_accuracies.append(train_acc)
        train_losses.append(loss_total / (len(train_data) // minibatch_size))
        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        trainning_accuracy = correct/total


        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            # print(predicted_label, gold_label)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        val_acc= correct/total
        val_accuracies.append(val_acc)
        val_losses.append(loss_total / (len(valid_data) // minibatch_size))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cycles:
                print(f"Training stopped due to no improvement in validation accuracy for {cycles} consecutive epochs.")
                break

        epoch += 1

        model.eval()
    correct = 0
    total = 0
    print("Testing started for epoch {}".format(epoch + 1))
    with torch.no_grad():
        for input_words, gold_label in tqdm(test_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
            #vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            vectors = torch.tensor(np.array(vectors)).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
    print("Testing completed ")
    print("Testing accuracy: {:.2%}".format(correct / total))
    testing_accuracy = correct / total

    learning_rate=0.0005

    dropout=0.3
    output_file_path = 'results/test2.out'

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'a') as f:
            f.write("="*65 + "\n")
            f.write(f"Epochs: {args.epochs}\t")
            f.write(f"Learning Rate: {learning_rate}\t")
            f.write(f"Dropout: {dropout}\t")
            f.write(f"Hidden Dim: {args.hidden_dim}\t")
            f.write(f"Batch size: {minibatch_size}\t")
            f.write("\n")
            f.write("="*65 + "\n")
            f.write(f"Testing Accuracy: {testing_accuracy:.4f}\n")
            f.write(f"{'Epoch':<10}{'Training Accuracy':<20}{'Validation Accuracy':<20}\n")
            f.write("="*65 + "\n")
            for epoch in range(len(train_accuracies)):
                train_acc = train_accuracies[epoch-1]
                val_acc = val_accuracies[epoch-1]
                f.write(f"{epoch:<10}{train_acc:<20.4f}{val_acc:<20.4f}\n")

    fig, ax1 = plt.subplots(figsize=(10, 6))

   
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='blue')
    ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue', marker='o')
    ax1.tick_params(axis='y', labelcolor='blue')

   
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='orange')
    ax2.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validation Accuracy', color='orange', marker='o')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(0.1,1.0) 

   
    plt.title('Training Loss and Validation Accuracy vs. Epochs')
    fig.tight_layout() 
    plt.grid()
    plt.show()

    # print(train_losses)
    # print(val_losses)


    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
