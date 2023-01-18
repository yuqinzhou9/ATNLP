import os
import pickle
import random
import time
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
from torch import optim

plt.switch_backend('agg')
import numpy as np
from tqdm import tqdm

from models import EncoderGRU, AttnDecoderGRU, EncoderLSTM, DecoderLSTM
from utils import Lang, tensorsFromPair, timeSince, showPlot

# % matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = 'SCAN-master'
# dataset_path = '/Users/liuzhaoqi/Liuzhaoqi/UCPH/ATNLP/ATNLP_5/dataset'
task_name = 'length'  # or 'length'

train_file_name = '{}_split/tasks_{}_{}.txt'.format(task_name, 'train', task_name)
test_file_name = '{}_split/tasks_{}_{}.txt'.format(task_name, 'test', task_name)
train_file_path = os.path.join(dataset_path, train_file_name)
test_file_path = os.path.join(dataset_path, test_file_name)
# train_file_path, test_file_path

SOS_token = 0
EOS_token = 1

command_le = Lang('command')
action_le = Lang('action')


def dataloader(path):
    with open(path, 'r') as f:
        dataset = f.readlines()

    def preprocess_data(line):
        line = line.strip().split()
        split_index = line.index('OUT:')
        inp = line[1: split_index]
        outp = line[split_index + 1:]
        command_le.addSentence(inp)
        action_le.addSentence(outp)
        return [inp, outp]

    pairs = list(map(preprocess_data, dataset))
    input_commands, output_actions = np.transpose(pairs).tolist()
    return input_commands, output_actions, pairs


commands_train, actions_train, pairs_train = dataloader(train_file_path)
commands_test, actions_test, pairs_test = dataloader(test_file_path)

MAX_LENGTH = max([len(action) for action in actions_test]) + 1

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion,
          model='gru'):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_hiddens = torch.zeros(input_length, encoder.hidden_size, device=device)

    loss = 0
    gold_pred = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)

        if model == 'gru':
            encoder_hiddens[ei] += encoder_hidden[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    preds = []

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            if model == 'gru':
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_hiddens)
            elif model == 'lstm':
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            pred = topi.squeeze()
            preds.append(topi.squeeze().item())

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]


    else:
        for di in range(target_length):
            if model == 'gru':
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_hiddens)
            elif model == 'lstm':
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            pred = topi.squeeze()
            preds.append(topi.squeeze().item())
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                target_length = di + 1
                break

    correct = torch.equal(torch.Tensor(preds).to(device), target_tensor.squeeze())

    loss.backward()

    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, correct


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100,
               learning_rate=0.001, model='gru'):
    start = time.time()

    accuracy = 0
    plot_losses = []
    plot_accs = []
    print_loss_total = 0
    plot_loss_total = 0
    print_pred_total = 0
    plot_pred_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs_train), command_le, action_le)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss, correct = train(input_tensor, target_tensor, encoder, decoder,
                              encoder_optimizer, decoder_optimizer, criterion,
                              model)
        print_pred_total += int(correct)
        plot_pred_total += int(correct)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_acc_avg = print_pred_total / print_every
            accuracy = print_acc_avg
            print_loss_avg = print_loss_total / print_every
            print_pred_total = 0
            print_loss_total = 0
            print('%s (%d %d%%) loss: %.4f acc: %.4f' % (timeSince(start, iter / n_iters),
                                                         iter, iter / n_iters * 100, print_loss_avg, print_acc_avg))

        if iter % plot_every == 0:
            plot_acc_avg = plot_pred_total / plot_every
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_accs.append(plot_acc_avg)
            plot_loss_total = 0
            plot_pred_total = 0

    showPlot(plot_losses, plot_accs)
    return accuracy


def evaluate(encoder, decoder, pair, model='gru', max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor, target_tensor = tensorsFromPair(pair, command_le, action_le)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_hiddens = torch.zeros(input_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

            if model == 'gru':
                encoder_hiddens[ei] += encoder_hidden[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, input_length)

        for di in range(max_length):
            if model == 'gru':
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_hiddens)
                decoder_attentions[di] = decoder_attention.squeeze().data
            elif model == 'lstm':
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(action_le.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_model(encoder, decoder, test_pairs, model='gru'):
    command_cnt = Counter([len(test_pair[0]) for test_pair in test_pairs])
    action_cnt = Counter([len(test_pair[1]) for test_pair in test_pairs])
    command_correct_cnt = defaultdict(int)
    action_correct_cnt = defaultdict(int)
    correct = 0

    for pair in tqdm(test_pairs):
        preds, attentions = evaluate(encoder, decoder, pair, model=model)
        preds = preds[:-1]
        target_output = pair[1]
        if preds == target_output:
            command_correct_cnt[len(pair[0])] += 1
            action_correct_cnt[len(pair[1])] += 1
            correct += 1

    command_correct_cnt = dict(command_correct_cnt)
    action_correct_cnt = dict(action_correct_cnt)
    command_cnt = dict(command_cnt)
    action_cnt = dict(action_cnt)

    command_acc = {}
    for command_length, cnt in command_cnt.items():
        command_acc[command_length] = command_correct_cnt.get(
            command_length, 0) / cnt

    action_acc = {}
    for action_length, cnt in action_cnt.items():
        action_acc[action_length] = action_correct_cnt.get(
            action_length, 0) / cnt

    return command_acc, action_acc, correct / len(test_pairs)


def evaluate_fixed_length(encoder, decoder, pair, model='gru', max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor, target_tensor = tensorsFromPair(pair, command_le, action_le)
        input_length = input_tensor.size()[0]
        output_length = target_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_hiddens = torch.zeros(input_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

            if model == 'gru':
                encoder_hiddens[ei] += encoder_hidden[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(output_length, input_length)

        for di in range(output_length):
            if model == 'gru':
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_hiddens)
                decoder_attentions[di] = decoder_attention.squeeze().data
            elif model == 'lstm':
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(2)
            if topi[0][0].item() == EOS_token and di != output_length - 1:
                topi = topi[0][1]
            else:
                topi = topi[0][0]

            decoded_words.append(action_le.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions


def evaluate_model_fix(encoder, decoder, test_pairs, model='gru'):
    command_cnt = Counter([len(test_pair[0]) for test_pair in test_pairs])
    action_cnt = Counter([len(test_pair[1]) for test_pair in test_pairs])
    command_correct_cnt = defaultdict(int)
    action_correct_cnt = defaultdict(int)
    correct = 0

    for pair in tqdm(test_pairs):
        preds, attentions = evaluate_fixed_length(encoder, decoder, pair, model=model)
        preds = preds[:-1]
        target_output = pair[1]
        if preds == target_output:
            command_correct_cnt[len(pair[0])] += 1
            action_correct_cnt[len(pair[1])] += 1
            correct += 1

    command_correct_cnt = dict(command_correct_cnt)
    action_correct_cnt = dict(action_correct_cnt)
    command_cnt = dict(command_cnt)
    action_cnt = dict(action_cnt)

    command_acc = {}
    for command_length, cnt in command_cnt.items():
        command_acc[command_length] = command_correct_cnt.get(
            command_length, 0) / cnt

    action_acc = {}
    for action_length, cnt in action_cnt.items():
        action_acc[action_length] = action_correct_cnt.get(
            action_length, 0) / cnt

    return command_acc, action_acc, correct / len(test_pairs)


def evaluateRandomly(encoder, decoder, model='gru', n=10):
    for i in range(n):
        pair = random.choice(pairs_test)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair, model=model)
        output_sentence = output_words
        print('<', output_sentence)
        print('')


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(encoder, decoder, pair, model='gru'):
    output_words, attentions = evaluate(
        encoder, decoder, pair, model=model)
    print('input =', pair[0])
    print('output =', output_words)
    showAttention(pair[0], output_words, attentions)


def evaluateAndShowAttentionExample(encoder, decoder, model='gru'):
    for i in range(len(pairs_test)):
        preds, attentions = evaluate(encoder, decoder, pairs_test[i], model=model)
        preds = preds[:-1]
        target_output = pairs_test[i][1]
        if preds == target_output:
            evaluateAndShowAttention(encoder, decoder, pairs_test[i])
            break


def main_run(hidden_size, num_iter, num_runs, model):
    input_size = command_le.n_words
    output_size = action_le.n_words

    best_encoder = None
    best_decoder = None
    best_acc = 0

    command_accs, action_accs, overall_accs, train_accs = [], [], [], []

    for i in range(num_runs):
        if model == 'gru':
            encoder = EncoderGRU(input_size, hidden_size, num_layers=1, dropout=0.5).to(device)
            decoder = AttnDecoderGRU(hidden_size, output_size, dropout=0.5).to(device)

        elif model == 'lstm':
            encoder = EncoderLSTM(input_size, hidden_size, num_layers=2, dropout=0.5).to(device)
            decoder = DecoderLSTM(hidden_size, output_size, num_layers=2, dropout=0.5).to(device)

        accuracy_train = trainIters(encoder, decoder, num_iter, print_every=1000, model=model)
        acc_command, acc_action, acc_overall = evaluate_model(encoder, decoder, pairs_test, model=model)

        if acc_overall > best_acc:
            best_encoder = encoder
            best_decoder = decoder
            best_acc = acc_overall

        command_accs.append(acc_command)
        action_accs.append(acc_action)
        overall_accs.append(acc_overall)
        train_accs.append(accuracy_train)

    with open('models_{}.pickle'.format(model), 'wb') as f:
        pickle.dump([best_encoder, best_decoder, best_acc], f)

    with open('train_results_{}.pickle'.format(model), 'wb') as f:
        pickle.dump([command_accs, action_accs, overall_accs, train_accs], f)

    return command_accs, action_accs, overall_accs, train_accs


def calculate_mean_std(acc_dict):
    mean = []
    error = []
    keys = sorted(acc_dict[0])
    num_runs = len(acc_dict)

    for key in keys:
        t = []
        for d in acc_dict:
            t.append(d[key])
        mean.append(np.mean(t))
        error.append(np.std(t) / np.sqrt(num_runs))
    return mean, error, keys


if __name__ == '__main__':

    # top-performing model: GRU with Attention
    main_run(hidden_size=50, num_iter=100000, num_runs=5, model='gru')

    # overall best model: LSTM
    main_run(hidden_size=200, num_iter=100000, num_runs=5, model='lstm')

    # Load saved models and results
    with open('models_gru.pickle', 'rb') as f:
        [best_encoder_gru, best_decoder_gru, best_acc_gru] = pickle.load(f)
    with open('models_lstm.pickle', 'rb') as f:
        [best_encoder_lstm, best_decoder_lstm, best_acc_lstm] = pickle.load(f)
    with open('train_results_gru.pickle', 'rb') as f:
        [command_accs_gru, action_accs_gru, overall_accs_gru, train_accs_gru] = pickle.load(f)
    with open('train_results_lstm.pickle', 'rb') as f:
        [command_accs_lstm, action_accs_lstm, overall_accs_lstm, train_accs_lstm] = pickle.load(f)

    best_encoder_lstm.to(device)
    best_decoder_lstm.to(device)
    best_encoder_gru.to(device)
    best_decoder_gru.to(device)

    command_mean_gru, command_error_gru, command_keys = calculate_mean_std(command_accs_gru)
    action_mean_gru, action_error_gru, action_keys = calculate_mean_std(action_accs_gru)

    # Plot the results for commands
    plt.rcParams['figure.figsize'] = (8, 6.4)
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(command_mean_gru)), command_mean_gru, yerr=command_error_gru,
           align='center', alpha=0.5, ecolor='black', color='darkgreen', capsize=2)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, symbol=None))
    ax.set_ylabel('Accuracy on new commands (%)', fontsize=20)
    ax.set_xlabel('Command length', fontsize=20)
    ax.set_xticks(np.arange(len(command_keys)))
    ax.set_xticklabels(command_keys)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    plt.show()

    # Plot the results for actions
    plt.rcParams['figure.figsize'] = (8, 6.4)
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(action_mean_gru)), action_mean_gru, yerr=action_error_gru,
           align='center', alpha=0.5, ecolor='black', color='darkgreen', capsize=2)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, symbol=None))
    ax.set_ylabel('Accuracy on new commands (%)', fontsize=20)
    ax.set_xlabel('Ground-truth action sequence length', fontsize=20)
    ax.set_xticks(np.arange(len(action_keys)))
    ax.set_xticklabels(action_keys)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    plt.show()

    # Visualize attention of correct example
    evaluateAndShowAttentionExample(best_encoder_gru, best_decoder_gru, model='gru')

    # Print overall accuracy
    overall_acc_lstm = np.mean(overall_accs_lstm)
    print("Test accuracy of LSTM:", overall_acc_lstm)
    train_acc_lstm = np.mean(train_accs_lstm)
    print("Train accuracy of LSTM:", train_acc_lstm)
    overall_acc_gru = np.mean(overall_accs_gru)
    print("Test accuracy of GRU:", overall_acc_gru)
    train_acc_gru = np.mean(train_accs_gru)
    print("Train accuracy of GRU:", train_acc_gru)

    # Evaluate models with predictions of fix length
    acc_command_gru_fix, acc_action_gru_fix, acc_overall_gru_fix = evaluate_model_fix(
        best_encoder_gru, best_decoder_gru, pairs_test, model='gru')
    print('Overall test accuracy of GRU with fixed length:', acc_overall_gru_fix)
    acc_command_lstm_fix, acc_action_lstm_fix, acc_overall_lstm_fix = evaluate_model_fix(
        best_encoder_lstm, best_decoder_lstm, pairs_test, model='lstm')
    print('Overall test accuracy of LSTM with fixed length:', acc_overall_lstm_fix)
