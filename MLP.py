import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random, time
import numpy as np
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ds', type=str, default='youtube',
        choices=['youtube', 'imdb', 'yelp', 'agnews', 'spouse', 'fbnews'],
        help='the name of dataset.'
    )
    parser.add_argument(
        '--denoise_label', type=str, default=None,
        help='the path of the denoised label that the model is trained on.'
    )
    parser.add_argument('--batch', type=int, default=0, help='batch size.')
    parser.add_argument('--hidden', type=int, default=256, help='the number of hidden-layer dimensions in MLP.')
    parser.add_argument('--lr', type=float, default=None, help='learning rate of MLP.')
    parser.add_argument('--epoch', type=int, default=50, help='the number of epoches.')
    parser.add_argument('--seed', type=int, default=0, help='random seed.')

    args = parser.parse_args()
    args.num_class = 2
    if args.ds in ['agnews', 'fbnews']:
        args.num_class = 4
    
    print(args)
    return args

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def merge_train_set(data_dict):
    train_data_dict = {}
    train_data_dict['bert_feature'] = torch.cat([data_dict['labeled']['bert_feature'], data_dict['unlabeled']['bert_feature']], 0)
    train_data_dict['major_label'] = torch.cat((data_dict['labeled']['major_label'], data_dict['unlabeled']['major_label']), 0)
    train_data_dict['lf'] = torch.cat((data_dict['labeled']['lf'], data_dict['unlabeled']['lf']), 0)
    train_data_dict['label'] = torch.cat((data_dict['labeled']['label'], data_dict['unlabeled']['label']), 0)
    return train_data_dict

def label_denoising(attn_score, weak_labels, n_class):
    score_matrix = torch.empty(len(weak_labels), n_class)
    for k in range(n_class):
        score_matrix[:, k] = (attn_score * (weak_labels == k).float()).sum(dim=1)
    denoised_labels = torch.argmax(score_matrix, 1)
    return denoised_labels

class ClassificationDataset(Dataset):
    def __init__(self, data, no_abstain=True):
        self.features = [example for example in data['bert_feature'].tolist()]
        self.true_labels = [example for example in data['label'].float().tolist()]
        if 'denoise_label' in data.keys():
            self.training_labels = [example for example in data['denoise_label'].tolist()]
        else:
            self.training_labels = [example for example in data['major_label'].tolist()]
            if no_abstain == True:
                no_abstain_idx = np.where(np.array(self.training_labels) != -1)[0].tolist()
                self.features = [self.features[i] for i in no_abstain_idx]
                self.true_labels = [self.true_labels[i] for i in no_abstain_idx]
                self.training_labels = [self.training_labels[i] for i in no_abstain_idx]

    def __getitem__(self, index):
        return [self.features[index], self.training_labels[index], self.true_labels[index]]

    def __len__(self):
        return len(self.features)

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): the size of the input vectors
            hidden_dim (int): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
        """
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the MLP
        Args:
            x_in (torch.Tensor): an input data tensor 
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        intermediate = F.relu(self.fc1(x_in))
        output = self.fc2(intermediate)
        
        if apply_softmax:
            output = F.softmax(output, dim=1)
        return output

def count_parameters(model):
    """count total parameters in a model. .numel() method is to get the number of parameters in a tensor. .requires_grad means that those tensors that need to compute gradients contains parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, dataloader, optimizer, device):
    for features, training_labels, true_labels in dataloader:
        features, training_labels, true_labels = torch.stack(features, 1).float().to(device), training_labels.float().to(device), true_labels.float().to(device)
        output = model(features)
        loss = F.cross_entropy(output, training_labels.type(torch.LongTensor))
        model.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader, device):
    count = correct = 0.0
    with torch.no_grad():
        for features, _, true_labels in dataloader:
            features, true_labels = torch.stack(features, 1).float().to(device), true_labels.float().to(device)
            # shape: (batch_size, n_labels)
            output = model(features)
            # shape: (batch_size,)
            predicted = output.argmax(dim=-1) # get the label with the highest probability
            count += len(predicted)
            correct += (predicted == true_labels).sum().item()
    accuracy = correct / count
    return accuracy

def main():
    args = parse_args()

    SEED = args.seed
    DATASET = args.ds
    DENOISE_LABEL = args.denoise_label
    BATCH_SIZE = args.batch
    HIDDEN_DIM = args.hidden
    M_CLASSES = args.num_class
    LEARNING_RATE = args.lr
    T_EPOCHES = args.epoch
    
    seed_everything(SEED)
    data = torch.load('./Data/' + DATASET + '/' + DATASET + '_organized_nb.pt')
    train_data = merge_train_set(data)
    if DENOISE_LABEL is not None:
        attn_score = torch.load(DENOISE_LABEL)['fix_score']
        denoised_labels = label_denoising(attn_score, train_data['lf'], M_CLASSES)
        train_data['denoise_label'] = denoised_labels
    train_set = ClassificationDataset(train_data)
    dev_set = ClassificationDataset(data['validation'])
    test_set = ClassificationDataset(data['test'])
    if BATCH_SIZE == 0:
        train_dataloader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
        dev_dataloader = DataLoader(dev_set, batch_size=len(dev_set), shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=len(test_set), shuffle=True)
    else:
        train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        dev_dataloader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    model = MultilayerPerceptron(input_dim=768, hidden_dim=HIDDEN_DIM, output_dim=M_CLASSES)
    print(f"Model has {count_parameters(model)} parameters.")

    # Adam is just a fancier version of SGD.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dev_accuracy_cache = [] # create a list to store minitoring metric
    best_dev_accuracy = 0
    best_model_param = None
    best_epoch = 0
    # randome baseline
    baseline_dev_accuracy = evaluate(model, dev_dataloader, device)
    dev_accuracy_cache.append(baseline_dev_accuracy)

    start_time = time.time() # start time
    for epoch in tqdm(range(T_EPOCHES)):
        train(model, train_dataloader, optimizer, device)
        dev_accuracy = evaluate(model, dev_dataloader, device)
        dev_accuracy_cache.append(dev_accuracy)
        if dev_accuracy >= best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_model_param = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
    print(f"------------ Training time: {time.time() - start_time} seconds ------------")
    print(f'[validation] best accuracy: {best_dev_accuracy} from epoch {best_epoch}')
    model.load_state_dict(best_model_param)
    test_accuracy = evaluate(model, test_dataloader, device)
    print(f'[test] accuracy: {test_accuracy}')

if __name__ == '__main__':
    main()
