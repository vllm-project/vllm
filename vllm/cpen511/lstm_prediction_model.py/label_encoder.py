import torch
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as Fun

from config import *


batch_size = config['batch_size']
sequence_length = config['sequence_length']


class PrefetchingDataset(Dataset):
    def __init__(self, pc, delta_in):
        #self.targets = targets
        self.delta_in = label_encoder_deltas.transform(delta_in)
        self.pcs = pc
       
    def __len__(self):
        return (int(batch_size * math.floor(len(self.delta_in)/batch_size)) - batch_size)

    def __getitem__(self, idx):
        #pcs = Fun.one_hot(torch.tensor(self.pcs[idx:idx+sequence_length]), max(num_pc, num_output_next))
        #deltas = Fun.one_hot(torch.tensor(self.delta_in[idx:idx+sequence_length]), max(num_pc, num_output_next))
        #pcs = torch.tensor(self.pcs[idx:idx+sequence_length])
        pcs = self.pcs[idx:idx+sequence_length]
        deltas = torch.tensor(self.delta_in[idx:idx+sequence_length])
        
        targets = Fun.one_hot(torch.tensor(self.delta_in[idx+sequence_length]), config['num_output_next'])
    
        return (pcs, deltas, targets)

def load_data(data, batch_size):
    data = data.copy()
    deltas = data['delta_in']
    # Modify the DataFrame safely
    data.loc[:, 'pc_encoded'] = label_encoder_pc.transform(data['pc'].values)
    pc = torch.tensor(data['pc_encoded'].values)
    dataset = PrefetchingDataset(pc, deltas)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # Get unique target keys
    target_keys = set(deltas.unique())
    return data_loader, len(label_encoder_pc.classes_), len(label_encoder_deltas.classes_), len(label_encoder_deltas.classes_), target_keys

def encode_data(train_data, test_data):
    label_encoder_pc.fit(list(set().union(train_data['pc'].values, test_data['pc'].values)))
    label_encoder_deltas.fit(list(set().union(train_data['delta_in'].values, test_data['delta_in'].values)))
    train_iter, num_pc, num_delta_in, num_output_next, target_keys = load_data(train_data, batch_size=batch_size)
    test_iter, _, _, _, _ = load_data(test_data, batch_size=batch_size)
    config['num_pc'] = num_pc
    config['num_delta_in'] = num_delta_in
    config['num_output_next'] = num_output_next
    
    print('number of unique pc: ', num_pc)
    print('number of unique input delta: ', num_delta_in)
    print('number of unique output delta: ', num_output_next)
    return train_iter, test_iter, target_keys