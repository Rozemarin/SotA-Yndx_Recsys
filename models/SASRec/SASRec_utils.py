import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.SASRec.SASRec_model import SASRec


def fix_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def sequential_batch_sampler(user_train, usernum, itemnum, batch_size, maxlen, seed, pad_token=None):
    if pad_token is None:
        pad_token = itemnum
    
    def sample(random_state):
        user = random_state.randint(usernum)
        while len(user_train.get(user, [])) <= 1:
            user = random_state.randint(usernum)
        user_items = user_train[user]
        seq = np.full(maxlen, pad_token, dtype=np.int32)
        pos = np.full(maxlen, pad_token, dtype=np.int32)
        neg = np.full(maxlen, pad_token, dtype=np.int32)
        nxt = user_items[-1]
        idx = maxlen - 1
        ts = set(user_items)
        for i in reversed(user_items[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(0, itemnum, ts, random_state)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        return (user, seq, pos, neg)

    random_state = np.random.RandomState(seed)
    while True:
        yield zip(*(sample(random_state) for _ in range(batch_size)))


def random_neq(l, r, s, random_state):
    t = random_state.randint(l, r)
    while t in s:
        t = random_state.randint(l, r)
    return t


def batch_generator(user_train, batch_size, maxlen, pad_token=None, seed=None):
    '''
    user_train - sequences
    '''
    sequences = user_train
    while True:
        batch = []
        for user, user_items in sequences.items():
            seq = np.full(maxlen, pad_token, dtype=np.int32)
            pos = np.full(maxlen, pad_token, dtype=np.int32)
            nxt = user_items[-1] # last watched film
            idx = maxlen - 1
            for i in reversed(user_items[:-1]):
                seq[idx] = i
                pos[idx] = nxt
                nxt = i
                idx -= 1
                if idx == -1:
                    break
            batch.append((user, seq, pos))
            if len(batch) == batch_size:
                yield zip(*(batch))
                batch = []
        if seed is None:
            yield zip(*(batch)) # return the rest of the data for test
            break


def data_to_sequences(data, data_description):
    userid = data_description['users']
    itemid = data_description['items']
    sequences = (
        data.sort_values([userid, data_description['order']])
        .groupby(userid, sort=False)[itemid].apply(list)
    )
    return sequences


def prepare_sasrec_model(config, data, data_description):
    n_users = data_description['n_users']
    n_items = data_description['n_items']
    model = SASRec(n_items, config)
    criterion = torch.nn.BCEWithLogitsLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    train_sequences = data_to_sequences(data, data_description)
    sampler = sequential_batch_sampler(
        train_sequences, n_users, n_items,
        batch_size = config['batch_size'],
        maxlen = config['maxlen'],
        seed = config['sampler_seed'],
        pad_token = model.pad_token
    )
    n_batches = len(train_sequences) // config['batch_size']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = config['learning_rate'],
        betas = (0.9, 0.98)
    )
    return model, sampler, n_batches, criterion, optimizer


def train_sasrec_epoch(model, num_batch, l2_emb, sampler, optimizer, criterion, device):
    model.train()
    pad_token = model.pad_token
    losses = []
    for _ in range(num_batch):
        _, *seq_data = next(sampler)
        # convert batch data into torch tensors
        seq, pos, neg = (torch.LongTensor(np.array(x)).to(device) for x in seq_data)
        pos_logits, neg_logits = model(seq, pos, neg)
        pos_labels = torch.ones(pos_logits.shape, device=device)
        neg_labels = torch.zeros(neg_logits.shape, device=device)
        optimizer.zero_grad()
        indices = torch.where(pos != pad_token)
        loss = criterion(pos_logits[indices], pos_labels[indices])
        loss += criterion(neg_logits[indices], neg_labels[indices])
        if l2_emb != 0:
            for param in model.item_emb.parameters():
                loss += l2_emb * torch.norm(param)**2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def build_sasrec_model(config, data, data_description):
    '''Simple MF training routine without early stopping'''
    model, sampler, n_batches, criterion, optimizer = prepare_sasrec_model(config, data, data_description)
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    losses = {}
    for epoch in tqdm(range(config['num_epochs'])):
        losses[epoch] = train_sasrec_epoch(
            model, n_batches, config['l2_emb'], sampler, optimizer, criterion, device
        )
    return model, losses


############# SCORING #############


def sasrec_model_scoring(params, data, data_description):
    model = params
    model.eval()
    tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    test_sequences = data_to_sequences(data, data_description)
    # perform scoring on a user-batch level
    scores = []
    for _, seq in test_sequences.items():
        with torch.no_grad():
            predictions = model.score(tensor(seq))
        scores.append(predictions.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)
