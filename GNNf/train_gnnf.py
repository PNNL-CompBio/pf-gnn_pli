import pickle
from gnnf import gnnf
import time
import numpy as np
import utils
import torch.nn as nn
import torch
import time
import os
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import argparse
import time
from torch.utils.data import DataLoader
from gnnf_dataset import MolDataset, collate_fn, DTISampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import csv
import sys

now = time.localtime()
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print(s)
parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default=10000)
parser.add_argument("--ngpu", help="number of gpu", type=int, default=1)
parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
parser.add_argument("--num_workers", help="number of workers", type=int, default=7)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default=4)
parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default=140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default=4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default=128)
parser.add_argument("--dude_data_fpath", help="file path of dude data", type=str, default='data_8/')
parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default='./save/')
parser.add_argument("--initial_mu", help="initial value of mu", type=float, default=4.0)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default=1.0)
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default=0.0)
parser.add_argument("--train_keys", help="train keys", type=str, default='keys/pdbbind_dude_20000_train.pkl')
parser.add_argument("--test_keys", help="test keys", type=str, default='keys/pdbbind_dude_20000_valid.pkl')
#parser.add_argument("--valid_keys", help="valid keys", type=str, default='keys/pdbbind_dude_20000_test.pkl')
args = parser.parse_args()
print(args)

# hyper parameters
num_epochs = args.epoch
lr = args.lr
ngpu = args.ngpu
batch_size = args.batch_size
dude_data_fpath = args.dude_data_fpath
save_dir = args.save_dir

# make save dir if it doesn't exist
if not os.path.isdir(save_dir):
    os.system('mkdir ' + save_dir)

# read data. data is stored in format of dictionary. Each key has information about protein-ligand complex.
with open(args.train_keys, 'rb') as fp:
    train_keys = pickle.load(fp)
with open(args.test_keys, 'rb') as fp:
    test_keys = pickle.load(fp)
#with open(args.valid_keys, 'rb') as fp:
#    valid_keys = pickle.load(fp)

# print simple statistics about dude data and pdbbind data
print(f'Number of train data: {len(train_keys)}')
print(f'Number of test data: {len(test_keys)}')
#print(f'Number of valid data: {len(valid_keys)}')
#train_keys.extend(valid_keys)
#print (f'Number of train data after combining valid keys: {len(train_keys)}')

# split train_dataset into ngpu and make list (len=ngpu) of DataLoaders
# shuffle train keys, split into ngpu
random.seed(2020)
random.shuffle(train_keys)
train_key_size = int(len(train_keys) / args.ngpu)
train_keys = [train_keys[x:x+train_key_size] for x in range(0, len(train_keys), train_key_size)]
train_keys = [x for x in train_keys if len(x) > 100]
print(f'Number of training sets generated: {len(train_keys)}')
print(f'Size of training sets: {train_key_size}')
train_dataset = [MolDataset(x, args.dude_data_fpath) for x in train_keys]
num_train_pos = [len([0 for k in x if 'CHEMBL' in k or 'pos' in k]) for x in train_keys]
num_train_neg = [len([0 for k in x if 'ZINC' in k or 'neg' in k]) for x in train_keys]
train_weights = [[1 / num_train_pos[i] if 'CHEMBL' in k or 'pos' in k else 1 / num_train_neg[i] for k in x] for i, x in enumerate(train_keys)]
train_sampler = [DTISampler(x, len(x), replacement=True) for x in train_weights]
train_dataloader = [DataLoader(x, args.batch_size,
     shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn,
     sampler = train_sampler[i]) for i, x in enumerate(train_dataset)]

# keep single test set
test_dataset = MolDataset(test_keys, args.dude_data_fpath)
test_dataloader = DataLoader(test_dataset, args.batch_size,
     shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn)


def setup(rank, world_size):
    if sys.platform == 'win32':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method="file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_DDP(rank, world_size,):
    print(f"Running DDP process on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = gnnf(args).to(device)
    print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = DDP(model, device_ids=[rank])

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #loss function
    loss_fn = nn.BCELoss()
    for epoch in range(args.epoch):
        st = time.time()
        #collect losses of each iteration
        train_losses = []
        test_losses = []

        #collect true label of each iteration
        train_true = []
        test_true = []

        #collect predicted label of each iteration
        train_pred = []
        test_pred = []

        model.train()

        for i_batch, sample in enumerate(train_dataloader[rank]):
            model.zero_grad()
            H, A1, A2, Y, V, keys = sample
            H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device), \
                              Y.to(device), V.to(device)

            # train neural network
            pred = model.module.train_model((H, A1, A2, V))

            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()

            # collect loss, true label and predicted label
            train_losses.append(loss.data.cpu().numpy())
            train_true.append(Y.data.cpu().numpy())
            train_pred.append(pred.data.cpu().numpy())
            # if i_batch>10 : break

        model.eval()
        for i_batch, sample in enumerate(test_dataloader):
            model.zero_grad()
            H, A1, A2, Y, V, keys = sample
            H, A1, A2, Y, V = H.to(device), A1.to(device), A2.to(device), \
                              Y.to(device), V.to(device)

            # train neural network
            pred = model.module.train_model((H, A1, A2, V))

            loss = loss_fn(pred, Y)

            # collect loss, true label and predicted label
            test_losses.append(loss.data.cpu().numpy())
            test_true.append(Y.data.cpu().numpy())
            test_pred.append(pred.data.cpu().numpy())
            # if i_batch>10 : break

        train_losses = np.mean(np.array(train_losses))
        test_losses = np.mean(np.array(test_losses))

        train_pred = np.concatenate(np.array(train_pred), 0)
        test_pred = np.concatenate(np.array(test_pred), 0)

        train_true = np.concatenate(np.array(train_true), 0)
        test_true = np.concatenate(np.array(test_true), 0)

        train_roc = roc_auc_score(train_true, train_pred)
        test_roc = roc_auc_score(test_true, test_pred)
        end = time.time()
        name = args.save_dir + '/save_' + str(epoch) + '.pt'
        torch.save(model.state_dict(), name)
        # save log in csv
        if epoch == 0:
            with open(f'{args.save_dir}/log-rank{rank}.csv', mode='a') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["epoch", "train_loss", "test_loss", "train_roc", "test_roc", "time"])
                writer.writerow([epoch, train_losses, test_losses, train_roc, test_roc, end - st])
        else:
            with open(f'{args.save_dir}/log-rank{rank}.csv', mode='a') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([epoch, train_losses, test_losses, train_roc, test_roc, end - st])

def run_train(fn_DDP, world_size):
    mp.spawn(fn_DDP,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)

if __name__ == '__main__':
    run_train(train_DDP, args.ngpu)

