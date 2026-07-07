#coding=utf-8
import os
import pickle
import argparse
import pandas as pd
import torch
import torch_geometric
from torch import nn
from model.transformer_model import transformer_1d
import numpy as np
from model.feature_fussion import TransformerEncoder
from model.gnn_model import GNN,GNNDecoder
from transformers import RobertaConfig, RobertaForMaskedLM
from model.dimenet import DimeNet
import torch.multiprocessing
from tqdm import tqdm
import torch.nn.functional as F
from utils import to_dense_with_fixed_padding
from process_dataset.MPP.utils.dist import init_distributed_mode
from process_dataset.MPP.data.DCGraphPropPredDataset.dataset import DCGraphPropPredDataset
from process_dataset.MPP.utils.evaluate import Evaluator
import random
from torch_geometric.loader import DataLoader

np.set_printoptions(threshold=np.inf)

device_ids = [1]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=256 * len(device_ids)
EPOCH = 1



def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attn_weights = nn.Linear(input_dim, 1)

    def forward(self, x, mask):
        scores = self.attn_weights(x).squeeze(-1)
        scores[mask == 0] = -1e9

        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1) # torch.Size([64, 100, 1])
        if torch.isnan(attn_weights).any():
            print("Tensor contains NaN values!")

            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        context = (attn_weights * x).sum(dim=1)

        return context


def masked_average(x, mask):
    mask = mask.unsqueeze(-1)
    x_masked = x * mask
    valid_counts = mask.sum(dim=1).clamp(min=1)
    avg_result = x_masked.sum(dim=1) / valid_counts

    return avg_result


def masked_sum(x, mask):
    mask = mask.unsqueeze(-1)
    x_masked = x * mask
    sum_result = x_masked.sum(dim=1)

    return sum_result

class property_predictor(nn.Module):
    def __init__(self, model_state_dict,input_dim, hidden_dim, output_dim, device, dropout=0.3):
        super(property_predictor, self).__init__()
        self.attn_pooling = AttentionPooling(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.pre = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.my_model = MyModel(device)
        self.my_model.load_state_dict(model_state_dict, strict=False)

        self.model_no_pretrain = MyModel(device)

    def forward(self, batch_data, aggre ='attn'):
        fussion_feature, valid_position = self.my_model(batch_data)

        original_fussion_feature, original_valid_position = self.model_no_pretrain(batch_data)
        if aggre == 'attn':
            x = self.attn_pooling(fussion_feature,valid_position)  # [batch_size, 128]
            original_x = self.attn_pooling(original_fussion_feature,original_valid_position)
        elif aggre == 'mean':
            x = masked_average(fussion_feature,valid_position)
            original_x = masked_average(original_fussion_feature, original_valid_position)
        elif aggre == 'sum':
            x = masked_sum(fussion_feature, valid_position)
            original_x = masked_sum(original_fussion_feature,original_valid_position)

        emd=x
        original_emd = original_x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # dropout
        x = self.pre(x)

        return x,emd,original_emd

class MyModel(nn.Module):
    def __init__(self,device):
        super(MyModel, self).__init__()
        self.device = device

        self.encoder_1d = transformer_1d()
        # 加载RoBERTa模型
        self.config = RobertaConfig.from_pretrained('./roberta-base')
        self.config.hidden_size = 128  # 修改 hidden_size
        self.config.mask_token_id = 2586
        self.config.type_vocab_size = 1
        self.config.vocab_size = 2586 + 1
        self.config.max_position_embeddings = 60
        self.config.num_attention_heads = 8
        self.decoder_1d = RobertaForMaskedLM(self.config)
        self.encoder_2d = GNN(num_layer=3, hidden_dim=128, output_dim=128)
        self.decoder_2d = GNNDecoder(hidden_dim=128, out_dim=9)
        self.decoder_3d = GNNDecoder(hidden_dim=128, out_dim=3)
        self.encoder_3d = DimeNet(hidden_channels=128,  # 隐藏层大小
                                  num_blocks=3,  # 多少层 DimeNet block
                                  num_bilinear=8,  # 双线性层数
                                  num_spherical=7,  # 球坐标展开阶数
                                  num_radial=6,  # 径向基展开阶数
                                  out_channels=128  # 输出通道
                                  )
        self.feature_fussion = TransformerEncoder(128, 128, 8, 4)

        self.token_bias = nn.Parameter(torch.randn(50, 128))
        self.graph_bias = nn.Parameter(torch.randn(50, 128))
        self.molecule_bias = nn.Parameter(torch.randn(50, 128))


    def forward(self, batch_data):
        #batch_data = Batch.from_data_list(batch_data)
        #print(batch_data)

        batch_size = len(batch_data)
        batch_data = batch_data.to(self.device)

        # 1d normal
        tokens_emb = torch.tensor(np.array(batch_data.tokens), dtype=torch.long).to(self.device)
        smi_mask = torch.tensor(np.array(batch_data.attention_mask), dtype=torch.bool).to(self.device)

        token_representation_1d = self.encoder_1d(tokens_emb, smi_mask)  # (batch_size, seq_length,emd_size)。
        mask_1d = torch.ones(batch_size, 50, dtype=torch.bool).to(self.device)
        node_representation_2d = self.encoder_2d(batch_data.x, batch_data.edge_index,
                                                 batch_data.edge_attr)  # (num_nodes_in_batch, emb_dim)
        node_representation_2d, mask_2d = to_dense_with_fixed_padding(node_representation_2d,
                                                                             batch_data.batch, 50)

        token_representation_1d = token_representation_1d + self.token_bias.unsqueeze(0).expand(batch_size, -1,-1)
        node_representation_2d = node_representation_2d + self.graph_bias.unsqueeze(0).expand(batch_size, -1,-1)

        emd_sum = torch.cat([token_representation_1d, node_representation_2d], dim=1)
        mask_label = torch.cat([mask_1d, mask_2d], dim=1)
        fussion_feature = self.feature_fussion(emd_sum, mask_label)

        return fussion_feature, mask_label


def finetune_train(train_loader, cls_predictor, optimizer,tag, epoch, target_per_class=70):
    step = 0
    total_loss = 0
    cls_predictor.train()


    for step, batch_data in enumerate(tqdm(train_loader, desc=tag)):
        batch_data = batch_data.to(device)
        if batch_data.x.shape[0] == 1 or batch_data.batch[-1] == 0:
            pass
        else:
            pred_result,emd,original_emd = cls_predictor(batch_data)
            valid_label = batch_data.y == batch_data.y
            criterion = nn.BCEWithLogitsLoss(reduction="none")
            loss = criterion(
                pred_result.to(torch.float32)[valid_label], torch.tensor(batch_data.y).to(torch.float32)[valid_label].to(batch_data.x.device))
            loss = loss.mean()   # ⭐关键修复点
            if torch.isnan(loss).any():
                print("Loss contains NaN!")
                break
            total_loss +=loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return total_loss / (step + 1)




def finetune_evaluate(valid_loader,cls_predictor,evaluator,tag):
    valid_loader = valid_loader
    valid_data_len = len(valid_loader)
    cls_predictor.eval()
    y_true = []
    y_pred = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(valid_loader, desc=tag)):
            batch_data = batch_data.to(device)
            try:
                pred_result,_,_ = cls_predictor(batch_data)
                y_true.append(torch.tensor(batch_data.y).view(pred_result.shape).detach().cpu())
                y_pred.append(pred_result.detach().cpu())
                total_preds = torch.cat((total_preds, pred_result.cpu()), 0)
                total_labels = torch.cat((total_labels, torch.tensor(batch_data.y).cpu()), 0)
            except:
                continue
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)

from finetune2.loader import MoleculeDataset
from splitters import scaffold_split, moleculeace_split
from torch.utils.data import Subset

def get_dataloader(args, seed=0):
    # Setup dataset
    dataset = MoleculeDataset("./dataset/moleculenet",
                              args.dataset)

    num_task = dataset.num_task
    print('Loading dataset {} of size {} with num_task={}'.format(args.dataset, len(dataset), num_task))

   
    train_idx, val_idx, test_idx = scaffold_split(dataset.smiles, frac_valid=0.1, frac_test=0.1, balanced=False)
    
    train_dataset, val_dataset, test_dataset = \
        Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

    import pandas as pd
    from rdkit import Chem

    # ===== SMILES标准化函数 =====
    def standardize_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None

    train = []
    valid = []
    test = []

    for i in train_dataset:
        train.append(standardize_smiles(i['smiles_ori']))
    
    for i in val_dataset:
        valid.append(standardize_smiles(i['smiles_ori']))

    for i in test_dataset:
        test.append(standardize_smiles(i['smiles_ori']))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return dataset, train_loader, val_loader, test_loader

SAVE_MODEL = './save_model2/model_15.pth'
def main():
    #test和valid要换一下可以达到74 75
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="bbbp")
    parser.add_argument("--hidden_size", type=int, default="64")
    parser.add_argument("--num_class", type=int, default="1")
    parser.add_argument("--learning_rate", type=float, default="3e-5")
    parser.add_argument("--weight_decay", type=float, default="1e-5")
    parser.add_argument("--patience", type=float, default=30)

    args = parser.parse_args()
    init_distributed_mode(args)
    batch_sizes = [16, 32, 64, 128]
    lrs = [1e-5, 5e-5, 1e-4]
    eval_epochs = [20, 30, 50, 60, 100]
    for i in batch_sizes:
        for j in lrs:
            args.batch_size = i
            args.learning_rate = j
            print(args)

            # Setup dataset and dataloader
            dataset, train_loader, valid_loader, test_loader = get_dataloader(args)

            evaluator = Evaluator(args.dataset, dataset=dataset)


            seeds = [42]
            results = []

            for seed in seeds:
                set_seed(seed)
                checkpoint = torch.load(SAVE_MODEL, map_location=device)
                model_state_dict = {}
                print(checkpoint['epoch'])
                for k, v in checkpoint['model_state_dict'].items():
                    model_state_dict[k[7:]] = v  # 去掉module
                cls_predictor = property_predictor(model_state_dict,128,64,args.num_class,device).to(device)
                optimizer = torch.optim.Adam(cls_predictor.parameters(),lr=args.learning_rate*3, weight_decay=args.weight_decay)
                valid_curve = []
                test_curve = []
                best_val_auc = 0
                best_test_auc = 0

                for epoch in range(1, args.epochs + 1):
                    print("Epoch {}".format(epoch))
                    print("Training...")
                    train_loss= finetune_train(train_loader, cls_predictor, optimizer,"Training",epoch)
                    print('Epoch {}, train loss: {:.4f}'.format(epoch, train_loss))
                    print("Evaluating...")
                    valid_perf = finetune_evaluate(valid_loader, cls_predictor, evaluator,"Validating")
                    test_perf = finetune_evaluate(test_loader, cls_predictor, evaluator,"Testing")
                    
                    val_auc = valid_perf['rocauc']
                    test_auc = test_perf['rocauc']
                    valid_curve.append(val_auc)
                    test_curve.append(test_auc)

                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_test_auc = test_auc
                    print(
                        "epoch", epoch,
                        "Validation", f"{valid_perf['rocauc']:.4f}",
                        "Test", f"{test_perf['rocauc']:.4f}",
                        "Best valid", f"{best_val_auc:.4f}",
                        "Best test", f"{best_test_auc:.4f}"
                    )
                    if epoch in eval_epochs:
                        results.append(best_test_auc)

                print("Finished training!")
                print(args)
                print(results)


if __name__ == "__main__":
    main()
