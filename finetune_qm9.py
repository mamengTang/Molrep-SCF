#coding=utf-8
import os
import argparse
import torch
import torch_geometric
from torch import nn
from model.transformer_model import transformer_1d
import numpy as np
from qm9_data import QM9_our
from model.feature_fussion import TransformerEncoder
from model.gnn_model import GNN,GNNDecoder
from transformers import RobertaConfig, RobertaForMaskedLM
from model.dimenet import DimeNet
from tqdm import tqdm
import torch.nn.functional as F
from utils import to_dense_with_fixed_padding
from process_dataset.MPP.utils.dist import init_distributed_mode
import random
from torch_geometric.utils import remove_self_loops
np.set_printoptions(threshold=np.inf)
device_ids = [2]
device = "cpu"
BATCH_SIZE=256 * len(device_ids)
EPOCH = 1
class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


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

        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)
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
    def __init__(self, model_state_dict,input_dim, hidden_dim, output_dim, device, dropout=0.5):
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
        node_representation_3d = self.encoder_3d(batch_data.x[:, 0].long(), batch_data.pos,
                                 batch_data.batch)
        node_representation_3d, mask_3d = to_dense_with_fixed_padding(node_representation_3d,
                                                                           batch_data.batch, 50)

        token_representation_1d = token_representation_1d + self.token_bias.unsqueeze(0).expand(batch_size, -1,-1)
        node_representation_2d = node_representation_2d + self.graph_bias.unsqueeze(0).expand(batch_size, -1,-1)
        node_representation_3d = node_representation_3d + self.molecule_bias.unsqueeze(0).expand(batch_size, -1, -1)

        emd_sum = torch.cat([torch.cat([token_representation_1d, node_representation_2d], dim=1), node_representation_3d], dim=1)
        mask_label = torch.cat([torch.cat([mask_1d,mask_2d],dim=1),mask_3d],dim=1)

        fussion_feature = self.feature_fussion(emd_sum, mask_label)


        return fussion_feature, mask_label


def finetune_train(train_loader, cls_predictor, optimizer,tag, epoch, target_per_class=70):
    step = 0
    total_loss = 0
    cls_predictor.train()


    for step, batch_data in enumerate(tqdm(train_loader, desc=tag)):
        if step == 100:
            return total_loss / (step + 1)
        batch_data = batch_data.to(device)
        if batch_data.x.shape[0] == 1 or batch_data.batch[-1] == 0:
            pass
        else:
            try:
                pred_result,emd,original_emd = cls_predictor(batch_data)
                valid_label = batch_data.y == batch_data.y
                loss = F.mse_loss(
                    pred_result.to(torch.float32)[valid_label], 
                    batch_data.y.to(torch.float32)[valid_label]
                )

                if torch.isnan(loss).any():
                    print("Loss contains NaN!")
                    break
                total_loss +=loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            except:
                continue
    return total_loss / (step + 1)




def finetune_evaluate(valid_loader, cls_predictor, tag, device):
    cls_predictor.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    
    with torch.no_grad():  # Disable gradient calculation to save memory
        for step, batch_data in enumerate(tqdm(valid_loader, desc=tag)):
            batch_data = batch_data.to(device)  # Move data to device
            
            try:
                # Get the predictions from the model
                pred_result, _, _ = cls_predictor(batch_data)
                
                # Append the true values and predictions
                y_true.append(batch_data.y.view(pred_result.shape).detach().cpu())
                y_pred.append(pred_result.detach().cpu())
                
                # Concatenate the predictions and labels
                total_preds = torch.cat((total_preds, pred_result.cpu()), 0)
                total_labels = torch.cat((total_labels, batch_data.y.cpu()), 0)
            except:
                continue
    
    # Convert the lists to tensors
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    
    # Calculate Mean Absolute Error (MAE)
    mae = F.l1_loss(total_preds, total_labels)  # or alternatively: torch.abs(total_preds - total_labels).mean()
    
    return mae.item()  # Return the MAE as a scalar value

# qm9_header_to_target = {
#     "Alpha": 1,
#     "Gap": 4,
#     "HOMO": 2,
#     "LUMO": 3,
#     "Mu": 0,
#     "Cv": 11,
#     "G298": 10,
#     "H298": 9,
#     "R2": 5,
#     "U298": 8,
#     "U0": 7,
#     "Zpve": 6,
#     "Avg": None,  # 平均指标，不对应QM9的单一target
# }
target = 1
class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data
SAVE_MODEL = './save_model/model_4.pth'

def main():
    #test和valid要换一下可以达到74 75
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="qm9")
    parser.add_argument("--hidden_size", type=int, default="64")
    parser.add_argument("--num_class", type=int, default="1")
    parser.add_argument("--learning_rate", type=float, default="3e-5")
    parser.add_argument("--weight_decay", type=float, default="1e-5")
    parser.add_argument("--patience", type=float, default=300)

    args = parser.parse_args()
    init_distributed_mode(args)
    print(args)

    path = './dataset/3d/QM9'
    transform = MyTransform()
    dataset = QM9_our(path, transform=transform).shuffle()
    

    randperm = torch.randperm(len(dataset))
    train_idxs = randperm[: int((0.84) * len(dataset))]
    valid_idxs = randperm[int(0.84 * len(dataset)):int(0.92 * len(dataset))]
    t_idxs = randperm[int(0.92 * len(dataset)):]


    train_loader = torch_geometric.loader.DataLoader(
        dataset[train_idxs], batch_size=BATCH_SIZE, drop_last=True, shuffle=True
    )
    valid_loader = torch_geometric.loader.DataLoader(
        dataset[valid_idxs], batch_size=BATCH_SIZE, drop_last=True, shuffle=True
    )
    test_loader = torch_geometric.loader.DataLoader(
        dataset[t_idxs], batch_size=BATCH_SIZE, drop_last=True, shuffle=True
    )

    seeds = [42]

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
        best_val_epoch = 0
        best_val_auc = 0
        best_test_auc = 0
        patience = args.patience
        counter = 0

        for epoch in range(1, args.epochs + 1):
            print("Epoch {}".format(epoch))
            print("Training...")
            train_loss= finetune_train(train_loader, cls_predictor, optimizer,"Training",epoch)
            print('Epoch {}, train loss: {:.4f}'.format(epoch, train_loss))
            print("Evaluating...")
            valid_perf = finetune_evaluate(valid_loader, cls_predictor,"Validating",device)
            test_perf = finetune_evaluate(test_loader, cls_predictor,"Testing",device)
            print("Validation", valid_perf, "Test", test_perf)
            val_auc = valid_perf
            test_auc = test_perf
            valid_curve.append(val_auc)
            test_curve.append(test_auc)

            now_best_val_epoch = np.argmin(np.array(valid_curve))
            if best_val_epoch != now_best_val_epoch:
                best_val_epoch = now_best_val_epoch
            #print("Test score: {}".format(test_curve[best_val_epoch]))

            if val_auc < best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc
                counter = 0
            else:
                counter+=1

            if counter >= patience:
                print("Early stopping triggered")
                break

        best_val_epoch = np.argmin(np.array(valid_curve))

        print("Finished training!")
        test_perf = finetune_evaluate(test_loader, cls_predictor,"Testing",device)
        print("Test score: {}".format(test_perf[dataset.eval_metric]))
        print(best_test_auc)


if __name__ == "__main__":
    main()
