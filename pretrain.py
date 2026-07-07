#coding=utf-8
import os
import argparse
import torch
import torch_geometric
from torch_geometric.nn import global_mean_pool
# 在代码中启用异常检测
torch.autograd.set_detect_anomaly(True)
from torch import nn
from torch.utils.data import Dataset
import torch.optim as optim
from pcqm4m import PCQM4Mv2Dataset
from model.transformer_model import transformer_1d,AttentionPoolingWithMask
import numpy as np
from model.gnn_model import GNN,GNNDecoder,GNN3to2
from transformers import RobertaConfig, RobertaForMaskedLM
from model.dimenet import DimeNet
from model.feature_fussion import TransformerEncoder
import torch.multiprocessing
from tqdm import tqdm
from utils import mask_tokens_batch,mask_graph_batch,add_noise_to_3d_structure_batch,to_dense_with_fixed_padding
from utils import mask_tokens_batch2,mask_graph_batch2,add_noise_to_3d_structure_batch2
from torch_geometric.data import Batch
from loss import sce_loss,masked_cross_entropy_loss,molecular_denoising_loss
from torch_geometric.utils import scatter
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
output_model_dir = './save_model/'
BATCH_SIZE=256
EPOCH = 20
LOAD_FROM_LAST=False
from torch.nn.modules.loss import _Loss
import torch.distributed as dist
from torch.nn import Linear, Sequential, SiLU, LayerNorm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12324'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup(rank, world_size):
    """ 初始化分布式环境 """
    dist.init_process_group(
        backend="nccl",  # 使用 NCCL 后端用于 GPU 通信
        init_method="env://",  # 通过环境变量进行初始化
        world_size=world_size,
        rank=rank
    )

def cleanup():
    """ 清理分布式环境 """
    dist.destroy_process_group()

def save_model(save_tag, epoch, my_model, optimizer, loss,):
    saver_dict = {
        'epoch': epoch,
        'model_state_dict': my_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_valid_loss': loss
    }
    if save_tag=='best':
        torch.save(saver_dict, output_model_dir + 'model_best.pth')
    elif save_tag == 'last':
        torch.save(saver_dict, output_model_dir + 'model_last.pth')
    else:
        torch.save(saver_dict, output_model_dir + 'model_' + str(epoch) + '.pth')
    return


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename, weights_only=True)

    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['best_valid_loss']

    print(f"Checkpoint loaded from epoch {epoch}, loss: {loss}")

    return epoch, loss

class PreprocessBatch:
    def process(self, batch):
        # batch: torch_geometric.data.Batch

        pos = batch.pos                              # [N, 3]
        batch_idx = batch.batch                      # [N] (long)

        pos_mean = global_mean_pool(pos, batch_idx)  # [G, 3]
        batch.pos = pos - pos_mean[batch_idx]        # 每个节点减所属图均值
        return batch


class ClipInfoCELoss(_Loss):
    def __init__(self, temperature=0.07):
        super(ClipInfoCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits_per_image, logits_per_text):

        # Compute similarity scores (cosine similarity is used in CLIP)
        sim_i2t = F.cosine_similarity(logits_per_image.unsqueeze(1), logits_per_text.unsqueeze(0), dim=2)  # (batch_size, batch_size)
        sim_t2i = F.cosine_similarity(logits_per_text.unsqueeze(1), logits_per_image.unsqueeze(0), dim=2)  # (batch_size, batch_size)
        
        # Apply temperature scaling
        sim_i2t /= self.temperature
        sim_t2i /= self.temperature
        
        # Labels are the identity of each example
        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
        
        # Cross-entropy loss for image-to-text similarity and text-to-image similarity
        loss_i = F.cross_entropy(sim_i2t, labels)  # image-to-text
        loss_t = F.cross_entropy(sim_t2i, labels)  # text-to-image
        
        # Average both losses
        loss = (loss_i + loss_t) / 2
        return loss, labels
    
class CrossAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        self.attention_1 = Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_2 = Linear(hidden_dim, hidden_dim, bias=False)
        self.rho = Sequential(
            Linear(hidden_dim, hidden_dim),
            SiLU(),
            Linear(hidden_dim, hidden_dim),
            SiLU(),
            LayerNorm(hidden_dim),
        )
        self.phi1 = LayerNorm(hidden_dim)
        self.phi2 = LayerNorm(hidden_dim)

    def forward(self, x_mole, x_conf, batch,B):
        counts = torch.bincount(batch).tolist()
        mchunks = torch.chunk(x_mole, chunks=B, dim=0)
        mvalid_chunks = [chunk[:c, :] for chunk, c in zip(mchunks, counts)]
        x_mole = torch.cat(mvalid_chunks, dim=0)

        cchunks = torch.chunk(x_conf, chunks=B, dim=0)
        cvalid_chunks = [chunk[:c, :] for chunk, c in zip(cchunks, counts)]
        x_conf = torch.cat(cvalid_chunks, dim=0)


        dot_product = torch.matmul(self.attention_1(self.phi1(x_mole)),
                                   self.attention_2(self.phi2(x_conf)).transpose(1, 0))
        mask = (batch.unsqueeze(1) == batch.unsqueeze(0)).float()
        max_values = (dot_product * mask).max(dim=1, keepdim=True).values
        masked_dot_product = (dot_product - max_values) * mask
        attention_weights = masked_dot_product.exp() / (masked_dot_product.exp() * mask).sum(dim=1, keepdim=True)
        attention_weights = attention_weights * mask
        x_weighted = torch.matmul(attention_weights, x_conf)
        x_encoded = self.rho(x_weighted)
        return x_encoded, attention_weights



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder_1d = transformer_1d()
        self.config = RobertaConfig.from_pretrained('./roberta-base')
        self.config.hidden_size = 128 
        self.config.mask_token_id = 2586
        self.config.type_vocab_size=1
        self.config.vocab_size=2586+1
        self.config.max_position_embeddings=60
        self.config.num_attention_heads = 8
        self.decoder_1d = RobertaForMaskedLM(self.config)
        self.encoder_2d = GNN(num_layer=3, hidden_dim=128,output_dim=128)
        self.encoder_3to2 = GNN3to2(num_layer=3, hidden_dim=128,output_dim=128)
        self.decoder_2d = GNNDecoder(hidden_dim=128, out_dim=9)
        self.decoder_3d = GNNDecoder(hidden_dim=128, out_dim=3)
        self.encoder_3d = DimeNet(hidden_channels=128,
                                  num_blocks=3,
                                  num_bilinear=8,
                                  num_spherical=7,
                                  num_radial=6,
                                  out_channels=128
                                  )
        self.feature_fussion = TransformerEncoder(128, 128, 8, 4)

        self.token_bias = nn.Parameter(torch.randn(50, 128))
        self.graph_bias = nn.Parameter(torch.randn(50, 128))
        self.molecule_bias = nn.Parameter(torch.randn(50, 128))

        self.preprocessor = PreprocessBatch()
        self.fc_mu = nn.Linear(128, 128)
        self.fc_var = nn.Linear(128, 128)

        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.attention_pooling_1d = AttentionPoolingWithMask(128)
        self.attention_pooling_2d = AttentionPoolingWithMask(128)
        self.attention_pooling_3d = AttentionPoolingWithMask(128)
        self.fuse_3d = nn.Linear(128 * 2, 128)
        self.d_23 = CrossAttention(128)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, batch_data):
        smiles_embedding,GNN_embedding_2d,GNN_embedding_3d,GNN_embedding_3d_y = self.forward2(batch_data)

        batch_size = len(batch_data)
        batch_data = batch_data.cuda()
        self.preprocessor.process(batch_data)

        tokens_emb = torch.tensor(np.array(batch_data.tokens), dtype=torch.long).cuda()
        smi_mask = torch.tensor(np.array(batch_data.attention_mask), dtype=torch.bool).cuda()
        batch_masked_tokens, batch_masked_token_indices=mask_tokens_batch(tokens_emb,smi_mask)
        batch_masked_graphs, batch_masked_graph_indices, batch_masked_atom_indices_2d=mask_graph_batch(batch_data,batch_data.atom2substructure,batch_masked_token_indices)

        batch_masked_graphs = Batch.from_data_list(batch_masked_graphs)
        batch_noisy_positions,batch_noisy_positions2, batch_noisy_position_indices, batch_masked_atom_indices_3d = add_noise_to_3d_structure_batch(batch_data.atom2substructure,batch_data.pos,batch_data.pos2,batch_data.batch, batch_masked_token_indices, batch_masked_graph_indices)
        masked_token_representation_1d = self.encoder_1d(batch_masked_tokens, smi_mask)  # (batch_size, seq_length,emd_size)。
        mask_1d = torch.ones(batch_size, 50, dtype=torch.bool).cuda()
        masked_node_representation_2d = self.encoder_2d(batch_masked_graphs.x, batch_masked_graphs.edge_index,
                                                 batch_masked_graphs.edge_attr)  # (num_nodes_in_batch, emb_dim)
        masked_node_representation_2d, mask_2d = to_dense_with_fixed_padding(masked_node_representation_2d,batch_data.batch,50)
        noisy_node_representation_3d = self.encoder_3d(batch_data.x[:, 0].long(), batch_noisy_positions,
                                 batch_data.batch)
        noisy_node_representation_3d, mask_3d = to_dense_with_fixed_padding(noisy_node_representation_3d,
                                                                           batch_data.batch, 50)
        
        noisy_node_representation_3d_2 = self.encoder_3d(batch_data.x[:, 0].long(), batch_noisy_positions2,
                                 batch_data.batch)
        noisy_node_representation_3d_2, mask_3d_2 = to_dense_with_fixed_padding(noisy_node_representation_3d_2,
                                                                           batch_data.batch, 50)
        
        B, N, C = masked_node_representation_2d.shape
        x_2d = masked_node_representation_2d.view(B * N, C)
        x_3d = noisy_node_representation_3d.view(B * N, C)
        x_3d2 = noisy_node_representation_3d_2.view(B * N, C)

        x_3d_fused = torch.cat([x_3d, x_3d2], dim=-1)
        x_3d_fused = self.fuse_3d(x_3d_fused) 


        counts = torch.bincount(batch_masked_graphs.batch).tolist()
        chunks = torch.chunk(x_3d_fused, chunks=B, dim=0)
        valid_chunks = [chunk[:c, :] for chunk, c in zip(chunks, counts)]
        res = torch.cat(valid_chunks, dim=0)  # 形状: [99, 128]
        masked_node_representation_2d2 = self.encoder_3to2(res, batch_masked_graphs.edge_index,
                                                 batch_masked_graphs.edge_attr) 
        x_topo, mask_2d2 = to_dense_with_fixed_padding(masked_node_representation_2d2,batch_data.batch,50)
        x_conf, attn_weight = self.d_23(x_3d,x_topo.view(B * N, C), batch_data.batch,B)

        noisy_node_representation_3d, _ = to_dense_with_fixed_padding(x_conf,batch_data.batch,50)
        masked_token_representation_1d = masked_token_representation_1d + self.token_bias.unsqueeze(0).expand(batch_size, -1, -1)
        masked_node_representation_2d = masked_node_representation_2d + self.graph_bias.unsqueeze(0).expand(batch_size, -1, -1)
        noisy_node_representation_3d = noisy_node_representation_3d + self.molecule_bias.unsqueeze(0).expand(batch_size, -1, -1)

        masked_emd_sum = torch.cat([torch.cat([masked_token_representation_1d, masked_node_representation_2d], dim=1), noisy_node_representation_3d], dim=1)
        mask_label = torch.cat([torch.cat([mask_1d,mask_2d],dim=1),mask_3d],dim=1)


        fussion_feature = self.feature_fussion(masked_emd_sum,mask_label)
        token_representation_1d,node_representation_2d, node_representation_3d = torch.chunk(fussion_feature, 3, dim=1)
        node_representation_2d = node_representation_2d [mask_2d.bool()]
        node_representation_3d = node_representation_3d [mask_3d.bool()]

        predict_token_representation_1d = self.decoder_1d(inputs_embeds=token_representation_1d, attention_mask=smi_mask)
        predict_token_representation_1d = predict_token_representation_1d.logits
        predict_node_representation_2d = self.decoder_2d(node_representation_2d,batch_data.edge_index, batch_data.edge_attr)
        predict_node_representation_3d = self.decoder_3d(node_representation_3d, batch_data.edge_index,
                                                         batch_data.edge_attr)

        return smiles_embedding,GNN_embedding_2d,GNN_embedding_3d,GNN_embedding_3d_y, tokens_emb,batch_masked_token_indices,predict_token_representation_1d,batch_data.x[batch_masked_atom_indices_2d],predict_node_representation_2d[batch_masked_atom_indices_2d],batch_data.pos[batch_masked_atom_indices_3d],predict_node_representation_3d[batch_masked_atom_indices_3d]
    
    def forward2(self, batch_data):

        batch_size = len(batch_data)
        batch_data = batch_data.cuda()
        self.preprocessor.process(batch_data)

        tokens_emb = torch.tensor(np.array(batch_data.tokens), dtype=torch.long).cuda()
        smi_mask = torch.tensor(np.array(batch_data.attention_mask), dtype=torch.bool).cuda()
        # =============
        batch_masked_tokens, batch_masked_token_indices=mask_tokens_batch2(tokens_emb,smi_mask)
        batch_masked_graphs, batch_masked_graph_indices, batch_masked_atom_indices_2d=mask_graph_batch2(batch_data,batch_data.atom2substructure,batch_masked_token_indices)
        batch_masked_graphs = Batch.from_data_list(batch_masked_graphs)
        batch_noisy_positions,batch_noisy_positions2, batch_noisy_position_indices, batch_masked_atom_indices_3d = add_noise_to_3d_structure_batch2(batch_data.atom2substructure,batch_data.pos,batch_data.pos2,batch_data.batch, batch_masked_token_indices, batch_masked_graph_indices)
        # =============
        masked_token_representation_1d = self.encoder_1d(batch_masked_tokens, smi_mask)  # (batch_size, seq_length,emd_size)
        smiles_embedding = self.attention_pooling_1d(masked_token_representation_1d, ~(~smi_mask^batch_masked_token_indices))

        masked_node_representation_2d = self.encoder_2d(batch_masked_graphs.x, batch_masked_graphs.edge_index,
                                                 batch_masked_graphs.edge_attr)  
        masked_node_representation_2d, mask_2d = to_dense_with_fixed_padding(masked_node_representation_2d,batch_data.batch,50)
        GNN_embedding_2d = self.attention_pooling_2d(masked_node_representation_2d, ~(~mask_2d^batch_masked_graph_indices))

        noisy_node_representation_3d = self.encoder_3d(batch_data.x[:, 0].long(), batch_noisy_positions,
                                 batch_data.batch)
        noisy_node_representation_3d_, mask_3d = to_dense_with_fixed_padding(noisy_node_representation_3d,
                                                  batch_data.batch, 50)
        GNN_embedding_3d = self.attention_pooling_3d(noisy_node_representation_3d_, ~(~mask_3d^batch_masked_graph_indices))
        
        noisy_node_representation_3d_y = self.encoder_3d(batch_data.x[:, 0].long(), batch_noisy_positions2,
                                 batch_data.batch)
        noisy_node_representation_3d__, mask_3d = to_dense_with_fixed_padding(noisy_node_representation_3d_y,
                                                  batch_data.batch, 50)
        GNN_embedding_3d_y = self.attention_pooling_3d(noisy_node_representation_3d__, ~(~mask_3d^batch_masked_graph_indices))
        

        return smiles_embedding,GNN_embedding_2d,GNN_embedding_3d,GNN_embedding_3d_y


def pretrain_train(train_loader,my_model,optimizer,rank):
    train_data_len = len(train_loader)
    my_model.train()
    total_loss=0
    for step, batch_data_list in enumerate(tqdm(train_loader, desc="Training", disable=True)):

        modality_1,modality_2,modality_3,modality_3_,token_representation_1d,batch_masked_token_indices,predict_token_representation_1d,node_representation_2d,predict_node_representation_2d,node_representation_3d, predict_node_representation_3d = my_model(batch_data_list)
        loss_1d = masked_cross_entropy_loss(predict_token_representation_1d, token_representation_1d,batch_masked_token_indices)
        loss_2d = sce_loss(predict_node_representation_2d,node_representation_2d)
        loss_3d = molecular_denoising_loss(predict_node_representation_3d,node_representation_3d)
        
        criterion = ClipInfoCELoss()

        loss12,_ = criterion(modality_1, modality_2)
        loss13,_ = criterion(modality_1, modality_3)
        loss23,_ = criterion(modality_2, modality_3)
       
        loss = loss_1d+loss_2d+loss_3d+loss12+loss13+loss23
        total_loss +=loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), 100)
        optimizer.step()
        optimizer.zero_grad()
        if rank == 0:
            print(f"TRAIN: step: {step}, "
                f"loss1d: {loss_1d.item():.2f}, "
                f"loss2d: {loss_2d.item():.2f}, "
                f"loss3d: {loss_3d.item():.2f}, "
                f"loss12: {loss12.item():.2f}, "
                f"loss13: {loss13.item():.2f}, "
                f"loss23: {loss23.item():.2f}, "
                f"grad: {grad_norm.item():.2f}")    

    train_loss = total_loss/train_data_len

    return train_loss


def pretrain_evaluate(valid_loader,my_model,optimizer,rank):
    valid_data_len = len(valid_loader)
    my_model.eval()
    total_loss = 0
    total_loss_1d = 0
    total_loss_2d = 0
    total_loss_3d = 0
    total_loss12 = 0
    total_loss13 = 0
    total_loss23 = 0

    with torch.no_grad():
        for step, batch_data_list in enumerate(tqdm(valid_loader, desc="Validing", disable=True)):
            try:
                modality_1,modality_2,modality_3,token_representation_1d,batch_masked_token_indices,predict_token_representation_1d,node_representation_2d,predict_node_representation_2d,node_representation_3d, predict_node_representation_3d = my_model(batch_data_list)
            except Exception as e:
                print(f"Exception at step {step}: {e}")

            loss_1d = masked_cross_entropy_loss(predict_token_representation_1d, token_representation_1d,batch_masked_token_indices)
            loss_2d = sce_loss(predict_node_representation_2d,node_representation_2d)
            loss_3d = molecular_denoising_loss(predict_node_representation_3d,node_representation_3d)
            
            criterion = ClipInfoCELoss()

            # 计算 InfoNCE 损失
            loss12,_ = criterion(modality_1, modality_2)
            loss13,_ = criterion(modality_1, modality_3)
            loss23,_ = criterion(modality_2, modality_3)

            losses =loss_1d+loss_2d+loss_3d+loss12+loss13+loss23

            total_loss += losses.detach().cpu()
            total_loss_1d += loss_1d.detach().cpu()
            total_loss_2d += loss_2d.detach().cpu()
            total_loss_3d += loss_3d.detach().cpu()
            total_loss12 += loss12.detach().cpu()
            total_loss13 += loss13.detach().cpu()
            total_loss23 += loss23.detach().cpu()

        valid_loss =total_loss/valid_data_len
        mean_loss_1d = total_loss_1d / valid_data_len
        mean_loss_2d = total_loss_2d / valid_data_len
        mean_loss_3d = total_loss_3d / valid_data_len
        mean_loss12 = total_loss12 / valid_data_len
        mean_loss13 = total_loss13 / valid_data_len
        mean_loss23 = total_loss23 / valid_data_len

        if rank == 0:
            print(f"Valid: step: {step}, "
                  f"loss1d: {mean_loss_1d.item():.2f}, "
                  f"loss2d: {mean_loss_2d.item():.2f}, "
                  f"loss3d: {mean_loss_3d.item():.2f}, "
                  f"loss12: {mean_loss12.item():.2f}, "
                  f"loss13: {mean_loss13.item():.2f}, "
                  f"loss23: {mean_loss23.item():.2f}, ")  
        return valid_loss

def train_mp(rank,world_size,strrr):
    dataset = PCQM4Mv2Dataset()
    print('dataset load finish')

    randperm = torch.randperm(len(dataset))
    train_idxs = randperm[: int((0.98) * len(dataset))]
    valid_idxs = randperm[int(0.98 * len(dataset)):]


    train_loader = torch_geometric.loader.DataLoader(
        dataset[train_idxs], batch_size=BATCH_SIZE, drop_last=True, shuffle=True
    )
    valid_loader = torch_geometric.loader.DataLoader(
        dataset[valid_idxs], batch_size=BATCH_SIZE, drop_last=True, shuffle=True
    )

    train_data_len = len(train_loader)
    val_data_len = len(valid_loader)
    print('train dataset length: ', train_data_len)
    print('val dataset length: ', val_data_len)

    print("??????"+str(rank))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    my_model = MyModel().to(rank)
    my_model = DDP(my_model, device_ids=[rank], find_unused_parameters=True)

    model_param_group = []
    model_param_group.append({'params': my_model.parameters(), 'lr': 0.0001 * 1})

    optimizer = optim.Adam(model_param_group, weight_decay=1e-5)
    optimal_loss = 1e10

    best_valid_loss = 10000
    current_epoch = 0

    if LOAD_FROM_LAST:
        current_epoch, best_valid_loss = load_checkpoint('./save_model/model_8.pth', my_model, optimizer)

    for epoch in range(current_epoch, EPOCH + 1):
        print('Epoch: {}'.format(epoch))
        print("Training")
        train_loss = pretrain_train(train_loader,my_model,optimizer,rank)
        print('Epoch {}, train loss: {:.4f}'.format(epoch + 1, train_loss))
        save_model('temp', epoch + 1, my_model, optimizer, best_valid_loss)
        print("  ")
        save_model('last', EPOCH, my_model, optimizer, best_valid_loss)
    dist.destroy_process_group()

def main():
    world_size = 1
    mp.spawn(train_mp,
        args=(world_size,"555"),
        nprocs=world_size,
        join=True)

if __name__ == "__main__":
    main()
    