import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layer import *


class GAT(nn.Module):
    def __init__(self, args):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.device = args.device
        self.num_nodes = len(args.entity2id)
        self.entity_in_dim = args.dim
        self.entity_out_dim = args.dim
        self.num_relation = len(args.relation2id)
        self.relation_in_dim = args.dim
        self.relation_out_dim = args.dim
        self.nheads_GAT = args.n_heads
        self.neg_num = args.neg_num_gat

        self.drop_GAT = args.dropout_gat
        self.alpha = args.alpha_gat # For leaky relu

        # Initial Embedding
        self.entity_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.entity_in_dim))
        self.relation_embeddings = nn.Parameter(torch.randn(self.num_relation, self.relation_in_dim))
        if args.pre_trained:
            self.entity_embeddings = nn.Parameter(torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/entity2vec.pkl', 'rb'))).float())
            self.relation_embeddings = nn.Parameter(torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/relation2vec.pkl', 'rb'))).float())
        # Final output Embedding
        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim * self.nheads_GAT))
        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.relation_out_dim * self.nheads_GAT))

        self.spgat = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT)

        self.W_entities = nn.Parameter(torch.zeros(size=(self.entity_in_dim, self.entity_out_dim * self.nheads_GAT)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)
        
        self.knowledge_bank = nn.ParameterList([
            nn.Parameter(torch.randn(100, self.entity_in_dim)) 
            for _ in range(3) 
        ])
        self.bank_projections = nn.ModuleList([
            nn.Linear(self.entity_in_dim, self.entity_out_dim, bias=False)
            for _ in range(3)
        ])
        
        
        self.release_gate = nn.Sequential(
            nn.Linear(self.entity_out_dim*2, self.nheads_GAT),
            nn.Softmax(dim=-1)
        )

    def _accumulate_knowledge(self, embeddings, bank_idx):
        
        bank = self.knowledge_bank[bank_idx]
        similarity = torch.mm(embeddings, bank.T)
        top_k = torch.topk(similarity, k=3, dim=1)
        
        
        if self.training and torch.rand(1) < 0.1:
            new_entry = embeddings.mean(dim=0).detach()
            self.knowledge_bank[bank_idx] = nn.Parameter(
                torch.cat([bank, new_entry.unsqueeze(0)], dim=0)
            )
        
        return self.bank_projections[bank_idx](bank[top_k.indices].mean(dim=1))
        
    def forward(self, adj, train_indices):
        edge_list = adj[0]
        if(CUDA):
            edge_list = edge_list.to(self.device)

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        self.relation_embeddings.data = F.normalize(
            self.relation_embeddings.data, p=2, dim=1).detach()

        mask_indices = torch.unique(train_indices[:, 2]).to(self.device)
        mask = torch.zeros(self.entity_embeddings.shape[0]).to(self.device)
        mask[mask_indices] = 1.0

        out_entity, out_relation = self.spgat(self.entity_embeddings, self.relation_embeddings, edge_list)
        out_entity = F.normalize(self.entity_embeddings.mm(self.W_entities)
                                 + mask.unsqueeze(-1).expand_as(out_entity) * out_entity, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity.data
        self.final_relation_embeddings.data = out_relation.data
        
        struct_knowledge = self._accumulate_knowledge(e_embed, 0)
        img_knowledge = self._accumulate_knowledge(e_img_embed, 1)
        text_knowledge = self._accumulate_knowledge(e_txt_embed, 2)
        
        
        gate_values = self.release_gate(
            torch.cat([out_entity, struct_knowledge + img_knowledge + text_knowledge], dim=1)
        )
        
        
        enhanced_entity = out_entity + torch.einsum('bd,b->bd', 
            struct_knowledge + img_knowledge + text_knowledge,
            gate_values.mean(dim=1)
        )
        
        
        out_entity = F.normalize(enhanced_entity, p=2, dim=1)
        return out_entity, out_relation

    def loss_func(self, train_indices, entity_embeddings, relation_embeddings):
        len_pos_triples = int(train_indices.shape[0] / (int(self.neg_num) + 1))
        pos_triples = train_indices[:len_pos_triples]
        neg_triples = train_indices[len_pos_triples:]
        pos_triples = pos_triples.repeat(int(self.neg_num), 1)

        source_embeds = entity_embeddings[pos_triples[:, 0]]
        relation_embeds = relation_embeddings[pos_triples[:, 1]]
        tail_embeds = entity_embeddings[pos_triples[:, 2]]
        x = source_embeds + relation_embeds - tail_embeds
        pos_norm = torch.norm(x, p=1, dim=1)

        source_embeds = entity_embeddings[neg_triples[:, 0]]
        relation_embeds = relation_embeddings[neg_triples[:, 1]]
        tail_embeds = entity_embeddings[neg_triples[:, 2]]
        x = source_embeds + relation_embeds - tail_embeds
        neg_norm = torch.norm(x, p=1, dim=1)

        y = -torch.ones(int(self.neg_num) * len_pos_triples).to(self.device)
        loss = F.margin_ranking_loss(pos_norm, neg_norm, y, margin=1.0)
        return loss
