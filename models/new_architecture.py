"""
Novel Architecture Components for NF/RO Rejection Prediction
=============================================================
Combines:
  1. GATv2 / GINE Graph Neural Networks with 3D Bond Attributes and Virtual Node
  2. Multi-Scale Molecular Graph Readout (Mean + Max + Sum Pooling)
  3. Deep Physics-Informed Tabular Encoder (24-D Input)
  4. 4-Head Bidirectional Cross-Modal Attention Fusion
  5. Huber Loss Optimization & MC-Dropout Uncertainty Quantification
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, GINEConv,
    global_mean_pool, global_max_pool, global_add_pool
)


# ============================================================================
# Component 1A: GATv2 Molecular Graph Encoder
# ============================================================================

class GATv2MolecularEncoder(nn.Module):
    """
    Molecular Graph Encoder using GATv2 with Edge Features and Virtual Node.
    """
    def __init__(self, node_dim=9, edge_dim=3, hidden_dim=128, num_layers=2,
                 heads=4, dropout=0.2, use_virtual_node=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.use_virtual_node = use_virtual_node
        self.dropout = dropout

        # Initial node & edge projections
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # GATv2 message-passing layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.res_linears = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = hidden_dim
            out_dim_per_head = hidden_dim // heads
            self.convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim_per_head,
                    heads=heads,
                    edge_dim=hidden_dim,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.res_linears.append(nn.Linear(in_dim, hidden_dim))

        # Virtual node
        if use_virtual_node:
            self.virtual_node_embedding = nn.Parameter(torch.randn(1, hidden_dim))
            self.virtual_node_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        # Multi-scale readout projection (Mean + Max + Sum -> 3 * hidden_dim)
        self.readout_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x.float(), data.edge_index, data.edge_attr.float(), data.batch
        )

        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        if self.use_virtual_node:
            num_graphs = batch.max().item() + 1
            vn_expand = self.virtual_node_embedding.expand(num_graphs, -1)

        for i in range(self.num_layers):
            if self.use_virtual_node:
                h = h + vn_expand[batch]

            h_res = self.res_linears[i](h)
            h_new = self.convs[i](h, edge_index, edge_attr=e)
            h = F.gelu(self.batch_norms[i](h_new)) + h_res
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.use_virtual_node and i < self.num_layers - 1:
                pooled_nodes = global_mean_pool(h, batch)
                vn_expand = vn_expand + pooled_nodes
                vn_expand = self.virtual_node_mlp(vn_expand)

        # Multi-scale pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        h_combined = torch.cat([h_mean, h_max, h_sum], dim=-1)

        graph_embedding = self.readout_proj(h_combined)
        return graph_embedding


# ============================================================================
# Component 1B: GINE (Graph Isomorphism Network with Edge features) Encoder
# ============================================================================

class GINEMolecularEncoder(nn.Module):
    """
    Molecular Graph Encoder using GINEConv with 3D Bond Attributes and Virtual Node.
    """
    def __init__(self, node_dim=9, edge_dim=3, hidden_dim=128, num_layers=2,
                 dropout=0.2, use_virtual_node=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_virtual_node = use_virtual_node
        self.dropout = dropout

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.res_linears = nn.ModuleList()

        for layer_idx in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.res_linears.append(nn.Linear(hidden_dim, hidden_dim))

        if use_virtual_node:
            self.virtual_node_embedding = nn.Parameter(torch.randn(1, hidden_dim))
            self.virtual_node_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        self.readout_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x.float(), data.edge_index, data.edge_attr.float(), data.batch
        )

        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        if self.use_virtual_node:
            num_graphs = batch.max().item() + 1
            vn_expand = self.virtual_node_embedding.expand(num_graphs, -1)

        for i in range(self.num_layers):
            if self.use_virtual_node:
                h = h + vn_expand[batch]

            h_res = self.res_linears[i](h)
            h_new = self.convs[i](h, edge_index, edge_attr=e)
            h = F.gelu(self.batch_norms[i](h_new)) + h_res
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.use_virtual_node and i < self.num_layers - 1:
                pooled_nodes = global_mean_pool(h, batch)
                vn_expand = vn_expand + pooled_nodes
                vn_expand = self.virtual_node_mlp(vn_expand)

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        h_combined = torch.cat([h_mean, h_max, h_sum], dim=-1)

        return self.readout_proj(h_combined)


# ============================================================================
# Component 2: Tabular Physics Feature Encoder
# ============================================================================

class TabularEncoder(nn.Module):
    """
    Encodes 24 physics-informed tabular features into a dense latent representation.
    """
    def __init__(self, input_dim=24, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Component 3: 4-Head Bidirectional Cross-Modal Attention Fusion
# ============================================================================

class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Attention between Tabular and Graph representations.
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Multi-Head Attention layers
        self.cross_attn_t2g = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_g2t = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.norm_t = nn.LayerNorm(d_model)
        self.norm_g = nn.LayerNorm(d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, table_feat, graph_feat):
        t_seq = table_feat.unsqueeze(1)
        g_seq = graph_feat.unsqueeze(1)

        t_attended, _ = self.cross_attn_t2g(query=t_seq, key=g_seq, value=g_seq)
        t_enriched = self.norm_t(table_feat + t_attended.squeeze(1))

        g_attended, _ = self.cross_attn_g2t(query=g_seq, key=t_seq, value=t_seq)
        g_enriched = self.norm_g(graph_feat + g_attended.squeeze(1))

        gate_input = torch.cat([t_enriched, g_enriched], dim=-1)
        gate_weight = self.gate(gate_input)

        fused = gate_weight * t_enriched + (1 - gate_weight) * g_enriched
        return self.proj(fused)


class ConcatFusion(nn.Module):
    """Simple concatenation fusion (baseline for ablation)."""
    def __init__(self, d_model=128, dropout=0.1, **kwargs):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, table_feat, graph_feat):
        return self.proj(torch.cat([table_feat, graph_feat], dim=-1))


class GatedFusion(nn.Module):
    """Gated fusion without cross-attention (simpler alternative)."""
    def __init__(self, d_model=128, dropout=0.1, **kwargs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, table_feat, graph_feat):
        gate_input = torch.cat([table_feat, graph_feat], dim=-1)
        g = self.gate(gate_input)
        fused = g * table_feat + (1 - g) * graph_feat
        return self.proj(fused)


# ============================================================================
# Component 4: Prediction Head
# ============================================================================

class PredictionHead(nn.Module):
    """
    Final regression head with optional MC-Dropout for uncertainty.
    init_bias removed: a fixed bias of 72.4 was collapsing all predictions
    toward the training mean, causing hallucinated output-range collapse.
    """
    def __init__(self, input_dim=128, hidden_dim=64, dropout=0.2, init_bias=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        # init_bias intentionally disabled — let PyTorch default-initialise
        # so the head can learn to predict the full rejection range freely.

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# Component 5: Physics-Constrained Loss
# ============================================================================

class PhysicsConstrainedLoss(nn.Module):
    """
    Composite loss = Huber + physics constraint penalty terms.
    Encodes domain knowledge about membrane separation directly into training.
    """
    def __init__(self, lambda_steric=1.0, lambda_bounds=0.1,
                 use_huber=True, huber_delta=5.0):
        super().__init__()
        # lambda_steric raised from 0.1 → 1.0: at 0.1 the steric penalty was
        # too weak to overcome the data loss and was effectively ignored.
        # lambda_bounds raised from 0.01 → 0.1: enforces [0,100] clamping harder.
        self.lambda_steric = lambda_steric
        self.lambda_bounds = lambda_bounds
        self.use_huber = use_huber
        if use_huber:
            self.base_loss = nn.HuberLoss(delta=huber_delta)
        else:
            self.base_loss = nn.MSELoss()

    def forward(self, pred, target, steric_ratios=None):
        # Base data loss
        loss = self.base_loss(pred, target)

        # Physics constraint 1: Bounds [0, 100]
        if self.lambda_bounds > 0:
            bounds_penalty = (
                torch.mean(F.relu(-pred) ** 2) +
                torch.mean(F.relu(pred - 100.0) ** 2)
            )
            loss = loss + self.lambda_bounds * bounds_penalty

        # Physics constraint 2: Steric exclusion
        if self.lambda_steric > 0 and steric_ratios is not None:
            steric_mask = steric_ratios >= 1.0
            if steric_mask.any():
                steric_penalty = torch.mean(F.relu(85.0 - pred[steric_mask]) ** 2)
                loss = loss + self.lambda_steric * steric_penalty

        return loss


# ============================================================================
# THE COMPLETE CHAMPION MODEL: PhysiChemNet (PhysiChem-GT)
# ============================================================================

class PhysiChemNet(nn.Module):
    """
    PhysiChemNet: Complete end-to-end multimodal model for membrane rejection prediction.
    
    Combines:
        Tabular Features (24-D) -> TabularEncoder -> 128-D
        Molecular Graph         -> GINEConv       -> 128-D
                                        |
                            CrossModalAttentionFusion (4-Head)
                                        |
                                PredictionHead -> Rejection %
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Graph encoder
        gnn_type = config.get('gnn_type', 'gine')
        if gnn_type == 'gatv2':
            self.graph_encoder = GATv2MolecularEncoder(
                node_dim=config.get('node_dim', 9),
                edge_dim=config.get('edge_dim', 3),
                hidden_dim=config.get('hidden_dim', 128),
                num_layers=config.get('gnn_layers', 2),
                heads=config.get('gnn_heads', 4),
                dropout=config.get('gnn_dropout', 0.2),
                use_virtual_node=config.get('use_virtual_node', True)
            )
        else:  # gine (Global Champion)
            self.graph_encoder = GINEMolecularEncoder(
                node_dim=config.get('node_dim', 9),
                edge_dim=config.get('edge_dim', 3),
                hidden_dim=config.get('hidden_dim', 128),
                num_layers=config.get('gnn_layers', 2),
                dropout=config.get('gnn_dropout', 0.2),
                use_virtual_node=config.get('use_virtual_node', True)
            )

        # Tabular encoder (24-D input)
        self.table_encoder = TabularEncoder(
            input_dim=config.get('table_dim', 24),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('table_layers', 2),
            dropout=config.get('table_dropout', 0.2)
        )

        # Fusion
        fusion_type = config.get('fusion_type', 'cross_attention')
        hidden_dim = config.get('hidden_dim', 128)
        fusion_dropout = config.get('fusion_dropout', 0.1)
        if fusion_type == 'cross_attention':
            self.fusion = CrossModalAttentionFusion(
                d_model=hidden_dim,
                nhead=config.get('fusion_heads', 4),
                dropout=fusion_dropout
            )
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(d_model=hidden_dim, dropout=fusion_dropout)
        else:
            self.fusion = ConcatFusion(d_model=hidden_dim, dropout=fusion_dropout)

        # Prediction head
        self.pred_head = PredictionHead(
            input_dim=hidden_dim,
            hidden_dim=config.get('pred_hidden', 64),
            dropout=config.get('pred_dropout', 0.2)
        )

    def forward(self, table_data, graph_data):
        table_embed = self.table_encoder(table_data)
        graph_embed = self.graph_encoder(graph_data)
        fused = self.fusion(table_embed, graph_embed)
        pred = self.pred_head(fused)
        return pred

    def predict_with_uncertainty(self, table_data, graph_data, n_samples=30):
        """MC-Dropout uncertainty estimation with eval BatchNorm."""
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(table_data, graph_data)
                predictions.append(pred)
        
        self.eval()
        preds = torch.stack(predictions, dim=0)
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)
        return mean_pred, std_pred

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
