import torch.nn.functional as F
import etw_pytorch_utils as pt_utils
from typing import Optional, List
from torch import nn, Tensor
from multihead_attention import MultiheadAttention


def _get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=3, num_pos_feats=256):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        # xyz : BxNx3
        xyz = xyz.transpose(1, 2).contiguous()
        # Bx3xN
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


num_layers = 1
d_model =9
multihead_attn = MultiheadAttention(
    feature_dim=d_model, n_head=1, key_feature_dim=128)
fea_layer = (pt_utils.Seq(1024)
             .conv1d(1024, bn=True)
             .conv1d(1024, activation=None))




class TransformerEncoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model, self_posembed=None):
        super().__init__()
        self.self_attn = multihead_attn
        # Implementation of Feedforward model
        self.FFN = FFN
        self.norm = nn.InstanceNorm1d(d_model)
        self.self_posembed = self_posembed

        self.dropout = nn.Dropout(0.1)

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, query_pos=None):
        # BxNxC -> BxCxN -> NxBxC
        if self.self_posembed is not None and query_pos is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
            # query_pos_embed=None
        else:
            query_pos_embed = None
        query = key = value = self.with_pos_embed(src, query_pos_embed)

        # self-attention
        # NxBxC
        src2 = self.self_attn(query=query, key=key, value=value)
        src = src + src2

        # NxBxC -> BxCxN -> NxBxC
        src = self.norm(src.permute(1, 2, 0)).permute(2, 0, 1)
        return F.relu(src)
        # return src


class TransformerEncoder(nn.Module):
    def __init__(self, multihead_attn, FFN,
                 d_model=512,
                 num_encoder_layers=6,
                 activation="relu",
                 self_posembed=None):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            multihead_attn, FFN, d_model, self_posembed=self_posembed)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)

    def forward(self, src, query_pos=None):
        num_imgs, batch, dim = src.shape
        output = src

        for layer in self.layers:
            output = layer(output, query_pos=query_pos)

        # import pdb; pdb.set_trace()
        # [L,B,D] -> [B,D,L]
        # output_feat = output.reshape(num_imgs, batch, dim)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model, self_posembed=None):
        super().__init__()
        self.self_attn = multihead_attn
        self.cross_attn = MultiheadAttention(
            feature_dim=d_model,
            n_head=1, key_feature_dim=128)

        self.FFN = FFN
        self.norm1 = nn.InstanceNorm1d(d_model)
        self.norm2 = nn.InstanceNorm1d(d_model)
        self.self_posembed = self_posembed

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, tgt, memory, query_pos=None):
        if self.self_posembed is not None and query_pos is not None:
            # query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
            query_pos_embed = None

        else:
            query_pos_embed = None
        # NxBxC

        # self-attention
        query = key = value = self.with_pos_embed(tgt, query_pos_embed)

        tgt2 = self.self_attn(query=query, key=key, value=value)
        # tgt2 = self.dropout1(tgt2)
        tgt = tgt + tgt2
        # tgt = F.relu(tgt)
        # tgt = self.instance_norm(tgt, input_shape)
        # NxBxC
        # tgt = self.norm(tgt)
        tgt = self.norm1(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        tgt = F.relu(tgt)

        mask = self.cross_attn(
            query=tgt, key=memory, value=memory)
        # mask = self.dropout2(mask)
        tgt2 = tgt + mask
        tgt2 = self.norm2(tgt2.permute(1, 2, 0)).permute(2, 0, 1)

        tgt2 = F.relu(tgt2)
        return tgt2


class TransformerDecoder(nn.Module):
    def __init__(self, multihead_attn, FFN,
                 d_model=512,
                 num_decoder_layers=6,
                 activation="relu",
                 self_posembed=None):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            multihead_attn, FFN, d_model, self_posembed=self_posembed)
        self.layers = _get_clones(decoder_layer, num_decoder_layers)

    def forward(self, tgt, memory, query_pos=None):
        assert tgt.dim() == 3, 'Expect 3 dimensional inputs'
        tgt_shape = tgt.shape
        num_imgs, batch, dim = tgt.shape

        output = tgt
        for layer in self.layers:
            output = layer(output, memory, query_pos=query_pos)
        return output


encoder_pos_embed = PositionEmbeddingLearned(3, d_model)
decoder_pos_embed = PositionEmbeddingLearned(3, d_model)

encoder = TransformerEncoder(
    multihead_attn=multihead_attn, FFN=None,
    d_model=d_model, num_encoder_layers=num_layers,
    self_posembed=encoder_pos_embed)
    self_posembed=None)





