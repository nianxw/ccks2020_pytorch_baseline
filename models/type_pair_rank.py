import torch
import torch.nn as nn
from torch.nn import functional as F


class TypePairRank(nn.Module):
    def __init__(self, sentence_encoder, config):
        super(TypePairRank, self).__init__()

        self.encoder = sentence_encoder
        self.left_fc = nn.Linear(config.hidden_size*2, 1)
        self.init_model_weights(self.left_fc)
        self.right_fc = nn.Linear(config.hidden_size*2, 1)
        self.init_model_weights(self.right_fc)
        self.type_fc = nn.Linear(config.hidden_size, 24)
        self.init_model_weights(self.type_fc)

    def init_model_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self,
                query_input_ids, query_masks, query_sentence_ids, qury_position_ids,
                left_input_ids, left_masks, left_sentence_ids, left_position_ids,
                right_input_ids, right_masks, right_sentence_ids, right_position_ids
    ):
        query_cls = self.encoder(query_input_ids, query_masks, query_sentence_ids, qury_position_ids)[1] # [batch_size, 768]
        left_cls = self.encoder(left_input_ids, left_masks, left_sentence_ids, left_position_ids)[1]  # [batch_size, 768]
        right_cls = self.encoder(right_input_ids, right_masks, right_sentence_ids, right_position_ids)[1]  # [batch_size, 768]

        left_concat = torch.cat([query_cls, left_cls], dim=-1)
        right_concat = torch.cat([query_cls, right_cls], dim=-1)

        left_logits = self.left_fc(left_concat)
        right_logits = self.right_fc(right_concat)

        pair_probs = F.sigmoid(left_logits - right_logits)

        type_out = self.type_fc(query_cls)  # [batch_size, 24]

        return pair_probs, type_out, left_logits, right_logits