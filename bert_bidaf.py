"""
BERT Base + BiDAF

Author:
    Timur Reziapov (treziapov@stanford.edu)
"""

import layers
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from transformers import *
from util import masked_softmax

class BertBidaf(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, bert_base_model, hidden_size, drop_prob=0.):
        super(BertBidaf, self).__init__()
        
        BERT_BASE_HIDDEN_SIZE = 768

        self.bert = bert_base_model

        self.enc = layers.RNNEncoder(input_size=BERT_BASE_HIDDEN_SIZE,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        # self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
        #                                  drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=2 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = BertBidafOutput(hidden_size=hidden_size,
                                   drop_prob=drop_prob)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        # Our usage of BERT embeddings
        c_emb = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # sequence_output = outputs[0]

        c_mask = torch.zeros_like(input_ids) != input_ids
        c_len = c_mask.sum(-1)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 1 * hidden_size)

        # att = self.att(c_enc, c_mask)    # (batch_size, c_len, 2 * hidden_size)

        mod = self.mod(c_enc, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class BertBidafAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BertBidafAttention, self).__init__()
        self.drop_prob = drop_prob
        self.att_projection = torch.nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, c, c_mask):
        batch_size, c_len, _ = c.size()

        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)



        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

class BertBidafOutput(nn.Module):

    def __init__(self, hidden_size, drop_prob):
        super(BertBidafOutput, self).__init__()
        
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = layers.RNNEncoder(input_size=2 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

class BertBidafForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config):
        super(BertBidafForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        BERT_BASE_HIDDEN_SIZE = 768
        self.hidden_size = 100
        self.drop_prob = 0.2

        # self.bert = BertModel(config)
        self.bert = BertModel.from_pretrained('bert-base-cased')
        
        self.enc = layers.RNNEncoder(input_size=BERT_BASE_HIDDEN_SIZE,
                                     hidden_size=self.hidden_size,
                                     num_layers=1,
                                     drop_prob=self.drop_prob)

        self.mod = layers.RNNEncoder(input_size=2 * self.hidden_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=2,
                                     drop_prob=self.drop_prob)

        self.out = BertBidafOutput(hidden_size=self.hidden_size,
                                   drop_prob=self.drop_prob)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import BertTokenizer, BertForQuestionAnswering
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
        assert answer == "a nice puppet"
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        c_emb = outputs[0]

        c_mask = torch.zeros_like(input_ids) != input_ids
        c_len = c_mask.sum(-1)
        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 1 * hidden_size)
        mod = self.mod(c_enc, c_len)        # (batch_size, c_len, 2 * hidden_size)
        start_logits, end_logits = self.out(mod, c_mask)  # 2 tensors, each (batch_size, c_len)
        
        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')
    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
    print(f"Input ids: {input_ids}")
    
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    print(f"Last hidden state shape: {last_hidden_states.shape}")
    