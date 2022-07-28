# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.text_field_embedders import TextFieldEmbedder

from tell.modules import (AdaptiveSoftmax, DynamicConv1dTBC, GehringLinear,
                          LightweightConv1dTBC, MultiHeadAttention)
from tell.modules.token_embedders import AdaptiveEmbedding
from tell.utils import eval_str_list, fill_with_neg_inf, softmax

from .decoder_base import Decoder, DecoderLayer
from torch import Tensor


@Decoder.register('dynamic_conv_decoder_pointer')
class DynamicConvDecoderPointer(Decoder):
    def __init__(self, vocab, embedder: TextFieldEmbedder, max_target_positions, dropout,
                 share_decoder_input_output_embed,
                 decoder_output_dim, decoder_conv_dim, decoder_glu,
                 decoder_conv_type, weight_softmax, decoder_attention_heads,
                 weight_dropout, relu_dropout, input_dropout,
                 decoder_normalize_before, attention_dropout, decoder_ffn_embed_dim,
                 decoder_kernel_size_list, adaptive_softmax_cutoff=None,
                 tie_adaptive_weights=False, adaptive_softmax_dropout=0,
                 tie_adaptive_proj=False, adaptive_softmax_factor=0, decoder_layers=6,
                 final_norm=True, padding_idx=0, namespace='target_tokens',
                 vocab_size=None, section_attn=False, article_embed_size=1024):
        super().__init__()
        self.vocab = vocab
        vocab_size = vocab_size or vocab.get_vocab_size(namespace)
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.share_input_output_embed = share_decoder_input_output_embed

        input_embed_dim = embedder.get_output_dim()
        embed_dim = input_embed_dim
        output_embed_dim = input_embed_dim

        padding_idx = padding_idx
        self.max_target_positions = max_target_positions

        self.embedder = embedder

        self.project_in_dim = GehringLinear(
            input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            DynamicConvPointerDecoderLayer(embed_dim, decoder_conv_dim, decoder_glu,
                                    decoder_conv_type, weight_softmax, decoder_attention_heads,
                                    weight_dropout, dropout, relu_dropout, input_dropout,
                                    decoder_normalize_before, attention_dropout, decoder_ffn_embed_dim,
                                    article_embed_size,
                                    kernel_size=decoder_kernel_size_list[i])
            for i in range(decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = GehringLinear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not tie_adaptive_weights else None

        if adaptive_softmax_cutoff is not None:
            adaptive_inputs = None
            if isinstance(embedder, AdaptiveEmbedding):
                adaptive_inputs = embedder
            elif hasattr(embedder, 'token_embedder_adaptive'):
                adaptive_inputs = embedder.token_embedder_adaptive
            elif tie_adaptive_weights:
                raise ValueError('Cannot locate adaptive_inputs.')
            self.adaptive_softmax = AdaptiveSoftmax(
                vocab_size,
                output_embed_dim,
                eval_str_list(adaptive_softmax_cutoff, type=int),
                dropout=adaptive_softmax_dropout,
                adaptive_inputs=adaptive_inputs,
                factor=adaptive_softmax_factor,
                tie_proj=tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(vocab_size, output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0,
                            std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = nn.LayerNorm(embed_dim)

        p_gen_input_size = input_embed_dim + output_embed_dim
        self.project_p_gens = GehringLinear(p_gen_input_size, 1)

    def forward(self, prev_target, contexts, article_ids, incremental_state=None,
                use_layers=None, **kwargs):

        # embed tokens and positions
        X = self.embedder(prev_target, incremental_state=incremental_state)

        inp_embed = X
        # if incremental_state is not None:
        #     X = X[:, -1:]

        if self.project_in_dim is not None:
            X = self.project_in_dim(X)

        X = F.dropout(X, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        X = X.transpose(0, 1)
        attn = None

        inner_states = [X]

        # decoder layers
        for i, layer in enumerate(self.layers):
            if not use_layers or i in use_layers:
                X, attn = layer(
                    X,
                    contexts,
                    incremental_state,
                )
                inner_states.append(X)

        if self.normalize:
            X = self.layer_norm(X)

        # T x B x C -> B x T x C
        X = X.transpose(0, 1)

        #print('-----------------X out shape', X.shape)
        if self.project_out_dim is not None:
            X = self.project_out_dim(X)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                X = F.linear(
                    X, self.embedder.token_embedder_bert.word_embeddings.weight)
            else:
                X = F.linear(X, self.embed_out)

        predictors = torch.cat((inp_embed, X), 2)
        p_gens = self.project_p_gens(predictors)
        #p_gens = torch.sigmoid(p_gens.float())
        assert attn is not None
        #print('-----------attn shape', attn.shape)
        X = self.output_layer(X, attn, article_ids, p_gens)


        #print('-----------------X out shape', X.shape)
        return X, {'attn': attn, 'inner_states': inner_states}

    def output_layer(
        self,
        features: Tensor,
        attn: Tensor,
        src_tokens: Tensor,
        p_gens: Tensor
    ) -> Tensor:
        """
        Project features to the vocabulary size and mix with the attention
        distributions.
        """
        # project back to size of vocabulary
        #if self.adaptive_softmax is None:
        #    logits = self.output_projection(features)
        #else:
        logits = features

        batch_size = logits.shape[0]
        output_length = logits.shape[1]
        assert src_tokens.shape[0] == batch_size
        src_length = src_tokens.shape[1]

        # The final output distribution will be a mixture of the normal output
        # distribution (softmax of logits) and attention weights.
        p_gens= torch.sigmoid(p_gens)
        gen_dists = self.get_normalized_probs_scriptable(
            logits, log_probs=False, sample=None
        )
        assert gen_dists.shape[2] == self.vocab_size

        gen_dists = torch.mul(gen_dists, p_gens)
        #padding_size = (batch_size, output_length, self.num_oov_types)
        #padding = gen_dists.new_zeros(padding_size)
        #gen_dists = torch.cat((gen_dists, padding), 2)
        #assert gen_dists.shape[2] == self.vocab_size

        # Scatter attention distributions to distributions over the extended
        # vocabulary in a tensor of shape [batch_size, output_length,
        # vocab_size]. Each attention weight will be written into a location
        # that is for other dimensions the same as in the index tensor, but for
        # the third dimension it's the value of the index tensor (the token ID).
        attn = attn[:, :, :-2]
        attn = torch.mul(attn.float(), 1 - p_gens)
        #attn_lprob = F.log_softmax(attn, dim=2)
        #attn = torch.mul(attn.float(), 1 - p_gens)
        #print('-------------ent tokens max:', torch.max(src_tokens))
        #print('-------------attn:', attn.shape)
        #print('-------------src:', src_tokens.shape)
        index = src_tokens[:, None, :]
        index = index.expand(batch_size, output_length, src_length)
        attn_dists_size = (batch_size, output_length, self.vocab_size)
        attn_dists = attn.new_zeros(attn_dists_size)

        #print('-------------ent tokens:', index.shape)
        #print('-------------attn:', attn.shape)
        assert index.shape[2] == attn.shape[2]
        assert index.shape[1] == attn.shape[1]
        assert index.shape[0] == attn.shape[0]
        attn_dists.scatter_add_(2, index.long(), attn)

        #p_gens_1 = F.logsigmoid(p_gens)
        #p_gens_2 = F.logsigmoid(-1.0*p_gens)
        #log_probs = torch.cat([attn_dists])
        #probs = gen_dists + attn_dists
        #print(torch.max(attn_dists, dim=2)[0])
        #print(torch.max(gen_dists, dim=2)[0])
        probs = gen_dists + attn_dists
        #print("-------probs check",torch.max(attn_dists, dim = 2)[0])
        #print("-------probs check",attn_dists[0][0][1])
        #print(torch.max(probs[0][0]))
        lprobs = probs.clamp(1e-10, 1.0).log()
        #print("-------logprob check",probs.sum(dim = 2))
        #print("-------logprob check",torch.exp(attn_dists).sum(dim = 2))
        # Final distributions, [batch_size, output_length, num_types].
        return lprobs

    def get_normalized_probs_scriptable(
        self,
        net_output: Tensor,
        log_probs: bool,
        sample = None
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output, target=target)
            return out.exp_() if not log_probs else out

        logits = net_output
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions
        # return min(self.max_target_positions, self.embedder.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # pylint: disable=access-member-before-definition
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(
                fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(
                self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            target = sample['target'] if sample else None
            out = self.adaptive_softmax.get_log_prob(
                net_output[0], target)
            return out.exp() if not log_probs else out

        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def filter_incremental_state(self, incremental_state, active_idx):
        if incremental_state is None:
            return
        for key in incremental_state:
            if 'DynamicConv1dTBC' in key:
                incremental_state[key] = incremental_state[key][:, active_idx]


@DecoderLayer.register('dynamic_conv_pointer')
class DynamicConvPointerDecoderLayer(DecoderLayer):
    def __init__(self, decoder_embed_dim, decoder_conv_dim, decoder_glu,
                 decoder_conv_type, weight_softmax, decoder_attention_heads,
                 weight_dropout, dropout, relu_dropout, input_dropout,
                 decoder_normalize_before, attention_dropout, decoder_ffn_embed_dim,
                 article_embed_size, kernel_size=0):
        super().__init__()
        self.embed_dim = decoder_embed_dim
        self.conv_dim = decoder_conv_dim
        if decoder_glu:
            self.linear1 = GehringLinear(self.embed_dim, 2*self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = GehringLinear(self.embed_dim, self.conv_dim)
            self.act = None
        if decoder_conv_type == 'lightweight':
            self.conv = LightweightConv1dTBC(self.conv_dim, kernel_size, padding_l=kernel_size-1,
                                             weight_softmax=weight_softmax,
                                             num_heads=decoder_attention_heads,
                                             weight_dropout=weight_dropout)
        elif decoder_conv_type == 'dynamic':
            self.conv = DynamicConv1dTBC(self.conv_dim, kernel_size, padding_l=kernel_size-1,
                                         weight_softmax=weight_softmax,
                                         num_heads=decoder_attention_heads,
                                         weight_dropout=weight_dropout)
        else:
            raise NotImplementedError
        self.linear2 = GehringLinear(self.conv_dim, self.embed_dim)

        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.input_dropout = input_dropout
        self.normalize_before = decoder_normalize_before

        self.conv_layer_norm = nn.LayerNorm(self.embed_dim)

        self.context_attns = nn.ModuleDict()
        self.context_attn_lns = nn.ModuleDict()
        C = 2048

        self.context_attns['image'] = MultiHeadAttention(
            self.embed_dim, decoder_attention_heads, kdim=C, vdim=C,
            dropout=attention_dropout)
        self.context_attn_lns['image'] = nn.LayerNorm(self.embed_dim)

        self.context_attns['article'] = MultiHeadAttention(
            self.embed_dim, decoder_attention_heads, kdim=article_embed_size, vdim=article_embed_size,
            dropout=attention_dropout)
        self.context_attn_lns['article'] = nn.LayerNorm(self.embed_dim)

        #entity_embed_size = 1024
        #self.context_attns['entity'] = MultiHeadAttention(
        #        self.embed_dim, decoder_attention_heads, kdim=entity_embed_size,
        #        vdim=entity_embed_size,dropout=attention_dropout)
        #self.context_attn_lns['entity'] = nn.LayerNorm(self.embed_dim)

        context_size = self.embed_dim * 2

        self.context_fc = GehringLinear(context_size, self.embed_dim)

        self.fc1 = GehringLinear(self.embed_dim, decoder_ffn_embed_dim)
        self.fc2 = GehringLinear(decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self, X, contexts, incremental_state):
        """
        Args:
            X (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = X
        X = self.maybe_layer_norm(self.conv_layer_norm, X, before=True)
        X = F.dropout(X, p=self.input_dropout, training=self.training)
        X = self.linear1(X)
        if self.act is not None:
            X = self.act(X)
        X = self.conv(X, incremental_state=incremental_state)
        X = self.linear2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = residual + X
        X = self.maybe_layer_norm(self.conv_layer_norm, X, after=True)

        attn = None
        X_contexts = []

        # Image attention
        residual = X
        X_image = self.maybe_layer_norm(
            self.context_attn_lns['image'], X, before=True)
        X_image, attn = self.context_attns['image'](
            query=X_image,
            key=contexts['image'],
            value=contexts['image'],
            key_padding_mask=contexts['image_mask'],
            incremental_state=None,
            static_kv=True,
            need_weights=(not self.training and self.need_attn))
        X_image = F.dropout(X_image, p=self.dropout, training=self.training)
        X_image = residual + X_image
        X_image = self.maybe_layer_norm(
            self.context_attn_lns['image'], X_image, after=True)
        X_contexts.append(X_image)

        # Article attention
        residual = X
        X_article = self.maybe_layer_norm(
            self.context_attn_lns['article'], X, before=True)
        X_article, attn = self.context_attns['article'](
            query=X_article,
            key=contexts['article'],
            value=contexts['article'],
            key_padding_mask=contexts['article_mask'],
            incremental_state=None,
            static_kv=True,
            need_weights=self.need_attn)
        X_article = F.dropout(X_article, p=self.dropout,
                              training=self.training)
        X_article = residual + X_article
        X_article = self.maybe_layer_norm(
            self.context_attn_lns['article'], X_article, after=True)

        X_contexts.append(X_article)

        X_context = torch.cat(X_contexts, dim=-1)
        X = self.context_fc(X_context)

        residual = X
        X = self.maybe_layer_norm(self.final_layer_norm, X, before=True)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, p=self.relu_dropout, training=self.training)
        X = self.fc2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = residual + X
        X = self.maybe_layer_norm(self.final_layer_norm, X, after=True)
        return X, attn

    def maybe_layer_norm(self, layer_norm, X, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(X)
        else:
            return X

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def extra_repr(self):
        return 'dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}'.format(
            self.dropout, self.relu_dropout, self.input_dropout, self.normalize_before)
