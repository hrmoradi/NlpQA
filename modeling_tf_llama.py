import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# from ...activations_tf import get_tf_activation
from transformers.activations_tf import get_tf_activation

# from ...modeling_tf_outputs import (
from transformers.modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPast,
    TFBaseModelOutputWithPooling,
)
# from ...modeling_tf_utils import (
from transformers.modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFSequenceSummary,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)

# from ...tf_utils import shape_list, stable_softmax
from transformers.tf_utils import shape_list, stable_softmax

# from ...utils import (
from transformers.utils import (
    DUMMY_INPUTS,
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    logging,
)

# from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
# from ...modeling_utils import PreTrainedModel
# from .configuration_llama import LlamaConfig

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig



LARGE_NEGATIVE = -1e8

def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz = input_ids_shape[0]
    tgt_len = input_ids_shape[1]
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    mask_cond = tf.range(shape_list(mask)[-1])

    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)

    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE

LLAMA_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.
    <Tip>
    TensorFlow models and layers in `transformers` accept two formats as input:
    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.
    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:
    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!
    </Tip>
    Args:
        config ([`LlamaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """


    
LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Llama Model transformer outputing raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)


class TFLlamaRMSNorm(tf.keras.layers.Layer):
    def __init__(self, eps=1e-6, **kwargs):
        """
        TFLlamaRMSNorm is equivalent to TFT5LayerNorm
        """
        super().__init__(**kwargs)
        self.variance_epsilon = eps
        
    def build(self, input_shape):
        """Build shared word embedding layer """
        self.weight = self.add_weight("weight", shape=(input_shape[-1],), initializer="ones")
        super().build(input_shape)
        
    def call(self, hidden_states):
        variance = tf.math.reduce_mean(tf.math.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * tf.math.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
        
class TFLlamaRotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, **kwargs):
        super().__init__(**kwargs)
        inv_freq = 1.0 / (base ** (tf.range(0, dim, 2, dtype=tf.float32) / dim))
        #self.inv_freq = tf.Variable(inv_freq, trainable=False, name="inv_freq")
        self.inv_freq = tf.constant(inv_freq, dtype=inv_freq.dtype, name="self_attn.rotary_emb.inv_freq")
        
        self.max_seq_len_cached = max_position_embeddings
        t = tf.range(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = tf.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = tf.concat([freqs, freqs], axis=-1)
        #self.cos_cached = tf.Variable(tf.math.cos(emb)[None, None, :, :], trainable=False, name="cos_cached")
        #self.sin_cached = tf.Variable(tf.math.sin(emb)[None, None, :, :], trainable=False, name="sin_cached")
        self.cos_cached = tf.constant(tf.math.cos(emb)[None, None, :, :], name="cos_cached")
        self.sin_cached = tf.constant(tf.math.sin(emb)[None, None, :, :], name="sin_cached")

    def call(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = tf.range(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
            freqs = tf.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = tf.concat([freqs, freqs], axis=-1)
            #self.cos_cached = tf.Variable(tf.math.cos(emb)[None, None, :, :], trainable=False, name="cos_cached")
            #self.sin_cached = tf.Variable(tf.math.sin(emb)[None, None, :, :], trainable=False, name="sin_cached")
            self.cos_cached = tf.constant(tf.math.cos(emb)[None, None, :, :], name="cos_cached")
            self.sin_cached = tf.constant(tf.math.sin(emb)[None, None, :, :], name="sin_cached")

        return self.sin_cached[:, :, :seq_len, :], self.cos_cached[:, :, :seq_len, :]

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return tf.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = tf.squeeze(cos, axis=[0,1]) # [seq_len, dim]
    sin = tf.squeeze(sin, axis=[0,1]) # [seq_len, dim]
    
    cos = tf.expand_dims(tf.gather(cos, position_ids), axis=1)  # [bs, 1, seq_len, dim]
    sin = tf.expand_dims(tf.gather(sin, position_ids), axis=1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class TFLlamaMLP(tf.keras.layers.Layer):

    def __init__(self, hidden_size, intermediate_size, hidden_act, **kwargs):
        super().__init__( **kwargs)
        self.gate_proj = tf.keras.layers.Dense(intermediate_size, use_bias=False, name="gate_proj")
        self.down_proj = tf.keras.layers.Dense(hidden_size, use_bias=False, name="down_proj")
        self.up_proj = tf.keras.layers.Dense(intermediate_size, use_bias=False, name="up_proj")
        self.act_fn = get_tf_activation(hidden_act)
        
    def call(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TFLlamaAttention(tf.keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""
    
    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = tf.keras.layers.Dense(self.num_heads * self.head_dim, use_bias=False, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(self.num_heads * self.head_dim, use_bias=False, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.num_heads * self.head_dim, use_bias=False, name="v_proj")
        self.o_proj = tf.keras.layers.Dense(self.hidden_size, use_bias=False, name="o_proj")
        self.rotary_emb = TFLlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, name="rotary_emb")
        
        
    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[tf.Tensor]]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    
        bsz, q_len, _ = shape_list(hidden_states)
        query_states = tf.transpose(
            tf.reshape(self.q_proj(hidden_states), (bsz, q_len, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        key_states = tf.transpose(
            tf.reshape(self.k_proj(hidden_states), (bsz, q_len, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        value_states = tf.transpose(
            tf.reshape(self.v_proj(hidden_states), (bsz, q_len, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        
        kv_seq_len = shape_list(key_states)[-2]
        if past_key_value is not None:
            print(past_key_value)
            kv_seq_len += shape_list(past_key_value[0])[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = tf.keras.layers.concatenate([past_key_value[0], key_states], axis=2)
            value_states = tf.keras.layers.concatenate([past_key_value[1], value_states], axis=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        attn_weights = tf.matmul(query_states, tf.transpose(key_states, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
        
        tf.debugging.assert_equal(
            shape_list(attn_weights),
            [bsz , self.num_heads, q_len, kv_seq_len],
            message=(
                f"Attention weights should be of size {(bsz , self.num_heads, q_len, kv_seq_len)}, but is"
                f" {shape_list(attn_weights)}"
            ),
        )
        
        if attention_mask is not None:
            tf.debugging.assert_equal(
                shape_list(attention_mask),
                [bsz, 1, q_len, kv_seq_len],
                message=(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is"
                    f" {shape_list(attention_mask)}"
                ),
            )
            attn_weights = attn_weights + tf.cast(attention_mask, dtype=attn_weights.dtype)
            attn_weights = tf.math.maximum(attn_weights, tf.constant(tf.float32.min, dtype=attn_weights.dtype))
            
        # upcast attention to fp32
        attn_weights = tf.cast(tf.nn.softmax(tf.cast(attn_weights, tf.float32), axis=-1), query_states.dtype)
        attn_output = tf.matmul(attn_weights, value_states)
            
        tf.debugging.assert_equal(
            shape_list(attn_output),
            [bsz , self.num_heads, q_len, self.head_dim],
            message=(
                f"Attention weights should be of size {(bsz , self.num_heads, q_len, self.head_dim)}, but is"
                f" {shape_list(attn_output)}"
            ),
        )
        
        attn_output = tf.transpose(attn_output, (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (bsz, q_len, self.hidden_size))
        
        attn_output = self.o_proj(attn_output)
    
        if not output_attentions:
            attn_weights = None
            
        return attn_output, attn_weights, past_key_value
        
class TFLlamaDecoderLayer(tf.keras.layers.Layer):
    config_class = LlamaConfig
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFLlamaDecoderLayer`]
    Args:
        config: LlamaConfig
        embed_tokens: output embedding
    """

    def __init__(self, config: LlamaConfig, embed_tokens: Optional[tf.keras.layers.Embedding] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = TFLlamaAttention(config=config, name="self_attn")
        self.mlp = TFLlamaMLP(hidden_size=self.hidden_size,
                              intermediate_size=config.intermediate_size,
                              hidden_act=config.hidden_act,
                              name="mlp",
                              )
        self.input_layernorm = TFLlamaRMSNorm(eps=config.rms_norm_eps, name="input_layernorm")
        self.post_attention_layernorm = TFLlamaRMSNorm(eps=config.rms_norm_eps, name="post_attention_layernorm")
        
    def call(
        self,
        hidden_states: Optional[Union[np.ndarray, tf.Tensor]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        past_key_value: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, Tuple[Tuple[tf.Tensor]]]:
    
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return (
            hidden_states,
            self_attn_weights,
            present_key_value,
        )


class TFLlamaEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.embedding_size = config.hidden_size
        self.initializer_range = config.initializer_range
       
    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.config.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        super().build(input_shape)
        
        
    def call(
        self,
        input_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.
        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)

        if input_ids is not None:
            # Note: tf.gather, on which the embedding layer is based, won't check positive out of bound
            # indices on GPU, returning zeros instead. This is a dangerous silent behavior.
            tf.debugging.assert_less(
                input_ids,
                tf.cast(self.config.vocab_size, dtype=input_ids.dtype),
                message=(
                    "input_ids must be smaller than the embedding layer's input dimension (got"
                    f" {tf.math.reduce_max(input_ids)} >= {self.config.vocab_size})"
                ),
            )
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        final_embeddings = inputs_embeds

        return final_embeddings



@keras_serializable        
class TFLlamaMainLayer(tf.keras.layers.Layer):
    config_class = LlamaConfig
    def __init__(self, config: LlamaConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        
        #self.embed_tokens = TFLlamaEmbeddings(config, name="embeddings")
        self.embed_tokens = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.hidden_size,
            embeddings_initializer = get_initializer(self.config.initializer_range),
            name="embed_tokens")

        
        self.layers = [TFLlamaDecoderLayer(config, name=f"layers.{i}") for i in range(config.num_hidden_layers)]
        self.norm = TFLlamaRMSNorm(eps=config.rms_norm_eps, name="norm")


    #def get_input_embeddings(self) -> tf.keras.layers.Layer:
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
        #self.embeddings.weight = value
        #self.embeddings.vocab_size = shape_list(value)[0]
        
    @unpack_inputs    
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        past_key_values: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = shape_list(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = shape_list(inputs_embeds)
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        
        seq_length_with_past = seq_length
        past_key_values_length = shape_list(past_key_values[0][0])[2] if past_key_values is not None else 0

        if past_key_values is not None:
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(past_key_values_length, seq_length + past_key_values_length), axis=0)
        else:
            position_ids = tf.reshape(position_ids, [-1, seq_length])
            position_ids = tf.cast(position_ids, tf.int64)

        
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = tf.ones(shape=(batch_size, seq_length_with_past), dtype=tf.bool, name="attention_mask"
    )
            
            
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        if (batch_size, seq_length)[-1] > 1:
            combined_attention_mask = _make_causal_mask((batch_size, seq_length), past_key_values_length=past_key_values_length)
        else:
            combined_attention_mask = _expand_mask(
                tf.ones((input_shape[0], (batch_size, seq_length)[1] + past_key_values_length)), tgt_len=(batch_size, seq_length)[-1]
            )

        if attention_mask is not None:
            combined_attention_mask = combined_attention_mask + _expand_mask(attention_mask, tgt_len=(batch_size, seq_length)[-1])    

        hidden_states = inputs_embeds
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        present_key_values = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
        
            hidden_states, layer_self_attn, present_key_value = decoder_layer(
                hidden_states,
                attention_mask=combined_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            if use_cache:
                present_key_values += (present_key_value,)

            
            if output_attentions:
                all_self_attns += (layer_self_attn,)
        
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        next_cache = present_key_value if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return TFBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)

class TFLlamaPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LlamaConfig
    base_model_prefix = "model"

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
                "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)

        
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class TFLlamaModel(TFLlamaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["inv_freq"]
    
    def __init__(self, config: LlamaConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.model = TFLlamaMainLayer(config, name="model")
        
    @unpack_inputs
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        past_key_values: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache = use_cache,
            return_dict=return_dict,
            training=training,
        )

        return outputs
    
    
    def serving_output(self, output: TFBaseModelOutput) -> TFBaseModelOutput:
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None

        return TFBaseModelOutput(last_hidden_state=output.last_hidden_state, hidden_states=hs, attentions=attns)
