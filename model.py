import jax
import jax.numpy as jnp
import jax.random as jrand
import equinox as eqx
from functools import partial
import numpy as np
import optax
from typing import NamedTuple
from kv_cache import KVCache, compress_cache
from layers import attention, MultiHeadAttention, TransformerBlock
    

class DecoderTransformer(eqx.Module):
    emb: eqx.nn.Embedding
    positional_emb: eqx.nn.Embedding
    transf_block1: TransformerBlock
    transf_block2: TransformerBlock
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    drop: eqx.nn.Dropout
    max_len: int = eqx.field(static=True)

    def __init__(self, key, vocab_size, max_len, heads=4, causal=True, embedding_size=128, mlp_size=256, p_drop=0.3):

        subkeys = jrand.split(key, 6)
        self.emb = eqx.nn.Embedding(vocab_size, embedding_size, key=subkeys[0])
        self.positional_emb = eqx.nn.Embedding(max_len, embedding_size, key=subkeys[1])
        self.transf_block1 = TransformerBlock(subkeys[2], embedding_size, mlp_size, heads, causal)
        self.transf_block2 = TransformerBlock(subkeys[3], embedding_size, mlp_size, heads, causal)
        self.layer_norm1 = eqx.nn.LayerNorm(embedding_size)
        self.layer_norm2 = eqx.nn.LayerNorm(embedding_size)
        self.lin1 = eqx.nn.Linear(embedding_size, mlp_size, key=subkeys[4])
        self.lin2 = eqx.nn.Linear(mlp_size, vocab_size, key=subkeys[5])
        self.max_len = max_len
        self.drop = eqx.nn.Dropout(p_drop)

    def init_cache(self, batch_size):
        # Initialize the cache for the transformer blocks
        new_block1 = self.transf_block1.init_cache(batch_size)
        new_block2 = self.transf_block2.init_cache(batch_size)

        # Update both blocks within the model structure
        return eqx.tree_at(
            lambda m: (m.transf_block1, m.transf_block2),
            self,
            (new_block1, new_block2)
        )


    def __call__(self, x, *, key=None, training: bool = False, start_pos: int = 0):

        batch_size, seq_len = x.shape
        if seq_len > self.max_len:
            raise ValueError("Sequence length exceeds maximum positional encoding length")

        x = jax.vmap(jax.vmap(self.emb))(x)
        positions = jnp.arange(start_pos, start_pos+seq_len)
        pos = jax.vmap(self.positional_emb)(positions)
        x = x + pos

        x = self.transf_block1(x)

        if training:
            if key is None:
                raise ValueError("When training=True a key must be provided to make Dropout work")
            key1, key2, key3 = jrand.split(key, 3)
            x = self.drop(x, key=key1)

        x = jax.vmap(jax.vmap(self.layer_norm1))(x)
        x = self.transf_block2(x)
        if training:
            x = self.drop(x, key=key2)

        x = jax.vmap(jax.vmap(self.layer_norm2))(x)
        x = jax.vmap(jax.vmap(self.lin1))(x)
        x = jax.nn.gelu(x)
        if training:
            x = self.drop(x, key=key3)
        x = jax.vmap(jax.vmap(self.lin2))(x)

        return x

    def decode(self, x_t, position):
        assert self.transf_block1.causal
        assert self.transf_block2.causal

        x_t = jax.vmap(self.emb)(x_t)
        if jnp.ndim(position) == 0:
            # Scalar position
            pos = self.positional_emb(position)  # (embedding_size,)
            pos = pos[None, :]  # (1, embedding_size) for broadcasting
        else:
            # Different position per batch item
            pos = jax.vmap(self.positional_emb)(position)
        x_t = x_t + pos

        new_block1, x_t = self.transf_block1.decode(x_t)

        x_t = jax.vmap(self.layer_norm1)(x_t)

        new_block2, x_t = self.transf_block2.decode(x_t)

        x_t = jax.vmap(self.layer_norm2)(x_t)

        x_t = jax.vmap(self.lin1)(x_t)
        x_t = jax.nn.gelu(x_t)
        x_t = jax.vmap(self.lin2)(x_t)
        logits = x_t

        new_self = eqx.tree_at(lambda m: (m.transf_block1, m.transf_block2), self, (new_block1, new_block2))

        return new_self, logits