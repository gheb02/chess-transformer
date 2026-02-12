import jax
import jax.numpy as jnp
import jax.random as jrand
import equinox as eqx
from functools import partial
import numpy as np
import optax
from typing import NamedTuple
from kv_cache import KVCache, compress_cache


def attention(q, k, v, mask=None):

    d = q.shape[-1]
    norm_dot_prod = jnp.einsum("bhqd, bhkd->bhqk", q, k)/jnp.sqrt(d)
    if mask is not None:
        norm_dot_prod += mask
    logits = jax.nn.softmax(norm_dot_prod, axis=-1)

    return jnp.einsum("bhqk, bhkd->bhqd", logits, v)


class MultiHeadAttention(eqx.Module):
    W_q: jnp.ndarray
    W_k: jnp.ndarray
    W_v: jnp.ndarray
    W_o: jnp.ndarray
    b_q: jnp.ndarray
    b_k: jnp.ndarray
    b_v: jnp.ndarray
    b_o: jnp.ndarray
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    causal: bool = eqx.field(static=True)
    cache: KVCache = eqx.field(static=False)

    def __init__(self, key, in_dim, out_dim, num_heads=1, use_bias=True, causal=True):

        if in_dim % num_heads != 0:
            raise ValueError("in_dim must be divisible by num_heads!")
        self.num_heads = num_heads
        self.head_dim = in_dim//num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.causal = causal
        self.use_bias = use_bias
        self.cache = None

        keyq, keyk, keyv, keyo = jrand.split(key, 4)

        # Initialize projection matrices using Xavier initialization
        std_q = jnp.sqrt(2.0/(in_dim+in_dim))
        std_o = jnp.sqrt(2.0/(in_dim+out_dim))

        self.W_q = jrand.normal(keyq, (in_dim, in_dim))*std_q
        self.W_k = jrand.normal(keyk, (in_dim, in_dim))*std_q
        self.W_v = jrand.normal(keyv, (in_dim, in_dim))*std_q
        self.W_o = jrand.normal(keyo, (in_dim, out_dim))*std_o

        # Initialize biases
        if self.use_bias:
            self.b_q = jnp.zeros((in_dim, ))
            self.b_k = jnp.zeros((in_dim, ))
            self.b_v = jnp.zeros((in_dim, ))
            self.b_o = jnp.zeros((out_dim, ))

    def init_cache(self, batch_size):

        # seq_len is initialized as 0, since no tokens have been processed
        empty = jnp.zeros((batch_size, self.num_heads, 0, self.head_dim))

        return eqx.tree_at(lambda m: m.cache, # selects which part of the model has to be changed (the cache)
                           self, KVCache(empty, empty))


    def __call__(self, x):
        assert self.cache is None

        batch, seq_len, _ = x.shape

        # Project using Query, Key and Value matrices
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        if self.use_bias:
            Q = Q + self.b_q
            K = K + self.b_k
            V = V + self.b_v

        Q = jnp.transpose(Q.reshape(batch, seq_len, self.num_heads, self.head_dim), (0, 2, 1, 3))
        K = jnp.transpose(K.reshape(batch, seq_len, self.num_heads, self.head_dim), (0, 2, 1, 3))
        V = jnp.transpose(V.reshape(batch, seq_len, self.num_heads, self.head_dim), (0, 2, 1, 3))

        if not self.causal:
            mask = None
        else:
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            mask = jnp.where(mask == 1, 0.0, -1e9).astype(Q.dtype)
            mask = mask[None, None, :, :]

        attn = attention(Q, K, V, mask)
        attn = jnp.transpose(attn, (0, 2, 1, 3))
        attn = attn.reshape(batch, seq_len, -1)

        out = attn @ self.W_o

        if self.use_bias:
            out = out + self.b_o

        return out

    def decode(self, x_t):
        assert self.cache is not None

        batch = x_t.shape[0]

        # Project using Query, Key and Value matrices
        Q = x_t @ self.W_q
        K = x_t @ self.W_k
        V = x_t @ self.W_v

        if self.use_bias:
            Q = Q + self.b_q
            K = K + self.b_k
            V = V + self.b_v

        Q = jnp.transpose(Q.reshape(batch, 1, self.num_heads, self.head_dim), (0, 2, 1, 3))
        K = jnp.transpose(K.reshape(batch, 1, self.num_heads, self.head_dim), (0, 2, 1, 3))
        V = jnp.transpose(V.reshape(batch, 1, self.num_heads, self.head_dim), (0, 2, 1, 3))


        new_cache = KVCache(key=jnp.concatenate([self.cache.key, K], axis=2),
                            value=jnp.concatenate([self.cache.value, V], axis=2))

        # Create a copy of the model, but with updated KV Cache
        new_self = eqx.tree_at(lambda m: m.cache, self, new_cache)

        attn = attention(Q, new_cache.key, new_cache.value, mask=None)
        attn = jnp.transpose(attn, (0, 2, 1, 3))
        attn = attn.reshape(batch, 1, -1)

        out = attn @ self.W_o

        if self.use_bias:
            out = out + self.b_o
        out = out[:, 0, :]
        return new_self, out
    

class TransformerBlock(eqx.Module):
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    attn: MultiHeadAttention
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    causal: bool = eqx.field(static=True)

    def __init__(self, key, size, mlp_size, heads=4, causal=True):
        self.causal = causal
        key1, key2, key3 = jrand.split(key, 3)
        self.layer_norm1 = eqx.nn.LayerNorm(size)
        self.attn = MultiHeadAttention(key1, size, size, heads, causal)
        self.lin1 = eqx.nn.Linear(size, mlp_size, key=key2)
        self.lin2 = eqx.nn.Linear(mlp_size, size, key=key3)
        self.layer_norm2 = eqx.nn.LayerNorm(size)

    def init_cache(self, batch_size):
        new_attn = self.attn.init_cache(batch_size)
        return eqx.tree_at(lambda m: m.attn, self, new_attn)


    def __call__(self, x):
        # Attention Path
        skip = x
        x = jax.vmap(jax.vmap(self.layer_norm1))(x)
        x = self.attn(x)
        x = x + skip

        # MLP Path
        skip = x
        x = jax.vmap(jax.vmap(self.layer_norm2))(x)
        x = jax.vmap(jax.vmap(self.lin1))(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(jax.vmap(self.lin2))(x)
        x = x + skip
        return x

    def decode(self, x_t):
        assert self.causal
        # Attention Path
        skip = x_t
        x_t = jax.vmap(self.layer_norm1)(x_t)
        new_attn, attn_out = self.attn.decode(x_t)
        x_t = attn_out + skip

        # MLP Path
        skip = x_t
        x_t = jax.vmap(self.layer_norm2)(x_t)
        x_t = jax.vmap(self.lin1)(x_t)
        x_t = jax.nn.gelu(x_t)
        x_t = jax.vmap(self.lin2)(x_t)
        x_t = x_t + skip

        new_self = eqx.tree_at(lambda m: m.attn, self, new_attn)
        return new_self, x_t
    