import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

class KVCache(NamedTuple):
    key: jnp.ndarray
    value: jnp.ndarray


@partial(jax.jit, static_argnames=['max_len'])
def compress_cache(cache, max_len):

     keys, values = cache.key, cache.value

     # Compute norms of the key embeddings
     norms = jnp.linalg.norm(keys, axis=-1)

     # Sort by norm and keep in memory only the keys with lowest L2 norm and the corresponding values
     indices = jnp.argsort(norms, axis=-1)[..., :max_len]
     # Sort back the indices to maintain the original order of the tokens  
     indices = jnp.sort(indices, axis=-1)

     # Extract keys and values relative to the identified indices to keep
     keys = jnp.take_along_axis(keys, indices[..., None], axis=2)
     values = jnp.take_along_axis(values, indices[... ,None], axis=2)

     return KVCache(key=keys, value=values)