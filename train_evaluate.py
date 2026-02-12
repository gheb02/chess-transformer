import jax
import jax.numpy as jnp
import jax.random as jrand
import equinox as eqx
from functools import partial
import numpy as np
import optax
from typing import NamedTuple
import tensorflow as tf
import tensorflow_datasets as tfds
import time
from kv_cache import KVCache, compress_cache
from layers import attention, MultiHeadAttention, TransformerBlock

def topk_accuracy(logits, targets, k, pad_token=0):
    vocab = logits.shape[-1]

    logits_flat = logits.reshape(-1, vocab)
    targets_flat = targets.reshape(-1)

    mask = targets_flat != pad_token

    # Indices of top-k predictions
    topk = jax.lax.top_k(logits_flat, k)[1]

    # Check if target appears in top-k
    correct = (topk == targets_flat[:, None]).any(axis=-1)
    correct = correct & mask
    denom = jnp.sum(mask) + 1e-9

    return jnp.sum(correct) / denom

def perplexity(loss):
    return jnp.exp(loss)

def compute_metrics(model, x, y, key=None, training=False):
    logits = model(x, key=key, training=training)
    vocab = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab)
    targets_flat = y.reshape(-1)

    loss_vals = optax.softmax_cross_entropy_with_integer_labels(
        logits_flat, targets_flat
    )

    mask = targets_flat != 0
    denom = jnp.sum(mask) + 1e-9
    loss = jnp.sum(loss_vals * mask) / denom

    # Metrics to track
    top1 = topk_accuracy(logits, y, k=1)
    top3 = topk_accuracy(logits, y, k=3)
    top5 = topk_accuracy(logits, y, k=5)
    ppl = jnp.exp(loss)

    return loss, (ppl, top1, top3, top5)

@eqx.filter_jit
def train_step(model, opt_state, x, y, *, key, optimizer):
    key, subkey = jrand.split(key)

    (loss, metrics), grads = eqx.filter_value_and_grad(
        compute_metrics, has_aux=True
    )(model, x, y, subkey, training=True)

    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss, metrics, key


@eqx.filter_jit
def eval_step(model, x, y):
    return compute_metrics(model, x, y, training=False)


def train_model(
    model,
    train_loader,
    val_loader,
    key = None,
    patience=5,
    epochs=50,
):
    # Initialize state using a global optimizer
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    history = {
        "train_loss": [], "val_loss": [], "train_ppl": [],
        "val_ppl": [], "train_top1": [], "val_top1": [],
        "val_top3": [], "val_top5": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model = model


    for epoch in range(epochs):
        train_iter = tfds.as_numpy(train_loader)
        val_iter = tfds.as_numpy(val_loader)
        # Training accumulation
        tr_loss = tr_ppl = tr_t1 = jnp.array(0.0)
        tr_batches = 0


        for x_batch, y_batch in train_iter:
            key, subkey = jrand.split(key)
            x = jnp.asarray(x_batch)
            y = jnp.asarray(y_batch)

            model, opt_state, loss, metrics, key = train_step(model, opt_state, x, y,
                                                              key=subkey, optimizer=optimizer)

            # Unpack auxiliary metrics from compute_metrics
            ppl, t1, t3, t5 = metrics

            tr_loss += loss
            tr_ppl += ppl
            tr_t1 += t1
            tr_batches += 1

        tr_loss = float(tr_loss / tr_batches)
        tr_ppl  = float(tr_ppl / tr_batches)
        tr_t1   = float(tr_t1 / tr_batches)


        # Validation accumulation
        val_loss = val_ppl = val_t1 = val_t3 = val_t5 = jnp.array(0.0)
        val_batches = 0
        for x_val, y_val in val_iter:
            x_val = jnp.asarray(x_val)
            y_val = jnp.asarray(y_val)
            loss, metrics = eval_step(model, x_val, y_val)
            ppl, t1, t3, t5 = metrics

            val_loss += loss
            val_ppl += ppl
            val_t1 += t1
            val_t3 += t3
            val_t5 += t5
            val_batches += 1

        val_loss = float(val_loss / val_batches)
        val_ppl  = float(val_ppl / val_batches)
        val_t1   = float(val_t1 / val_batches)
        val_t3   = float(val_t3 / val_batches)
        val_t5   = float(val_t5 / val_batches)


        # Update History
        history["train_loss"].append(tr_loss)
        history["train_ppl"].append(tr_ppl)
        history["train_top1"].append(tr_t1)

        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["val_top1"].append(val_t1)
        history["val_top3"].append(val_t3)
        history["val_top5"].append(val_t5)

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_model = model
            patience_counter = 0
            status = "Best"
        else:
            patience_counter += 1
            status = f"Patience {patience_counter}/{patience}"

        # Logging results
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss {tr_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"Train Perplexity {tr_ppl:.3f} | Val Perplexity {val_ppl:.3f} | Train top-1 Acc {tr_t1:.3f} |\n"
            f"Val top-1 Acc {val_t1:.3f} | Val top-3 Acc {val_t3:.3f} | Val top-5 Acc {val_t5:.3f} | "
            f"{status}"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    return best_model, history

def decode_no_cache(model, prompt, max_new_tokens):
    tokens = prompt
    per_token_times = []

    # TTFT
    t0 = time.perf_counter()
    logits = model(tokens, training=False)
    jax.block_until_ready(logits)
    ttft = time.perf_counter() - t0

    next_token = jnp.argmax(logits[:, -1, :], axis=-1)

    # Autoregressive Loop
    for _ in range(max_new_tokens - 1):
        tokens = jnp.concatenate([tokens, next_token[:, None]], axis=1)

        t0 = time.perf_counter()
        logits = model(tokens, training=False)
        jax.block_until_ready(logits)
        per_token_times.append(time.perf_counter() - t0)

        next_token = jnp.argmax(logits[:, -1, :], axis=-1)

    return {
        "ttft": ttft,
        "per_token_times": per_token_times,
    }


def decode_with_cache(model, prompt, max_new_tokens, start_pos):
    batch = prompt.shape[0]
    model = model.init_cache(batch)

    per_token_times = []

    # Warmup
    t0 = time.perf_counter()
    model, logits = model.decode(prompt[:, 0], start_pos)
    jax.block_until_ready(logits)
    ttft = time.perf_counter() - t0

    next_token = jnp.argmax(logits, axis=-1)

    # Incremental Decoding
    for i in range(1, max_new_tokens):
        t0 = time.perf_counter()
        model, logits = model.decode(next_token, start_pos + i)
        jax.block_until_ready(logits)
        per_token_times.append(time.perf_counter() - t0)

        next_token = jnp.argmax(logits, axis=-1)

    # Cache length
    cache_len = model.transf_block1.attn.cache.key.shape[2]

    return {
        "ttft": ttft,
        "per_token_times": per_token_times,
        "cache_len": cache_len,
    }


def benchmark_model(
    model,
    test_loader,
    *,
    max_new_tokens=32,
    start_pos=0,
):
    model = eqx.tree_inference(model, value=True)

    results = {
        "ttft_no_cache": [],
        "tok_time_no_cache": [],
        "ttft_cache": [],
        "tok_time_cache": [],
        "final_cache_len": [],
    }

    for x, _ in tfds.as_numpy(test_loader):
        x = jnp.asarray(x)
        prompt = x[:, :1]   # one-token prompt

        # No Cache
        nc = decode_no_cache(model, prompt, max_new_tokens)

        # Cache
        c = decode_with_cache(model, prompt, max_new_tokens, start_pos)

        results["ttft_no_cache"].append(nc["ttft"])
        results["tok_time_no_cache"].extend(nc["per_token_times"])
        results["ttft_cache"].append(c["ttft"])
        results["tok_time_cache"].extend(c["per_token_times"])
        results["final_cache_len"].append(c["cache_len"])

    return results

def test_model(test_loader, model):
     test_iter = tfds.as_numpy(test_loader)

     test_loss = test_ppl = test_t1 = test_t3 = test_t5 = jnp.array(0.0)
     test_batches = 0

     for x_test, y_test in test_iter:
          x_test = jnp.asarray(x_test)
          y_test = jnp.asarray(y_test)

          loss, metrics = eval_step(model, x_test, y_test)
          ppl, t1, t3, t5 = metrics

          test_loss += loss
          test_ppl  += ppl
          test_t1   += t1
          test_t3   += t3
          test_t5   += t5
          test_batches += 1

     test_loss = float(test_loss / test_batches)
     test_ppl  = float(test_ppl / test_batches)
     test_t1   = float(test_t1 / test_batches)
     test_t3   = float(test_t3 / test_batches)
     test_t5   = float(test_t5 / test_batches)

     print(f"Test Loss: {test_loss:.4f}")
     print(f"Test Perplexity: {test_ppl:.2f}")
     print(f"Test top-1 Accuracy: {100*test_t1:.2f}%")
     print(f"Test top-3 Accuracy: {100*test_t3:.2f}%")
     print(f"Test top-5 Accuracy: {100*test_t5:.2f}%")

def summary(name, data):
    print(f"{name} Time To First Token (TTFT) Summary")
    print(f"Mean:   {np.mean(data):.4f} s")
    print(f"Median: {np.median(data):.4f} s")
    print(f"Std:    {np.std(data):.4f} s")
    print(f"Min:    {np.min(data):.4f} s")
    print(f"Max:    {np.max(data):.4f} s")
    print(f"95th Percentile:    {np.percentile(data, 95):.4f} s")
    print(f"99th Percentile:    {np.percentile(data, 99):.4f} s")
    print()


def evaluate_metrics_compression(model, test_loader, cache_lengths):
    results = {
        "cache_lengths": cache_lengths,
        "perplexity": [],
        "top1": [], "top3": [], "top5": []
    }

    for k_len in cache_lengths:
        total_nll = 0.0
        correct_1, correct_3, correct_5 = 0, 0, 0
        total_tokens = 0

        for batch_x, batch_y in test_loader:
            batch_x = jnp.array(batch_x.numpy())
            batch_y = jnp.array(batch_y.numpy())
            num_b, seq_l = batch_x.shape

            current_model = model.init_cache(num_b)

            for t in range(seq_l):
                current_model, logits = current_model.decode(batch_x[:, t], t)

                targets = batch_y[:, t]
                # Create mask to ignore padding
                mask = (targets != 0)
                mask_sum = jnp.sum(mask)

                # Skip calculations if the entire batch at this timestep is padding
                if mask_sum == 0:
                    continue

                log_probs = jax.nn.log_softmax(logits, axis=-1)

                # Masked Perplexity Calculation
                target_log_probs = jnp.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze()
                total_nll += -jnp.sum(target_log_probs * mask)

                # Masked Accuracy Calculation
                top_indices = jnp.argsort(logits, axis=-1)
                correct_1 += jnp.sum((targets == top_indices[:, -1]) * mask)
                correct_3 += jnp.sum(jnp.any(targets[:, None] == top_indices[:, -3:], axis=-1) * mask)
                correct_5 += jnp.sum(jnp.any(targets[:, None] == top_indices[:, -5:], axis=-1) * mask)

                total_tokens += mask_sum

                # Compression
                if current_model.transf_block1.attn.cache.key.shape[2] > k_len:
                    new_c1 = compress_cache(current_model.transf_block1.attn.cache, k_len)
                    new_c2 = compress_cache(current_model.transf_block2.attn.cache, k_len)
                    current_model = eqx.tree_at(
                        lambda m: (m.transf_block1.attn.cache, m.transf_block2.attn.cache),
                        current_model, (new_c1, new_c2)
                    )

        # Final stats for this cache length
        avg_nll = total_nll / (total_tokens + 1e-9)
        results["perplexity"].append(float(jnp.exp(avg_nll)))
        results["top1"].append(float(correct_1 / (total_tokens + 1e-9)))
        results["top3"].append(float(correct_3 / (total_tokens + 1e-9)))
        results["top5"].append(float(correct_5 / (total_tokens + 1e-9)))

    return results