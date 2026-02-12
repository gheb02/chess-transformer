import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences



def filter_and_clean_moves(df, pattern=r'\d+\.+\s?'):

     df = df.copy()
     df = df.dropna()
     # Remove links for event name
     df['Event'] = df['Event'].str.replace(r'http\S+', '', regex=True).str.strip()
     df = df[df['Event'].str.contains('classical', case=False, na=False)]

     # Remove turn indices
     df['clean_moves'] = df['Moves'].str.replace(pattern, '', regex=True).str.strip()

     # Convert each sequence to list and add [BOS] and [EOS] at the beginning and end of each sequence
     df['clean_moves'] = df['clean_moves'].str.split()
     df['clean_moves'] = [['[BOS]'] + moves + ['[EOS]'] for moves in df['clean_moves']]
     return df

class Tokenizer:
    def __init__(self, vocab):
        """
        vocab: list of tokens, including special tokens
        """
        self.vocab = vocab
        self.str_to_idx = {token: i for i, token in enumerate(vocab)}
        self.idx_to_str = {i: token for i, token in enumerate(vocab)}

    @classmethod
    def create_vocab(cls, move_sequences, special_tokens=None):
        """
        move_sequences: list[list[str]]
        """
        if special_tokens is None:
            special_tokens = ['[PAD]', '[BOS]', '[EOS]', '[UNK]']
        else:
            if '[PAD]' in special_tokens:
                special_tokens.remove('[PAD]')
            special_tokens = ['[PAD]'] + special_tokens

        moves = set()
        for seq in move_sequences:
            moves.update(seq)

        # Remove any potential duplicates if moves already contains special tokens
        for token in special_tokens:
            moves.discard(token)

        vocab = special_tokens + sorted(list(moves))
        print(f'Size of the vocabulary: {vocab}')
        return cls(vocab)

    def encode(self, moves):
        """
        moves: list[str]
        returns: list[int]
        """
        unk = self.str_to_idx['[UNK]']
        return [self.str_to_idx.get(m, unk) for m in moves]

    def decode(self, ids, skip_special_tokens=False):
      """
      ids: list[int] or np.array
      returns: list[str]
      """
      ids = np.array(ids).flatten()
      tokens = [self.idx_to_str[int(i)] for i in ids]

      if skip_special_tokens:
          tokens = [
              t for t in tokens
              if t not in {'[PAD]', '[BOS]', '[EOS]'}
          ]
      return tokens
    
def split_input_target(sequence):
    x = sequence[:-1]
    y = sequence[1:]
    return x, y

def create_dataset(moves_array, batch_size=128, buffer_size=1000, is_training=False):
    dataset = tf.data.Dataset.from_tensor_slices(moves_array)

    # Create (x, y) pairs
    dataset = dataset.map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle for training
    if is_training:
        dataset = dataset.shuffle(buffer_size)

    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset