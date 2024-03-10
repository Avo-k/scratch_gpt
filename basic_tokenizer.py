import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from itertools import pairwise


def is_sublist_of_length_two(sublist, mainlist):
    return any(sublist == pair for pair in pairwise(mainlist))


class BasicTokenizer:
    def __init__(self):
        self.decoding_dict = {}

    @staticmethod
    def get_stats(ids, counts=None):
        """
        Given a list of integers, return a dictionary of counts of consecutive pairs
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Optionally allows to update an existing dictionary of counts
        """
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):  # iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def get_most_common_pair(self, enc):
        stats = self.get_stats(enc)
        return max([(c, p) for p, c in stats.items()])

    # @staticmethod
    # def replace_pair_with_key(seq, mcp, key):

    #     if not is_sublist_of_length_two(mcp, seq):
    #         return seq

    #     seq = np.array(seq, dtype="object")

    #     view = sliding_window_view(seq, mcp_length)
    #     indexes = np.where((view == mcp).all(axis=1))[0]
    #     replace = np.empty(0, dtype="object")

    #     for index in indexes:
    #         replace = np.append(replace, [key] + [None] * (mcp_length - 1))

    #     np.put(seq, np.concatenate([indexes + i for i in range(mcp_length)]), replace)
    #     out = seq[seq != None]
    #     return out.tolist()

    @staticmethod
    def replace_pair_with_key(seq: list, mcp: list, key: int):
        "replace pair of values by a corresponding key, return a new seq"

        if not is_sublist_of_length_two(mcp, seq):
            return seq

        i = 0
        while i < len(seq) - 1:
            if seq[i] == mcp[0] and seq[i + 1] == mcp[1]:
                seq[i] = key
                del seq[i + 1]
                i += 1
            i += 1
        return seq

    @staticmethod
    def replace_key_with_pair(seq, key, pair):
        i = 0
        while i < len(seq):
            if seq[i] == key:
                seq[i : i + 1] = pair
                i += len(pair) - 1
            i += 1
        return seq

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:

        assert vocab_size > 256

        encoded_text = list(text.encode("utf-8"))
        starting_len = len(encoded_text)
        next_key = 256

        for i in range(vocab_size - 256):
            count, mcp = self.get_most_common_pair(encoded_text)

            self.decoding_dict[next_key] = mcp
            encoded_text = self.replace_pair_with_key(encoded_text, mcp, next_key)

            if verbose:
                print(f"replacing {mcp} by {next_key} occuring {count} times ")

            next_key += 1

        if verbose:
            print("-" * 30)
            print(f"{starting_len:_} tokens")
            print(f"{len(encoded_text):_} ids")
            print(f"{starting_len/len(encoded_text):.02f}x compression ratio")
            print("-" * 30)

    def encode(self, text):
        enc = list(text.encode("utf-8"))
        for key, pair in self.decoding_dict.items():
            enc = self.replace_pair_with_key(enc, pair, key)
        return enc

    def decode(self, ids):
        for key, pair in reversed(list(self.decoding_dict.items())):
            ids = self.replace_key_with_pair(ids, key, pair)
        return bytes(ids).decode("utf-8")
