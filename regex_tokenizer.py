import regex as re

from basic_tokenizer import BasicTokenizer
from tqdm import trange


def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode("utf-8", errors="replace")
    return repr(s)


class RegexTokenizer(BasicTokenizer):
    def __init__(self):
        super().__init__()
        self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.GPT4_SPLIT_PATTERN)

        self.merges = {}  # key: pair of bytes
        self.vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size > 256

        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        starting_len = len(list(text.encode("utf-8")))

        for key in trange(256, vocab_size):
            stats = {}
            for chunk_ids in ids:
                self.get_stats(chunk_ids, stats)

            mcp, count = max(stats.items(), key=lambda x: x[1])
            mcp1, mcp2 = mcp
            self.merges[key] = mcp
            self.vocab[key] = self.vocab[mcp1] + self.vocab[mcp2]
            ids = [self.replace_pair_with_key(chunk_ids, mcp, key) for chunk_ids in ids]

            # if verbose:
            #     print(f"replacing {mcp} by {key} occuring {count} times ")

        len_ids = len(sum(ids, []))

        if verbose:
            print("-" * 30)
            print(f"{starting_len:_} tokens")
            print(f"{len_ids:_} ids")
            print(f"{starting_len/len_ids:.02f}x compression ratio")
            print("-" * 30)

    def write_vocab(self):

        vocab_file = "regex.vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")


def test_tokenizer(tok):

    with open("shake.txt", "r") as f:
        shake = f.read()
    starting_shake = shake[:500_000]
    test_set = shake[:100_000]

    tok.train(shake, 10_000, True)

    # encoded = tok.encode(test_set)
    # decoded = tok.decode(encoded)

    # assert test_set == decoded


if __name__ == "__main__":
    tok = RegexTokenizer()
    test_tokenizer(tok)

    tok.write_vocab()
