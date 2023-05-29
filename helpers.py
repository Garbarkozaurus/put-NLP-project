import enchant
from enchant.checker import SpellChecker
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch
from typing import Callable


def mask_sentence(string: str):
    """Masks incorrect words according to enchant library.
    Returns masked string and a list of misspelled words (in order)
    """
    checker = SpellChecker("en_US", string)
    misspelled_words = []
    for err in checker:
        misspelled_words.append(err.word)
        err.replace("[MASK]")
    masked_string = checker.get_text()

    return masked_string, misspelled_words


def best_word_from_list(target_word: str, words_list: list[str],
                        func: Callable = enchant.utils.levenshtein) \
                        -> tuple[int, str]:
    """Return the word with minimum distance (as determined by `func`) to
    `target_word` from `words_list` and its distance"""
    mn = 1000000
    mn_word = ""
    for b in words_list:
        if func(target_word, b) < mn:
            mn = func(target_word, b)
            mn_word = b
    return mn, mn_word


def best_words_from_list(target_word: str, words_list: list[str],
                         func: Callable = enchant.utils.levenshtein) \
                         -> tuple[int, list[str]]:
    """Return all words with minimum distance (as determined by `func`) to
      `target_word` from `words_list` and the corresponding minimum distance"""
    mn = 1000000
    mn_word = []
    for b in words_list:
        if func(target_word, b) < mn:
            mn = func(target_word, b)
            mn_word = [b]
        elif func(target_word, b) == mn:
            mn_word.append(b)
    return mn, mn_word


def choose_words(text: str, k: int = 200) -> list[list[str]]:
    """Find k best substitutions for each [MASK] in the text using BERT."""
    def top_k(masked_word, k):
        return torch.topk(masked_word, k, dim=1)[1][0]

    bert_dict = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_dict)
    model = BertForMaskedLM.from_pretrained(bert_dict, return_dict=True)

    input = tokenizer.encode_plus(text, return_tensors="pt")
    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

    output = model(**input)
    logits = output.logits
    softmax = F.softmax(logits, dim=-1)

    best_words = []
    for index in mask_index[0]:
        index = torch.tensor([index])
        mask_word = softmax[0, index, :]
        top = top_k(mask_word, k)
        best_words.append(top)

    best_words = [[tokenizer.decode([t]) for t in words]
                  for words in best_words]
    return best_words


def correct_sentence(input_string: str) -> str:
    """Corrects input string to its probably original meaning"""
    masked_sentence, misspelled_words = mask_sentence(input_string)
    best_words = choose_words(masked_sentence)
    for misspelled, best_candidates in zip(misspelled_words, best_words):
        _, best = best_word_from_list(misspelled, best_candidates)
        input_string = input_string.replace(misspelled, best, 1)
    return input_string
