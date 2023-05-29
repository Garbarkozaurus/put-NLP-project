import enchant
from enchant.checker import SpellChecker
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch


def mask_sentence(sentence):
    """
    Masks incorrect words according to enchant library.
    Returns masked seentence and a list of misspelled words (in order)
    """
    checker = SpellChecker("en_US", sentence)
    misspelled_words = []
    for err in checker:
        misspelled_words.append(err.word)
        err.replace("[MASK]")
    masked_sentence = checker.get_text()

    return masked_sentence, misspelled_words


def best_word_from_list(a, words_list, func=enchant.utils.levenshtein):
    """Return closest word to words from words_list and its score."""
    mn = 1000000
    mn_word = ""
    for b in words_list:
        if func(a, b) < mn:
            mn = func(a, b)
            mn_word = b
    return mn, mn_word


def best_words_from_list(a, words_list, func=enchant.utils.levenshtein):
    """Return closest words to words from words_list and its score."""
    mn = 1000000
    mn_word = []
    for b in words_list:
        if func(a, b) < mn:
            mn = func(a, b)
            mn_word = [b]
        elif func(a, b) == mn:
            mn_word.append(b)
    return mn, str(mn_word)


def choose_words(text, k=200):
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

    best_words = [[tokenizer.decode([t])for t in words]for words in best_words]
    return best_words


def correct_sentence(input_sentence):
    """Corrects input sentence to its probably original meaning"""
    masked_sentence, misspelled_words = mask_sentence(input_sentence)
    best_words = choose_words(masked_sentence)
    for misspelled, best_words in zip(misspelled_words, best_words):
        _, best = best_word_from_list(misspelled, best_words)
        print(_)
        input_sentence = input_sentence.replace(misspelled, best, 1)
    return input_sentence
