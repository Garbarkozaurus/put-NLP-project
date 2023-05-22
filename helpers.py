import enchant
from enchant.checker import SpellChecker
from enchant.tokenize import get_tokenizer
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch
from copy import deepcopy

def mask_sentence(sentence):
    checker = SpellChecker("en_US", sentence)
    misspelled_words = []
    for err in checker:
        misspelled_words.append(err.word)
        err.replace("[MASK]")
    masked_sentence = checker.get_text()
    return masked_sentence, misspelled_words

def levenshtein(a, b):
    assert type(a) is str
    assert type(b) is str
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    if a[0] == b[0]:
        return levenshtein(a[1:], b[1:])
    else:
        return 1 + min(levenshtein(a, b[1:]), levenshtein(a[1:], b), levenshtein(a[1:], b[1:]))

#TODO: what if 2 words are min???
def best_word_from_list(a, words_list, func = levenshtein):
    mn = 1000000
    mn_word = ""#[]
    for b in words_list:
        if func(a, b) < mn:
            mn = func(a, b)
            #print("Better score found: ", mn, "-->", b)
            mn_word = ""#[]
            #mn_word.append(b)
        #elif func(a, b) == mn:
        #    print("\tsimilar score word found: ", b)
        #    mn_word.append(b)

    return mn, mn_word

def choose_words(text, k=50):
    def top_k(masked_word, k):
        return torch.topk(masked_word, k, dim = 1)[1][0]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased',    return_dict = True)

    input = tokenizer.encode_plus(text, return_tensors = "pt")
    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

    output = model(**input)
    logits = output.logits
    softmax = F.softmax(logits, dim = -1)

    best_words = []
    for index in mask_index[0]:
        index = torch.tensor([index])
        mask_word = softmax[0, index, :]
        top = top_k(mask_word, k)
        best_words.append(top)

    best_words = [[tokenizer.decode([t]) for t in words]for words in best_words]
    return best_words
