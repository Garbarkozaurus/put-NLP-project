from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch
from enchant.tokenize import get_tokenizer
from enchant.checker import SpellChecker
from noisify import noisify


def enchanted_words(file_path: str) -> list[str]:
    """Given a file, load all words within it into a list.
    The words are separated using a tokenizer from the `enchant.tokenize`
    library
    :param str file_path: Path to the file to load
    """
    tokenizer = get_tokenizer("en_US")
    text_original = ""
    with open(file_path, 'r') as file:
        text_original = file.read()

    word_list = [w for w in tokenizer(text_original)]
    return [w[0] for w in word_list]


def mask_word(word: str, spell_checker: SpellChecker = SpellChecker("en_US"),
              mask: str = "[MASK]") -> str:
    """Use the `spell_checker` to decide if the word is spelled correctly. In
    such cases returns the word without any change. Otherwise the value of
    `mask` is returned.
    """
    if spell_checker.check(word) or spell_checker.check(word.capitalize()):
        return word
    return mask


def masked_indices(word_list: list[str], spell_checker: SpellChecker =
                   SpellChecker("en_US")) -> list[int]:
    """Returns the list of indices of misspelled words in `word_list`
    """
    indices = []
    for idx, word in enumerate(word_list):
        if not spell_checker.check(word) and \
           not spell_checker.check(word.capitalize()):
            indices.append(idx)
    return indices


def clean_and_noisy_lists(file_path: str, random_char_prob: float = 0.05) -> \
                          tuple[list[str], list[str]]:
    """Given path to a file, return two lists of words:
      - list of words contained in the file on `file_path`, tokenized using
        `enchanted_words()`
      - the same list, but with each character could be changed to a random
       letter by `noisify()` with probability `random_char_prob`
    """
    words_from_file = enchanted_words(file_path)
    noisy_text = noisify(" ".join(words_from_file).lower(), random_char_prob)
    # no need to use enchant here - it might get thrown off by weird, randomly
    # generated words and noisifer can't mess with spaces, so this ensures
    # that the lists represent corresponding words
    noisy_words = noisy_text.split()
    return words_from_file, noisy_words


def apply_BERT_to_context(model: BertForMaskedLM, tokenizer: BertTokenizer,
                          masked_words: list[str], mask_index: int,
                          context_before: int = 10, context_after: int = 10,
                          k: int = 10, print_context: bool = False) \
                          -> list[tuple[str, float]]:
    """Given a BERT model and tokenizer, predict the top k most likely values
    of a mask token present at `mask_index` within `masked_words` based on
    a context of specified dimensions. Returns a list of pairs containing the
    `k` tokens BERT considers to be most likely and their prediction values
    :param list[str] masked_words: complete list of masked words within which
    the context is situated
    :param int mask_index: position of the masked token for BERT to predict
    within `masked_words`
    :param int context_before: at most how many tokens before the main mask
    should be used as context for the prediction
    :param int context_after: at most how many tokens after the main mask
    should be used as context for the prediction
    :param int k: how many of the most likely predictions should be returned
    """
    context_start_idx = max(mask_index-context_before, 0)
    context_end_idx = min(mask_index+context_after+1, len(masked_words)-1)
    masked_text = " ".join(masked_words[context_start_idx:context_end_idx])
    if print_context:
        print(masked_text, "mask idx:", mask_index-context_start_idx)
    model_input = tokenizer.encode_plus(masked_text, return_tensors="pt")
    model_output = model(**model_input)
    logits = model_output.logits
    softmax = F.softmax(logits, dim=-1)
    word_masks = softmax[0, torch.tensor([mask_index-context_start_idx]), :]
    top_k = torch.topk(word_masks, k, dim=1)
    top_k_values = top_k.values.detach().numpy()[0]
    top_k_indices = top_k.indices[0]
    top_k_words = [tokenizer.decode([idx]) for idx in top_k_indices]
    return list(zip(top_k_words, top_k_values))


if __name__ == "__main__":
    clean_words, noisy_words = clean_and_noisy_lists("pastas.txt")
    mask_idxs = masked_indices(noisy_words)
    masked_words = [mask_word(w) for w in noisy_words]
    model = BertForMaskedLM.from_pretrained('bert-base-uncased',
                                            return_dict=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for mask_idx in mask_idxs:
        BERT_predictions = apply_BERT_to_context(model, tokenizer,
                                                 masked_words, mask_idx,
                                                 print_context=True)
        for token, value in BERT_predictions:
            print(token, value)
