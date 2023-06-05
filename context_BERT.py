#!/usr/bin/env python3

from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch
from enchant.tokenize import get_tokenizer
from enchant.checker import SpellChecker
from noisify import noisify
from enchant.utils import levenshtein
from typing import Callable, Optional
from sys import argv, stderr


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


def clean_and_noisy_lists(file_path: str, random_char_prob: float = 0.05,
                          noise_seed: Optional[int] = None
                          ) -> tuple[list[str], list[str]]:
    """Given path to a file, return two lists of words:
      - list of words contained in the file on `file_path`, tokenized using
        `enchanted_words()`
      - the same list, but with each character could be changed to a random
       letter by `noisify()` with probability `random_char_prob`
    """
    words_from_file = enchanted_words(file_path)
    noisy_text = noisify(" ".join(words_from_file).lower(), random_char_prob,
                         seed=noise_seed)
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
    context_end_idx = min(mask_index+context_after, len(masked_words)-1)
    masked_text = " ".join(masked_words[context_start_idx:context_end_idx+1])
    if print_context:
        print(masked_text, "||", mask_index-context_start_idx)
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


def apply_sequential_BERT(model: BertForMaskedLM, tokenizer: BertTokenizer,
                          noisy_words: list[str],
                          masked_words: list[str], mask_indices: list[int],
                          context_before: int = 10, context_after: int = 10,
                          k: int = 10, print_context: bool = False
                          ) -> list[str]:
    """Sequentially fills all masks in a list of words. Later predictions
    use the values that were already predicted for earlier masks.
    :param lit[srt] noisy_words: list of noisy words (essentially:
    masked_words, but without masks). Used for selecting best BERT suggestion,
    based on minimum levenshtein distance
    :param list[str] masked_words: complete list of masked words within which
    the context is situated
    :param list[int] mask_indices: positions of of all masked tokens for BERT
    to predict within `masked_words`
    :param int context_before: at most how many tokens before the main mask
    should be used as context for the prediction. Contexts are truncated to
    never include other masks.
    :param int context_after: at most how many tokens after the main mask
    should be used as context for the prediction.
    :param int k: how many of the most likely predictions should be taken
    returned for each mask. The top among them is then selected using
    :function:`best_word_from_list()`

    :return List of words with masks replaced by appropriate suggestions
    :rtype: list[str]
    """
    restored_words = masked_words[:]
    for i, mask_index in enumerate(mask_indices):
        previous_mask = 0 if i == 0 else mask_indices[i-1]
        next_mask = len(masked_words)-1 if i == len(mask_indices)-1 else \
            mask_indices[i+1]
        c_before = min(mask_index-previous_mask-1, context_before)
        c_after = min(next_mask-mask_index-1, context_after)
        predictions = apply_BERT_to_context(model, tokenizer, restored_words,
                                            mask_index, c_before, c_after,
                                            k, print_context)
        suggestion = best_word_from_list(noisy_words[mask_index], predictions)
        restored_words[mask_index] = suggestion
    return restored_words


def best_word_from_list(target_word: str, word_conf_list:
                        list[tuple[str, float]], func: Callable =
                        levenshtein) -> str:
    """Find the word that is the most similar to `target_word` among a list
    of BERT's suggestions
    :param list[tuple[str, float]] word_conf_list: list of (word, confidence)
     pairs - outputs of BERT which were supposed to replace the
    (masked to BERT) `target_word`
    :param Callable func: distance metric used to compare the suggestions with
    `target_word`. In case of a tie - the suggestion BERT is more confident
    in is selected.
    """
    min_dist = 1000
    max_conf = 0.0
    best_word = ""
    for word, conf in word_conf_list:
        dist = func(target_word, word)
        if dist < min_dist or (dist == min_dist and conf > max_conf):
            min_dist = dist
            max_conf = conf
            best_word = word
    return best_word


def present_results(mask_idx: int, word_conf_list: list[tuple[str, float]],
                    clean_words: list[str], noisy_words: list[str],
                    context_before: int = 10, context_after: int = 10) -> None:
    """Print the results of the experiment to standard output

    They include:
    - clean context: the piece of the original text without errors
    - noisy context: clean context after noisification and applying masks
    to misspelled words. This is what BERT was dealing with
    - restored text - noisy context but with the central mask replaced
    by the BERT suggestion which had the highest similarity to the noisy word.
    The new word is surrounded with inequality signs, with the vertices
    pointing towards it, e. g. >steamed<
    """
    # the two helper variables below ensure that context windows are extracted
    # properly around the edges of the input
    # they also make everything way more readable
    context_start_idx = max(mask_idx-context_before, 0)
    context_end_idx = min(mask_idx+context_after, len(masked_words)-1)
    target_word = noisy_words[mask_idx]
    top_suggestion = best_word_from_list(target_word, word_conf_list)

    clean_context = " ".join(clean_words[context_start_idx:context_end_idx+1])
    masked_context_list = masked_words[context_start_idx:context_end_idx+1]
    noisy_context_list = noisy_words[context_start_idx:context_end_idx+1]
    noisy_context_words = " ".join(noisy_context_list)
    masked_context_list[mask_idx-context_start_idx] = '>'+top_suggestion+'<'
    noisy_with_suggestion = " ".join(masked_context_list)

    print(f"CLEAN CONTEXT: {clean_context}")
    print(f"NOISY CONTEXT: {noisy_context_words}")
    print(f"RESTORED: {noisy_with_suggestion}")


def evaluate_BERT_quality(correct_words: list[str], predicted_words: list[str],
                          noisy_words: list[str]
                          ) -> tuple[float, float, float]:
    """Gives basic metrics of prediction quality (accuracy & 2 lev. distances)
    :param list[str] correct_words: words from the original text, that were
    noisified and masked - the ones we want the model to predict
    :param list[str] predicted_words: the words that the model predicted to
    the replace the corresponding correct words
    :param list[str] noisy_words: the words that were masked; Levenshtein
    distance to them used to determine the best choice from model's outputs
    :return Accuracy of the suggestions (ratio of perfect/all); average
    levenshtein distance to target clean word; average levenshtein distance
    to the noisy word
    :rtype tuple[float, float, float]
    """
    perfect_suggestions = 0
    target_levenshtein_sum = 0
    noisy_lev_sum = 0
    n = len(correct_words)
    for correct, predicted, noisy in zip(correct_words, predicted_words,
                                         noisy_words):
        if correct.lower() == predicted:
            perfect_suggestions += 1
        target_levenshtein_sum += levenshtein(correct, predicted)
        noisy_lev_sum += levenshtein(noisy, predicted)
    return perfect_suggestions/n, target_levenshtein_sum/n, noisy_lev_sum/n


if __name__ == "__main__":
    CONTEXT_BEFORE = 100
    CONTEXT_AFTER = 100
    TOP_K = 200
    assert len(argv) == 2, "Exactly 1 argument expected (path to text corpus)"
    clean_words, noisy_words = clean_and_noisy_lists(argv[1])
    mask_idxs = masked_indices(noisy_words)
    masked_words = [mask_word(w) for w in noisy_words]
    model = BertForMaskedLM.from_pretrained('bert-base-uncased',
                                            return_dict=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    win_predictions = []
    for i, mask_idx in enumerate(mask_idxs):
        print(f"\rpredicting mask {i}/{len(mask_idxs)}" +
              f"({i / len(mask_idxs) * 100:.2f}%) ... ", end="", file=stderr)
        BERT_predictions = apply_BERT_to_context(model, tokenizer,
                                                 masked_words, mask_idx,
                                                 CONTEXT_BEFORE,
                                                 CONTEXT_AFTER, TOP_K)
        pred = best_word_from_list(noisy_words[mask_idx], BERT_predictions)
        win_predictions.append(pred)
        # present_results(mask_idx, BERT_predictions, clean_words, noisy_words,
        #                 CONTEXT_BEFORE, CONTEXT_AFTER)
        # print()
    print("done.", file=stderr)

    seq_restored_words = apply_sequential_BERT(model, tokenizer, noisy_words,
                                               masked_words, mask_idxs,
                                               CONTEXT_BEFORE, CONTEXT_AFTER,
                                               TOP_K)
    targets = []
    seq_predictions = []
    noisy = []
    print("CLEAN, NOISY, WINDOW, SEQUENTIAL")
    for i, id in enumerate(mask_idxs):
        print(clean_words[id], noisy_words[id], win_predictions[i],
              seq_restored_words[id])
        targets.append(clean_words[id])
        seq_predictions.append(seq_restored_words[id])
        noisy.append(noisy_words[id])

    win_acc, win_avg_lev, win_noisy_lev = \
        evaluate_BERT_quality(targets, win_predictions, noisy)
    seq_acc, seq_avg_lev, seq_noisy_lev = \
        evaluate_BERT_quality(targets, seq_predictions, noisy)
    print("WINDOW:")
    print(f"ACCURACY: {win_acc}\tAVERAGE LEVENSHTEIN DISTANCE TO TARGET: "
          f"{win_avg_lev}\t AVERAGE LEV. DISTANCE TO NOISY: {win_noisy_lev}")
    print("SEQUENTIAL")
    print(f"ACCURACY: {seq_acc}\tAVERAGE LEVENSHTEIN DISTANCE TO TARGET: "
          f"{seq_avg_lev}\t AVERAGE LEV. DISTANCE TO NOISY: {seq_noisy_lev}")
