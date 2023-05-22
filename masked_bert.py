from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch
from enchant.tokenize import get_tokenizer
from enchant.checker import SpellChecker
import diagnostics
from noisify import noisify


def enchanted_words(file_path: str) -> list[str]:
    tokenizer = get_tokenizer("en_US")
    text_original = ""
    with open(file_path, 'r') as file:
        text_original = file.read()

    word_list = [w for w in tokenizer(text_original)]
    return [w[0] for w in word_list]


def enchanted_words_from_str(text: str) -> list[str]:
    # tokenizer = get_tokenizer("en_US")
    # word_list = [w for w in tokenizer(text)]
    word_list = text.split()
    return [w[0] for w in word_list]


def mask_word(word: str, spell_checker: SpellChecker,
              mask: str = "[MASK]") -> str:
    if spell_checker.check(word) or spell_checker.check(word.capitalize()):
        return word
    return mask


def masked_indices(word_list: list[str], spell_checker: SpellChecker) \
                   -> list[int]:
    indices = []
    for idx, word in enumerate(word_list):
        if not spell_checker.check(word):
            indices.append(idx)
    return indices


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)

text_from_file = open("moby_bit.sty").read().replace("\n", " ").replace("\t", " ")
noisy_text = noisify(text_from_file.lower(), 0.05)
# words = enchanted_words("moby_bit.sty")
# words = enchanted_words_from_str(noisy_text)
# words = text_from_file.split()
# clean_words = enchanted_words_from_str(text_from_file)
words = noisy_text.split()
clean_words = text_from_file.split()
print(len(words), len(clean_words))
spell_checker = SpellChecker("en_US")
masked_words = [mask_word(w, spell_checker, tokenizer.mask_token)
                for w in words]
text_masked = " ".join(masked_words)

# words = list(["I", "would", "like", "ot", "try", "tomats", "on", "pizza"])
# text_masked = "I would like [MASK] try [MASK] on pizza"

model_input = tokenizer.encode_plus(text_masked, return_tensors="pt")
mask_index = torch.where(model_input["input_ids"][0] ==
                         tokenizer.mask_token_id)
m_indices = masked_indices(words, spell_checker)
output = model(**model_input)
for mask_idx, idx in zip(m_indices, mask_index[0]):
    idx = torch.tensor([idx])
    logits = output.logits
    softmax = F.softmax(logits, dim=-1)
    word_masks = softmax[0, idx, :]
    top_10 = torch.topk(word_masks, 10, dim=1)[1][0]
    print(mask_idx, clean_words[mask_idx], words[mask_idx])
    print("==========")
    for token in top_10:
        word = tokenizer.decode([token])
        print(word)
    # print()
    # diagnostics.test_single(words[mask_idx], )