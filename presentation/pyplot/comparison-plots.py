#!/usr/bin/env python3

import matplotlib.pyplot as plt

# Display names of each compared model
MODEL_BASE_WIN = "bw"
MODEL_BASE_SEQ = "bs"
MODEL_LARGE_WIN = "Lw"
MODEL_LARGE_SEQ = "Ls"

# Display names of each compared measure
MEASURE_ACCURACY = "Accuracy"
MEASURE_LEVENSHTEIN_CLEAN = "Levenshtein (clean)"
MEASURE_LEVENSHTEIN_NOISY = "Levenshtein (noisy)"

# Data (filled in manually)
data = {
    "pastas.txt": {
        MEASURE_ACCURACY: {
            MODEL_BASE_WIN: 0.172,
            MODEL_BASE_SEQ: 0.222,
            MODEL_LARGE_WIN: 0.170,
            MODEL_LARGE_SEQ: 0.212,
        },
        MEASURE_LEVENSHTEIN_CLEAN: {
            MODEL_BASE_WIN: 3.280,
            MODEL_BASE_SEQ: 3.158,
            MODEL_LARGE_WIN: 3.239,
            MODEL_LARGE_SEQ: 3.066,
        },
        MEASURE_LEVENSHTEIN_NOISY: {
            MODEL_BASE_WIN: 3.179,
            MODEL_BASE_SEQ: 3.153,
            MODEL_LARGE_WIN: 3.179,
            MODEL_LARGE_SEQ: 3.088,
        },
    },
    "101-convos.txt": {
        MEASURE_ACCURACY: {
            MODEL_BASE_WIN: 0.339,
            MODEL_BASE_SEQ: 0.314,
            MODEL_LARGE_WIN: 0.331,
            MODEL_LARGE_SEQ: 0.308,
        },
        MEASURE_LEVENSHTEIN_CLEAN: {
            MODEL_BASE_WIN: 2.273,
            MODEL_BASE_SEQ: 2.430,
            MODEL_LARGE_WIN: 2.266,
            MODEL_LARGE_SEQ: 2.439,
        },
        MEASURE_LEVENSHTEIN_NOISY: {
            MODEL_BASE_WIN: 2.427,
            MODEL_BASE_SEQ: 2.541,
            MODEL_LARGE_WIN: 2.449,
            MODEL_LARGE_SEQ: 2.585,
        },
    },
    "animal-farm.txt": {
        MEASURE_ACCURACY: {
            MODEL_BASE_WIN: 0.229,
            MODEL_BASE_SEQ: 0.286,
            MODEL_LARGE_WIN: 0.251,
            MODEL_LARGE_SEQ: 0.289,
        },
        MEASURE_LEVENSHTEIN_CLEAN: {
            MODEL_BASE_WIN: 2.761,
            MODEL_BASE_SEQ: 2.653,
            MODEL_LARGE_WIN: 2.692,
            MODEL_LARGE_SEQ: 2.646,
        },
        MEASURE_LEVENSHTEIN_NOISY: {
            MODEL_BASE_WIN: 2.886,
            MODEL_BASE_SEQ: 2.843,
            MODEL_LARGE_WIN: 2.833,
            MODEL_LARGE_SEQ: 2.820,
        },
    }
}


if __name__ == "__main__":
    fig, axs = plt.subplots(3, 3)
    x = [
        MODEL_BASE_WIN,
        MODEL_BASE_SEQ,
        MODEL_LARGE_WIN,
        MODEL_LARGE_SEQ,
    ]

    for corpus_name, corpus_data in data.items():
        plt.figure(figsize=(10, 5))
        plt.suptitle(corpus_name)

        measure = MEASURE_ACCURACY
        ax = plt.subplot(1, 3, 1)
        ax.set_title(measure)
        plt.ylim(0, 1)
        y = [corpus_data[measure][idx] for idx in x]
        plt.bar(x, y)

        measure = MEASURE_LEVENSHTEIN_CLEAN
        ax = plt.subplot(1, 3, 2)
        ax.set_title(measure)
        plt.ylim(0, 4)
        y = [corpus_data[measure][idx] for idx in x]
        plt.bar(x, y)

        measure = MEASURE_LEVENSHTEIN_NOISY
        ax = plt.subplot(1, 3, 3, sharey=ax)
        ax.set_title(measure)
        plt.ylim(0, 4)
        y = [corpus_data[measure][idx] for idx in x]
        plt.bar(x, y)

        plt.savefig(f"comparison-{corpus_name.strip('.txt')}.pdf")
        plt.clf()
