# ==== Draw wordcloud ====#
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np


def cloud(fled_score, fled_gene):
    d = {}
    for i in range(len(fled_gene)):
        d[fled_gene[i]] = fled_score[i]

    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    fled_score.sort()
    plt.figure()
    plt.hist(fled_score, bins=20)
    plt.xlabel("Gene importance", fontsize=16)
    plt.ylabel("Population", fontsize=16)
    plt.show()

