import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np


def visualize():
    voice = pd.read_csv('./data/voice.csv')
    headers = voice.columns
    xs, y = voice[headers[:-1]], voice['label']
    y = y.replace({'male': 1, 'female': 0})

    # Rescale data before TSNE
    xs_scale = StandardScaler().fit_transform(xs)

    # sk-learn TSNE
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    xs_t = tsne.fit_transform(xs_scale)

    plt.figure()
    plt.scatter(xs_t[np.where(y == 0), 0],
                xs_t[np.where(y == 0), 1],
                marker='o', color='#4286f4',
                linewidth='1', alpha=0.8, label='Male')
    plt.scatter(xs_t[np.where(y == 1), 0],
                xs_t[np.where(y == 1), 1],
                marker='o', color='#ffa544',
                linewidth='1', alpha=0.8, label='Female')

    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title('T-SNE')
    plt.legend(loc='best')
    plt.savefig('./output/tsne.png')

if __name__ == '__main__':
    visualize()
