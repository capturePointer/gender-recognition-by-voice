import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def pearson_heatmap():
    df = pd.read_csv('./data/voice.csv')
    pearson_correlation_matrix = df.iloc[:, :-1].astype(float).corr()

    colormaps = 'RdYlBu', 'viridis', 'hsv', 'coolwarm'
    plt.figure(figsize=(15, 15))
    plt.title('Pearson Correlation', y=1.05, size=30)
    sns.heatmap(pearson_correlation_matrix,
                linewidths=.1, square=True, annot=True, cmap=colormaps[0])
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.savefig('./output/heatmap')


if __name__ == '__main__':
    pearson_heatmap()
