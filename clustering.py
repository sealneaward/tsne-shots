from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from PIL import Image

import config as CONFIG

def cluster(data):
    """
    Use DBSCAN clustering to form offensive based clustering based on shot charts

    Parameters
    ----------
    data: pandas.DataFrame
        contains t-sne information of encoding for player's shot chart

    Returns
    -------
    player_roles: pandas.DataFrame
        role information for players
    """
    X = data.drop(['player_id', 'season_id'], axis=1, inplace=False)
    X = StandardScaler().fit_transform(X)

    db = DBSCAN(eps=0.11, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    data['cluster'] = labels

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.savefig(CONFIG.plots.dir + '/overall_cluster.png')
    return data

def mean_images(data):
    """
    Get images of each player shot charts and create mean image
    from identified from clusters.

    Parameters
    ----------
    data: pandas.DataFrame

    Returns
    -------
    """
    roles = data['cluster'].drop_duplicates(inplace=False)
    w = 256
    h = 256

    for role in roles:
        arr = np.zeros((w,h,3), np.float)
        role_players = data.loc[data.cluster == role, :]
        n_players = len(role_players)
        for index, player in role_players.iterrows():
            player_id = player['player_id']
            season_id = player['season_id']
            img_name = str(int(player_id)) + '_' + season_id + '.jpg'
            img = Image.open(CONFIG.shots.dir + '/' + img_name)
            img = img.resize((w,h))
            img = np.array(img,dtype=np.float)
            arr = arr+img/n_players

        arr = np.array(np.round(arr),dtype=np.uint8)
        out = Image.fromarray(arr,mode="RGB")
        plt.subplot(1, 3, int(role + 1))
        plt.axis('off')
        plt.imshow(out)

    plt.savefig(CONFIG.plots.dir + '/roles.png')


if __name__ == '__main__':
    data = pd.read_csv(CONFIG.data.dir + '/tsne_shots.csv', header=0)
    data = cluster(data)
    data = data[['player_id', 'season_id', 'cluster']]
    mean_images(data)
    data = data.sort(['player_id', 'season_id'], ascending=[1,1])
    data.to_csv(CONFIG.data.dir + '/overall_cluster.csv', index=False)
    data = data.loc[data.season_id == '2015-16', :]
    data.to_csv(CONFIG.data.dir + '/overall_cluster_players.csv', index=False)
