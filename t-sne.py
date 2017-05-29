from tsne import bh_sne
import pandas as pd
import config as CONFIG
import matplotlib.pyplot as plt

def tsne_vis(shots):
    """
    Create t-SNE visualizations of players based on shots

    Parameters
    ----------
    shots: pandas.DataFrame
        all types of shots and associate percentages for each player

    Returns
    -------
    """
    data = shots.drop(['player_id', 'season_id'], axis=1, inplace=False)
    vis_data = bh_sne(data)
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    labels = shots['season_id'].str[-2:].astype(int)
    # labels = range(len(labels))
    shots = shots.loc[:, ['player_id', 'season_id']]
    shots['tsne_x'] = vis_x
    shots['tsne_y'] = vis_y
    shots.to_csv(CONFIG.data.dir + '/tsne_shots.csv', index=False)

    fig, ax = plt.subplots()
    plt.scatter(vis_x, vis_y, c=labels)
    plt.savefig(CONFIG.plots.dir + '/tsne-shots.svg')
    plt.close()

if __name__ == '__main__':
    shots = pd.read_csv(CONFIG.data.dir + '/encoded_data.csv')
    # shots = shots.loc[shots.season_id == '2015-16',:]
    tsne_vis(shots)
