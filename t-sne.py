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
    shots = shots.drop(['PLAYER', 'PLAYER_ID'], axis=1, inplace=False)
    vis_data = bh_sne(shots)
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    fig, ax = plt.subplots()
    plt.scatter(vis_x, vis_y, c='blue')
    plt.savefig(CONFIG.plots.dir + 'tsne-shots.svg')
    plt.close()

if __name__ == '__main__':
    shots = pd.read_csv(CONFIG.data.dir + 'total_shots.csv')
    tsne_vis(shots)
