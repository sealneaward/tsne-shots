from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import config as CONFIG
import numpy as np

def plot_shots_player(shots_player, player_id):
    """
    Create shot chart of individual shots over the season.
    Making distinctions between made and missed shots
    """
    plt.figure(figsize=(12,11))
    made_shots = shots_player.loc[shots.SHOT_MADE_FLAG == 1,:]
    miss_shots = shots_player.loc[shots.SHOT_MADE_FLAG == 0,:]

    plt.scatter(made_shots.LOC_X, made_shots.LOC_Y, c='g', s=50)
    plt.scatter(miss_shots.LOC_X, miss_shots.LOC_Y, c='r', s=50)

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.grid(False)
    plt.tight_layout(pad=0)
    plt.xlim(-250, 250)
    plt.ylim(422.5, -47.5)

    plt.savefig(CONFIG.shots.dir + str(player_id) + '.jpg')
    plt.close()


if __name__ == '__main__':
    players = pd.read_csv(CONFIG.data.dir + 'players.csv')
    shots = pd.read_csv(CONFIG.data.dir + 'shots.csv')

    player_ids = players['PLAYER_ID'].drop_duplicates().values

    # plot shots for each player
    for player_id in player_ids:
        player = players.loc[players.PLAYER_ID == player_id,:]
        print('Plotting shots from player: %s' % (player['PLAYER']))
        shots_player = shots.loc[shots.PLAYER_ID == player_id,:]
        plot_shots_player(shots_player, player_id)
