from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
import config as CONFIG
import numpy as np

def shot_zone(X,Y):
    '''
    Uses shot coordinates x and y (in feet - divide by 10 if using the shotchart units)
    and returns a tuple with the zone location
    '''
    r = np.sqrt(X**2+Y**2)
    a = np.arctan2(Y,X)*180.0/np.pi
    if (Y<0) & (X > 0):
        a = 0
    elif (Y<0) & (X < 0):
        a = 180
    if r<=8:
        z = ('Less Than 8 ft.','Center(C)')
    elif (r>8) & (r<=16):
        if a < 60:
            z = ('8-16 ft.','Right Side(R)')
        elif (a>=60) & (a<=120):
            z = ('8-16 ft.','Center(C)')
        else:
            z = ('8-16 ft.','Left Side(L)')
    elif (r>16) & (r<=23.75):
        if a < 36:
            z = ('16-24 ft.','Right Side(R)')
        elif (a>=36) & (a<72):
            z = ('16-24 ft.','Right Side Center(RC)')
        elif (a>=72) & (a<=108):
            z = ('16-24 ft.','Center(C)')
        elif (a>108) & (a<144):
            z = ('16-24 ft.','Left Side Center(LC)')
        else:
            z = ('16-24 ft.','Left Side(L)')
    elif r>23.75:
        if a < 72:
            z = ('24+ ft.','Right Side Center(RC)')
        elif (a>=72) & (a<=108):
            z = ('24+ ft.','Center(C)')
        else:
            z = ('24+ ft.','Left Side Center(LC)')
    if (np.abs(X)>=22):
        if (X > 0) & (np.abs(Y)<8.75):
            z = ('24+ ft.','Right Side(R)')
        elif (X < 0) & (np.abs(Y)<8.75):
            z = ('24+ ft.','Left Side(L)')
        elif (X > 0) & (np.abs(Y)>=8.75):
            z = ('24+ ft.','Right Side Center(RC)')
        elif (X < 0) & (np.abs(Y)>=8.75):
            z = ('24+ ft.','Left Side Center(LC)')
    if Y >= 40:
        z = ('Back Court Shot', 'Back Court(BC)')
    return z

def plot_shots_player(shots_player, player_id, season, league_average):
    """
    Create shot chart of individual shots over the season.
    Making distinctions between made and missed shots
    """
    try:
        league_average = league_average.groupby(['SHOT_ZONE_RANGE','SHOT_ZONE_AREA']).sum()
        league_average['FGP'] = 1.0*league_average['FGM']/league_average['FGA']
        player = shots_player.groupby(['SHOT_ZONE_RANGE','SHOT_ZONE_AREA','SHOT_MADE_FLAG']).size().unstack(fill_value=0)
        player['FGP'] = 1.0*player.loc[:,1]/player.sum(axis=1)
        player_vs_league = (player.loc[:, 'FGP'] - league_average.loc[:, 'FGP']) * 100
        x,y = 0.1*shots_player.LOC_X.values, 0.1*shots_player.LOC_Y.values # get players shot coordinates
    except Exception as err:
        print(err)
        return

    # get hexbin to do the hard work for us. Use the extent and gridsize to get the desired bins
    poly_hexbins = plt.hexbin(x,y, gridsize=35, extent=[-25,25,-6.25,50-6.25])
    plt.close()

    # get counts and vertices from histogram
    counts = poly_hexbins.get_array()
    verts = poly_hexbins.get_offsets()

    fig = plt.figure(figsize=(12,11),facecolor='white')
    ax = fig.gca(xlim = [30,-30],ylim = [-10,40],xticks=[],yticks=[],aspect=1.0)
    s = 0.85
    bins = np.concatenate([[-np.inf],np.linspace(-9,9,200),[np.inf]])


    colors = [(0.66, 0.75, 0.66),(0.9,1.0,0.6), (0.8, 0, 0)]
    cm = LinearSegmentedColormap.from_list('my_list', colors, N=len(bins)-1) # create linear color scheme

    # create hexagons
    xy = s*np.array([np.cos(np.linspace(np.pi/6,np.pi*330/180,6)),np.sin(np.linspace(np.pi/6,np.pi*330/180,6))]).T
    b = np.zeros((6,2))

    # adjust size scaling factor depending on the frequency of shots. Size-frequency relationship was choosen empirically
    counts_norm = np.zeros_like(counts)
    counts_norm[counts>=4] = 1 # 4 shots or more= full size hexagon
    counts_norm[(counts>=2) & (counts<4)] = 0.5 # 2 or 3 shots = half size hexagon
    counts_norm[(counts>=1) & (counts<2)] = 0.3 # 1 shot = small hexagon

    # start creating patch and color list
    patches=[]
    colors=[]
    # loop over vertices
    for offc in xrange(verts.shape[0]):
        if counts_norm[offc] != 0:
            xc,yc = verts[offc][0],verts[offc][1] # vertex center
            b[:,0] = xy[:,0]*counts_norm[offc] + xc # hexagon x coordinates
            b[:,1] = xy[:,1]*counts_norm[offc] + yc # hexagon y coordinates
            p_diff = player_vs_league.loc[shot_zone(xc,yc)] # FG% difference for the specific zone
            inds = np.digitize(p_diff, bins,right=True)-1 # convert FG% difference to color index
            patches.append(Polygon(b))
            colors.append(inds)


    # plot all patches
    p = PatchCollection(patches,cmap=cm,alpha=1)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    p.set_clim([0, len(bins)-1])

    # remove axis and ticks
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # No padding
    fig.tight_layout(pad=0)
    fig.patch.set_visible(False)
    ax.axis('off')

    fig.savefig(CONFIG.shots.dir + '/' + str(player_id) + '_' + str(season) + '.jpg', facecolor='white')
    # plt.savefig(CONFIG.plots.dir + '/example_shot_chart.jpg', facecolor='white')
    plt.close()


if __name__ == '__main__':
    players = pd.read_csv(CONFIG.data.dir + '/' + 'players.csv')
    shots = pd.read_csv(CONFIG.data.dir + '/' + 'shots.csv')
    league_averages = pd.read_csv(CONFIG.data.dir + '/' + 'league_averages.csv')

    player_ids = players['PLAYER_ID'].drop_duplicates().values

    # plot shots for each player
    count = 0
    for player_id in player_ids:
        count += 1
        player = players.loc[players.PLAYER_ID == player_id,:]
        print('Plotting %d/%d players' % (count, len(player_ids)))
        # print('Plotting shots from player: %s' % (player['PLAYER_NAME'].values[0]))
        shots_player = shots.loc[shots.PLAYER_ID == player_id,:]
        player_seasons = shots_player.loc[:,'SEASON_ID'].drop_duplicates(inplace=False).values
        for season in player_seasons:
            # print('Plotting shots from season: %s' % (season))
            league_average = league_averages.loc[league_averages.SEASON_ID == season,:]
            shots_player_season = shots_player.loc[shots_player.SEASON_ID == season, :]
            plot_shots_player(shots_player_season, player_id, season, league_average)
