from __future__ import print_function

import api
import pandas as pd
import config as CONFIG
import os
import numpy as np

# turn warning off
pd.options.mode.chained_assignment = None

tracked_seasons = [
    '1996-97',
    '1997-98',
    '1998-99',
    '1999-00',
    '2000-01',
    '2001-02',
    '2002-03',
    '2003-04',
    '2004-05',
    '2005-06',
    '2006-07',
    '2007-08',
    '2008-09',
    '2009-10',
    '2010-11',
    '2011-12',
    '2012-13',
    '2013-14',
    '2014-15',
    '2015-16',
    '2016-17'
]

shots_cols = [
            'GRID_TYPE', 'GAME_ID', 'GAME_EVENT_ID', 'PLAYER_ID',
            'PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME', 'PERIOD',
            'MINUTES_REMAINING', 'SECONDS_REMAINING', 'EVENT_TYPE',
            'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA',
            'SHOT_ZONE_RANGE', 'SHOT_DISTANCE', 'LOC_X', 'LOC_Y',
            'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG', 'GAME_DATE', 'HTM',
            'VTM'
       ]

def create_dirs():
    """
    Create necessary directories

    Parameters
    ----------

    Returns
    -------
    """
    if not os.path.exists(CONFIG.data.dir):
        os.makedirs(CONFIG.data.dir)

    if not os.path.exists(CONFIG.plots.dir):
        os.makedirs(CONFIG.plots.dir)

    if not os.path.exists(CONFIG.img.dir):
        os.makedirs(CONFIG.img.dir)

    if not os.path.exists(CONFIG.hdf5.dir):
        os.makedirs(CONFIG.hdf5.dir)

    if not os.path.exists(CONFIG.shots.dir):
        os.makedirs(CONFIG.shots.dir)

    if not os.path.exists(CONFIG.vae_shots.dir):
        os.makedirs(CONFIG.vae_shots.dir)

def create_shot_frame(action_types, initialize_frame):
    """
    Create shot data frame for organization

    Parameters
    ----------
    action_types: str list
        list of different shot types
    initialize_frame: bool
        make empty row of zeroes

    Returns
    -------
    shots: pandas.DataFrame
        empty data frame with columns to populate from organized shots
    """
    org_cols = ['PLAYER_ID', 'PLAYER']
    for action_type in action_types:
        name = action_type
        name_fga = action_type + ' FGA'
        name_fgm = action_type + ' FGM'
        name_fg = action_type + ' FG'

        org_cols.extend([name_fga, name_fgm, name_fg])

    if initialize_frame:
        empty_row = np.zeros(len(org_cols))
        shots = pd.DataFrame(columns=org_cols, data=[empty_row])
    else:
        shots = pd.DataFrame(columns=org_cols)

    return shots

def organize_shots(shots, action_types, player, player_id):
    """
    Create dataframe that represents totals for each type of shot for a player

    Parameters
    ----------
    shots: pandas.DataFrame
        dataframe of each single shot
    action_types_dict: dict
        dictionary of action type values to map to shots
    player: str
        name of player
    player_id: int
        id of player

    Returns
    -------
    org_shots: pandas.DataFrame
        representation of all shots for each type of shot
    """
    org_shots = create_shot_frame(action_types, initialize_frame=True)

    for action_type in action_types:
        shots_action = shots[shots['ACTION_TYPE'] == action_type]
        fg_attempt = len(shots_action)
        fg_made = len(shots_action[shots_action['SHOT_MADE_FLAG'] == 1])

        if fg_attempt > 0:
            fg_pct = float(fg_made)/float(fg_attempt)
        else:
            fg_pct = 0

        # append shot type to organized shot frame
        name = action_type
        name_fga = action_type + ' FGA'
        name_fgm = action_type + ' FGM'
        name_fg = action_type + ' FG'
        org_shots[name_fga] = fg_attempt
        org_shots[name_fgm] = fg_made
        org_shots[name_fg] = fg_pct

    org_shots['PLAYER'] = player
    org_shots['PLAYER_ID'] = player_id

    return org_shots

if __name__ == '__main__':
    create_dirs()

    # get players
    players = pd.read_csv(CONFIG.data.dir + '/players.csv')

    # get shots per player
    shots = pd.DataFrame(columns=shots_cols)
    for index, player in players.iterrows():
        player_id = player['PLAYER_ID']
        seasons = api.get_seasons(player_id)
        for season in seasons:
            if season not in tracked_seasons:
                continue

            print('Collecting shots for player: %d during season: %s' % (player_id, season))
            player_shots = api.get_shooting(player_id=player_id, season=season)
            if player_shots.empty:
                continue
            shots = shots.append(player_shots)
    shots.to_csv(CONFIG.data.dir + '/' + 'shots.csv', index=False)

    # get league averages for each season
    shots = pd.read_csv(CONFIG.data.dir + '/' + 'shots.csv')
    league_averages = pd.DataFrame(columns=['SEASON_ID','GRID_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE','FGA', 'FGM', 'FG_PCT'])
    for season in tracked_seasons:
            print('Gettin league average for season: %s' % (season))
            player_in_season = shots.loc[shots.SEASON_ID == season,'PLAYER_ID']
            if player_in_season.empty:
                continue
            else:
                player_in_season = player_in_season.drop_duplicates(inplace=False).values[0]
            league_average = api.get_league_averages(player_id=player_in_season, season=season)
            league_averages = league_averages.append(league_average)
    league_averages.to_csv(CONFIG.data.dir + '/' + 'league_averages.csv', index=False)

    # shots = pd.read_csv(CONFIG.data.dir + 'shots.csv')
    # action_types = shots['ACTION_TYPE'].drop_duplicates(inplace=False).tolist()
    #
    # # organize shots per player
    # org_shots = create_shot_frame(action_types, initialize_frame=False)
    # for index, player in players.iterrows():
    #     player_id = player['PLAYER_ID']
    #     player = player['PLAYER']
    #     player_shots = shots[shots['PLAYER_ID'] == player_id]
    #     shots_player = organize_shots(player_shots, action_types, player, player_id)
    #     org_shots = org_shots.append(shots_player)
    #
    # org_shots.to_csv(CONFIG.data.dir + 'total_shots.csv', index=False)
