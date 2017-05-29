import requests
import pandas as pd

HEADERS = {
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en-US,en;q=0.8',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'x-nba-stats-token': 'true',
    'Referer': 'http://stats.nba.com/player/',
    'Connection': 'keep-alive',
    'x-nba-stats-origin': 'stats',
}

def get_shooting(player_id, game_id='', season='2015-16'):
    url = 'http://stats.nba.com/stats/shotchartdetail?Period=0&VsConference='\
    '&LeagueID=00&LastNGames=0&TeamID=0&Position=&Location=&Outcome='\
    '&ContextMeasure=FGA&DateFrom=&StartPeriod=&DateTo=&OpponentTeamID=0'\
    '&ContextFilter=&RangeType=&Season='+season+'&AheadBehind='\
    '&PlayerID='+str(player_id)+'&EndRange=&VsDivision=&PointDiff='\
    '&RookieYear=&GameSegment=&Month=0&ClutchTime=&StartRange='\
    '&EndPeriod=&SeasonType=Regular+Season&SeasonSegment='\
    '&GameID='+str(game_id)+'&PlayerPosition='

    response = requests.get(url, headers=HEADERS)
    while response.status_code != 200:
        response = requests.get(url)
    headers = response.json()['resultSets'][0]['headers']
    data = response.json()['resultSets'][0]['rowSet']
    shots = pd.DataFrame(data, columns=headers)
    shots['SEASON_ID'] = season
    return shots

def get_league_averages(player_id, game_id='', season='2015-16'):
    player_id = int(player_id)
    url = 'http://stats.nba.com/stats/shotchartdetail?Period=0&VsConference='\
    '&LeagueID=00&LastNGames=0&TeamID=0&Position=&Location=&Outcome='\
    '&ContextMeasure=FGA&DateFrom=&StartPeriod=&DateTo=&OpponentTeamID=0'\
    '&ContextFilter=&RangeType=&Season='+season+'&AheadBehind='\
    '&PlayerID='+str(player_id)+'&EndRange=&VsDivision=&PointDiff='\
    '&RookieYear=&GameSegment=&Month=0&ClutchTime=&StartRange='\
    '&EndPeriod=&SeasonType=Regular+Season&SeasonSegment='\
    '&GameID='+str(game_id)+'&PlayerPosition='

    response = requests.get(url, headers=HEADERS)
    while response.status_code != 200:
        response = requests.get(url)
    headers = response.json()['resultSets'][1]['headers']
    data = response.json()['resultSets'][1]['rowSet']
    shots = pd.DataFrame(data, columns=headers)
    shots['SEASON_ID'] = season
    return shots


def get_players():
    url = 'http://stats.nba.com/stats/leagueLeaders?ActiveFlag=No&LeagueID=00'\
    '&PerMode=PerGame&Scope=S&Season=All+Time&SeasonType=Regular+Season'\
    '&StatCategory=PTS'

    response = requests.get(url, headers=HEADERS)
    while response.status_code != 200:
        response = requests.get(url)
    headers = response.json()['resultSet']['headers']
    data = response.json()['resultSet']['rowSet']
    players = pd.DataFrame(data, columns=headers)
    return players

def get_seasons(player_id):
    url = 'http://stats.nba.com/stats/playercareerstats?LeagueID=00'\
    '&PerMode=PerGame&PlayerID=' + str(player_id)

    response = requests.get(url, headers=HEADERS)
    while response.status_code != 200:
        response = requests.get(url)
    headers = response.json()['resultSets'][0]['headers']
    data = response.json()['resultSets'][0]['rowSet']
    seasons = pd.DataFrame(data, columns=headers)
    seasons = seasons['SEASON_ID'].values
    return seasons
