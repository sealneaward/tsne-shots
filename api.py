import requests
import pandas as pd

def get_shooting(player_id, game_id=''):
    url = 'http://stats.nba.com/stats/shotchartdetail?Period=0&VsConference='\
    '&LeagueID=00&LastNGames=0&TeamID=0&Position=&Location=&Outcome='\
    '&ContextMeasure=FGA&DateFrom=&StartPeriod=&DateTo=&OpponentTeamID=0'\
    '&ContextFilter=&RangeType=&Season=2015-16&AheadBehind='\
    '&PlayerID='+str(player_id)+'&EndRange=&VsDivision=&PointDiff='\
    '&RookieYear=&GameSegment=&Month=0&ClutchTime=&StartRange='\
    '&EndPeriod=&SeasonType=Regular+Season&SeasonSegment='\
    '&GameID='+str(game_id)+'&PlayerPosition='

    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    while response.status_code != 200:
        response = requests.get(url)
    headers = response.json()['resultSets'][0]['headers']
    data = response.json()['resultSets'][0]['rowSet']
    shots = pd.DataFrame(data, columns=headers)
    return shots

def get_players():
    url = 'http://stats.nba.com/stats/leagueLeaders?LeagueID=00'\
    '&PerMode=PerGame&Scope=S&Season=2015-16&SeasonType=Regular+Season'\
    '&StatCategory=PTS'

    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    while response.status_code != 200:
        response = requests.get(url)
    headers = response.json()['resultSet']['headers']
    data = response.json()['resultSet']['rowSet']
    players = pd.DataFrame(data, columns=headers)
    return players
