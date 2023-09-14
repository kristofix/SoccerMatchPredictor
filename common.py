import pandas as pd
from config import min_time, max_time, minbetodd, maxbetodd, insufficient, dif_threshold
def removeDotFromColumnNames(df):
    #df.columns = df.columns.str.replace('.', '')
    df.columns = df.columns.str.replace('.', '', regex=False)

    return df

def dropMinutes(df):
    df = df[(df['framestime'] > min_time) & (df['framestime'] < max_time)]
    df.drop(['framestime'], axis=1, inplace=True)
    return df

def sortByDate(df):
    df['datetimestamp'] = pd.to_datetime(df['datetimestamp'], unit='ms')
    df = df.sort_values(by='datetimestamp')
    return df

def dropNotDraw(df):
  #drop mathces without draw in betting frame
  df['draw'] = df['frameshomescore'] - df['framesawayscore']
  df = df[df['draw'] == 0]
  df.drop(['frameshomescore','framesawayscore'], axis=1, inplace=True)
  #drop 'draw' column
  df.drop(['draw'], axis=1, inplace=True)
  return df

def oddsFilter(df):
  df = df[(df['frameshomeodd'] >= minbetodd) & (df['frameshomeodd'] <= maxbetodd) & (df['framesawayodd'] >= minbetodd) & (df['framesawayodd'] <= maxbetodd)]
  return df

def addSumStats(df):
  df['sumAstats'] = df['frameshomeshotsOnTarget'] + df['frameshomeshotsOffTarget']+ df['frameshomeattacks']+ df['frameshomedangerousAttacks']
  df['sumBstats'] = df['framesawayshotsOnTarget'] + df['framesawayshotsOffTarget']+ df['framesawayattacks']+ df['framesawaydangerousAttacks']
  return df

def dropInsufficient(df):
  #drop records with insufficient value of summary data
  df = df[(df['sumAstats'] >= insufficient) | (df['sumBstats'] >= insufficient)]
  return df

def dif_threshold(df):
  # Calculate the difference between the values in columns 'sumAstats' and 'sumBstats'
  df['diff'] = abs(df['sumAstats'] - df['sumBstats'])
  # Delete rows where the difference is less than dif_threshold
  df = df[df['diff'] >= dif_threshold]
  # Drop the 'diff' column, as it is no longer needed
  df = df.drop(columns='diff')
  return df