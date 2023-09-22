import pandas as pd
from config import min_time, max_time, minbetodd, maxbetodd, insufficient, threshold

def removeDotFromColumnNames(df):
    df.columns = df.columns.str.replace('.', '', regex=False)
    return df

def dropMinutes(df):
    df = df[(df['framestime'] > min_time) & (df['framestime'] < max_time)]
    return df

def sortByDate(df):
    df['datetimestamp'] = pd.to_datetime(df['datetimestamp'], unit='ms')
    df = df.sort_values(by='datetimestamp')
    return df

def dropNotDraw(df):
  #drop mathces without draw in betting frame
  df['draw'] = df['frameshomescore'] - df['framesawayscore']
  df = df[df['draw'] == 0]
  return df

def oddsFilter(df):
  df = df[(df['frameshomeodd'] >= minbetodd) & (df['frameshomeodd'] <= maxbetodd) & (df['framesawayodd'] >= minbetodd) & (df['framesawayodd'] <= maxbetodd)]
  return df

def addSumStats(df):
    # Compute new columns and hold them temporarily
    sumAstats = df['frameshomeshotsOnTarget'] + df['frameshomeshotsOffTarget'] + df['frameshomeattacks'] + df['frameshomedangerousAttacks']
    sumBstats = df['framesawayshotsOnTarget'] + df['framesawayshotsOffTarget'] + df['framesawayattacks'] + df['framesawaydangerousAttacks']
    # Insert new columns before the last column
    df.insert(loc=len(df.columns) - 1, column='sumAstats', value=sumAstats)
    df.insert(loc=len(df.columns) - 1, column='sumBstats', value=sumBstats)
    return df

def dropInsufficient(df):
  df = df[(df['sumAstats'] >= insufficient) | (df['sumBstats'] >= insufficient)]
  return df

def dif_threshold(df):
  # Calculate the difference between the values in columns 'sumAstats' and 'sumBstats'
  df['diff'] = abs(df['sumAstats'] - df['sumBstats'])
  df = df[df['diff'] >= threshold]
  return df

def dropUnnecessary(df):
    cols_to_drop = ['framestime', 'frameshomescore', 'framesawayscore', 'draw', 'diff', 'sumAstats', 'sumBstats', 'datetimestamp']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    return df
