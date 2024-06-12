#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from wordcloud import WordCloud
import seaborn as sns
import sqlite3


# In[12]:


class integration:
    def __init__(self, csv):
        """
        Contains a dictionary of column names to be replaced for easier reading.
        Loads both the raw data and database.
        """
        try:
            # convert csv to dataframe
            self.df = pd.read_csv(csv, low_memory=False)
            self.conn = None
            
            # default data holder for each table in sqlite
            self.players = None
            self.teams = None
            self.agents = None
            self.matches = None
            self.match_stat = None

            # dictionary of column name
            self.column_names = {
                'match-datetime': 'match_datetime',
                'patch': 'game_patch',
                'map': 'match_map',
                'team1': 'team1',
                'team2': 'team2',
                'team1-score': 'team1_score',
                'team2-score': 'team2_score',
                'player-name': 'p_name',
                'player-team': 't_name',
                'agent': 'agent',
                'rating': 'rating',
                'rating-t': 'rating_attackers',
                'rating-ct': 'rating_defenders',
                'acs': 'average_combat_score',
                'acs-t': 'average_combat_score_t',
                'acs-ct': 'average_combat_score_ct',
                'k': 'kills',
                'k-t': 'kills_attackers',
                'k-ct': 'kills_defenders',
                'd': 'death',
                'd-t': 'death_t',
                'd-ct': 'death_ct',
                'a': 'assist',
                'a-t': 'assist_t',
                'a-ct': 'assist_ct',
                'tkmd': 'total_kills_minus_deaths',
                'tkmd-t': 'total_kills_minus_deaths_t',
                'tkmd-ct': 'total_kills_minus_deaths_ct',
                'kast': 'kill_assist_survive_trade',
                'kast-t': 'kill_assist_survive_trade_t',
                'kast-ct': 'kill_assist_survive_trade_ct',
                'adr': 'round_average_damage',
                'adr-t': 'round_average_damage_t',
                'adr-ct': 'round_average_damage_ct',
                'hs': 'headshot',
                'hs-t': 'headshot_t',
                'hs-ct': 'headshot_ct',
                'fk': 'first_kill',
                'fk-t': 'first_kill_t',
                'fk-ct': 'first_kill_ct',
                'fd': 'first_death',
                'fd-t': 'first_death_t',
                'fd-ct': 'first_death_ct',
                'fkmd': 'kill_minus_deaths',
                'fkmd-t': 'kill_minus_deaths_t',
                'fkmd-ct': 'kill_minus_deaths_ct',
            }
            self.preprocess_split('Unnamed: 0')
        except Exception as e:
            print(e)

    def preprocess_split(self, drop=None):
        """
        Preprocesses the data before splitting it (take note all texts are not lowercase or changed to make a value unique).
        Split the data into tables.
        
        parameters:
            none
        """
        try:
            # remove rows with null values
            self.df.dropna(inplace = True)

            # rename column names
            self.df = self.df.rename(columns=self.column_names)

            # Check the column names after renaming(for debugging)
            # print("Columns after renaming:", self.df.columns)

            # split the data into tables adding primary and reference key
            
            self.players = self.create_table('p_name', 'player', 9143570)
            
            self.teams = self.create_table('t_name', 'team', 10240)
            
            self.agents = self.create_table('agent', 'agent', 463850)
            
            self.matches = self.df[['match_datetime', 'game_patch','t_name', 'match_map','team1_score','team2_score','team1','team2']].drop_duplicates()
            self.matches['match_id'] = self.df['Unnamed: 0'] + 641094570
            
            # merge teams to get team_id1 and team_id2
            self.matches = self.matches.merge(self.teams[['t_name', 'team_id']], left_on='team1', right_on='t_name', how='left')
            self.matches.rename(columns={'team_id': 'team_id1'}, inplace=True)
            
            self.matches = self.matches.merge(self.teams[['t_name', 'team_id']], left_on='team2', right_on='t_name', how='left')
            self.matches.rename(columns={'team_id': 'team_id2'}, inplace=True)
            
            # concatenate the stats to the id of each table
            merge_m_s = self.df.merge(self.matches[['t_name', 'team1_score', 'team2_score', 'team_id1', 'team_id2', 'team1', 'team2', 'match_id']],
              on=['t_name', 'team1_score', 'team2_score', 'team1', 'team2'],
              how='left')
            
            self.matches.drop(columns=['team1','team2','t_name_y','t_name_x','t_name'], inplace=True)
            merge_m_s.drop(columns=['game_patch','team1_score', 'team2_score', 'match_map', 'match_datetime'], inplace=True)
            
            merge_m_p_s = merge_m_s.merge(self.players[['p_name', 'player_id']], on=['p_name'], how='left')
            merge_m_p_s.drop(columns=['p_name'], inplace=True)
            
            merge_m_p_t_s = merge_m_p_s.merge(self.teams[['t_name', 'team_id']], on=['t_name'], how='left')
            merge_m_p_t_s.drop(columns=['t_name','team_id1','team_id2','team1','team2'], inplace=True)
            
            self.match_stat = merge_m_p_t_s.merge(self.agents[['agent', 'agent_id']], on=['agent'], how='left')
            self.match_stat.drop(columns=['agent', 'Unnamed: 0'], inplace=True)
            self.match_stat.dropna(inplace=True)
            self.matches.dropna(inplace=True)
            self.match_stat['player_id'].astype(int)
            self.match_stat['team_id'].astype(int)

            # drop unnecessary columns
            if drop is not None:
                self.df.drop([drop], axis=1, inplace=True)

        except Exception as e:
            print(e)

    def create_table(self, column_name, value_type, id_value):
        """
        Create a table with the required data
        
        parameters:
            column_name - the name of hhe column of both self.df and table to be created
            value_type - the type of data being inputted (i.e. person, team, entity, time)
            id_value - the respective id to be inputted to each value unique
        """
        try:
            # Verify column existence before processing
            if column_name not in self.df.columns:
                raise ValueError(f"Column {column_name} does not exist in DataFrame")

            # Apply the function to the column
            split_df = self.df[[column_name]].drop_duplicates()
            split_df[column_name] = split_df[column_name].apply(self.filter_non_numeric)
            
            # Drop rows where column_name is None
            split_df = split_df.dropna(subset=[column_name])
            
            # Add ID to each value
            split_df['{}_id'.format(value_type)] = [x + id_value for x in range(len(split_df[column_name]))]
            return split_df
        except Exception as e:
            print(e)
            return pd.DataFrame()

    @staticmethod
    def filter_non_numeric(value):
        """
        Filter out numeric values from the data.
        
        parameters:
            value - the value to check
        """
        # Handle the string and numeric values in a column, remove numbers
        if isinstance(value, (int, float)):
            return None 
        try:
            float(value)
            return None
        except ValueError:
            return value
    
    def create_table_load_data(self, db):
        """
        Create table and Load data to database
        
        parameters:
            none
        """
        try:
            # create a cursor
            self.conn = sqlite3.connect(db, timeout=10)
            cur = self.conn.cursor()
            cur.execute('BEGIN')

            # create table in the database
            cur.execute('''
                CREATE TABLE IF NOT EXISTS players(
                	player_id INTEGER NOT NULL PRIMARY KEY,
                	p_name TEXT NOT NULL
                );
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS teams(
                	team_id INTEGER NOT NULL PRIMARY KEY,
                	t_name TEXT NOT NULL
                );
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS agents(
                	agent_id INTEGER NOT NULL PRIMARY KEY,
                	agent TEXT NOT NULL
                );
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS matches(
                	match_id INTEGER NOT NULL PRIMARY KEY,
                	match_datetime TEXT NOT NULL,
                	match_map TEXT NOT NULL,
                	team_id1 INTEGER NOT NULL,
                	team_id2 INTEGER NOT NULL,
                	team1_score INTEGER NOT NULL,
                	team2_score INTEGER NOT NULL,
                	game_patch FLOAT NOT NULL,
                	FOREIGN KEY(team_id1) REFERENCES teams(team_id),
                	FOREIGN KEY(team_id2) REFERENCES teams(team_id)
                );
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS player_match_stats(
                	rating FLOAT NOT NULL,
                	rating_attackers FLOAT NOT NULL,
                	rating_defenders FLOAT NOT NULL,
                	average_combat_score INTEGER NOT NULL,
                	average_combat_score_t INTEGER NOT NULL,
                	average_combat_score_ct INTEGER NOT NULL,
                	kills INTEGER NOT NULL,
                	kills_attackers INTEGER NOT NULL,
                	kills_defenders INTEGER NOT NULL,
                	death INTEGER NOT NULL,
                	death_t INTEGER NOT NULL,
                	death_ct INTEGER NOT NULL,
                	assist INTEGER NOT NULL,
                	assist_t INTEGER NOT NULL,
                	assist_ct INTEGER NOT NULL,
                	total_kills_minus_deaths INTEGER NOT NULL,
                	total_kills_minus_deaths_t INTEGER NOT NULL,
                	total_kills_minus_deaths_ct INTEGER NOT NULL,
                	kill_assist_survive_trade FLOAT NOT NULL,
                	kill_assist_survive_trade_t FLOAT NOT NULL,
                	kill_assist_survive_trade_ct FLOAT NOT NULL,
                	round_average_damage INTEGER NOT NULL,
                	round_average_damage_t INTEGER NOT NULL,
                	round_average_damage_ct INTEGER NOT NULL,
                	headshot FLOAT NOT NULL,
                	headshot_t FLOAT NOT NULL,
                	headshot_ct FLOAT NOT NULL,
                	first_kill INTEGER NOT NULL,
                	first_kill_t INTEGER NOT NULL,
                	first_kill_ct INTEGER NOT NULL,
                	first_death INTEGER NOT NULL,
                	first_death_t INTEGER NOT NULL,
                	first_death_ct INTEGER NOT NULL,
                	kill_minus_deaths INTEGER NOT NULL,
                	kill_minus_deaths_t INTEGER NOT NULL,
                	kill_minus_deaths_ct INTEGER NOT NULL,
                	match_id INTEGER NOT NULL,
                	player_id INTEGER NOT NULL,
                	team_id INTEGER NOT NULL,
                	agent_id INTEGER NOT NULL,
                	FOREIGN KEY(match_id) REFERENCES matches(match_id),
                	FOREIGN KEY(player_id) REFERENCES players(player_id),
                	FOREIGN KEY(team_id) REFERENCES teams(team_id),
                	FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
                );
            ''')
            
            # add to the respective table the values
            self.players.to_sql('players', self.conn, if_exists='append', index=False)
            
            self.teams.to_sql('teams', self.conn, if_exists='append', index=False)
            
            self.agents.to_sql('agents', self.conn, if_exists='append', index=False)
            
            self.matches.to_sql('matches', self.conn, if_exists='append', index=False)
            
            self.match_stat.to_sql('player_match_stats', self.conn, if_exists='append', index=False)
            

            # close and commit all object
            self.conn.commit()
            cur.close()
            self.conn.close()
            
            print('Successfully integrated data to database.')
        except Exception as e:
            cur.close()
            self.conn.close()
            print(e)
            print('Failed to integrate data to database.')


# In[ ]:


class selection:
    def __init__(self):
        self.conn = sqlite3.connect(db, timeout=20)

    def get_data(self, query):
        cur = self.conn.cursor()
        cur.execute('BEGIN')
        try:
            # execute and store data
            cur.execute(query)
            data = cursor.fetchall()

            # close connections
            cur.close()
            self.conn.close()

            # convert and return pandas dataframe
            return pd.DataFrame(data)
            
        except Exception as e:
            # close connections and relay error message
            cur.close()
            self.conn.close()
            print(e)
            print('Failed to select data from database.')


# In[ ]:


class visualization:
    def __init__(self):
        pass

