##### define time 
# dates: START_DATE and END_DATE must both be a Monday (weekday = 0)
START_DATE = '2019-12-30'  # inclusive
END_DATE = '2023-05-01'  # exclusive: download does not include this day 
# number of weeks to aggregate
N_WEEK = 1
# number of lags
N_LAGS = 4

##### define database
DATABASE_NAME = 'horn_africa_forecast'

##### define tables
# GDELT BRZ (raw)
GDELT_ERROR_TABLE = 'gdelt_errors'
GDELT_EVENT_TABLE = 'gdelt_events_brz'
GDELT_EMBED_TABLE = 'gdelt_embed_brz'
# GDELT SLV (processed - intermediate)
GDELT_EVENT_PROCESS = 'gdelt_events_cameo1'
GDELT_EVENT_PROCESS_TABLE = f'{GDELT_EVENT_PROCESS}_slv'
GDELT_EMBED_PROCESS_TABLE = ''
# GDELT GLD (final - for modeling)
GDELT_EMBED_PROCESS_LAG_TABLE = GDELT_EMBED_PROCESS_TABLE.replace('_slv', f'{N_LAGS}lags_gld')
GDELT_TITLE_FILL_TABLE = f'{GDELT_EVENT_PROCESS}_title_fill_{N_WEEK}w_gld'
GDELT_TITLE_CONCAT_TABLE = f'{GDELT_EVENT_PROCESS}_title_concat_{N_WEEK}w_gld'

##### define more tables
# ACLED
ACLED_CONFL_HIST_TABLE = ''

##### define countries
COUNTRY_CODES = ['SU', 'OD', 'ET', 'ER', 'DJ', 'SO', 'UG', 'KE']  
COUNTRY_KEYS = {'SU': 214, 'OD': 227, 'ET': 108, 'ER': 104, 'DJ': 97, 'SO': 224, 'UG': 235, 'KE': 175}

##### define cameo codes 
CAMEO_LST = ['11','14','15','17','18','19','20']  # if no filter, set to None
