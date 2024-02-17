### define time 
# dates: for best results, START_DATE and END_DATE should both be a Monday (weekday = 0)
START_DATE = '2023-01-01'  # inclusive
END_DATE = '2023-05-01'  # exclusive: download does not include this day 
# number of weeks to roll up
N_WEEK = 1
# number of lags
N_LAGS = 4

### define database
DATABASE_NAME = 'horn_africa_forecast'

### define tables
# GDELT BRZ (raw)
GDELT_ERROR_TABLE = 'gdelt_errors'
GDELT_EVENT_TABLE = 'gdelt_events_brz'
GDELT_EMBED_TABLE = 'gdelt_embed_brz'
# GDELT SLV (processed)
GDELT_EVENT_PROCESS = 'gdelt_events_cameo1'
GDELT_EVENT_PROCESS_TABLE = f'{GDELT_EVENT_PROCESS}_slv'
GDELT_EMBED_PROCESS_TABLE = ''
GDELT_TITLE_FILL_TABLE = f'{GDELT_EVENT_PROCESS}_title_fill_{N_WEEK}w_slv'
GDELT_TITLE_CONCAT_TABLE = f'{GDELT_EVENT_PROCESS}_title_concat_{N_WEEK}w_slv'
# GDELT GLD (final)
GDELT_EMBED_PROCESS_LAG_TABLE = GDELT_EMBED_PROCESS_TABLE.replace('_slv', f'{N_LAGS}lags_gld')


### define countries
COUNTRY_CODES = ['SU', 'OD', 'ET', 'ER', 'DJ', 'SO', 'UG', 'KE']  

### define cameo codes 
CAMEO_LST = ['11','14','15','17','18','19','20']  # if no filter, set to None
