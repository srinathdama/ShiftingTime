import pandas as pd
import numpy as np
from scipy.stats import gmean
import os

root_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_base_path = os.path.dirname(root_base_path)
import sys
sys.path
sys.path.append(root_base_path)

model_name = 'KRNO' # 'FNO_residual' ## 'FNO_residual_mean', 'FNO
csv_file_name = 'best_model_metrics.csv'
sort_col_name = 'val_loss_norm' # 'val_nmae' # 'test_nmae'

data = 'Traj' # m4/Cryptos/Traj
seasonal_patterns = 'Monthly'  #Hourly, Daily, Weekly, Monthly, Quarterly, Yearly

if data=='m4':  
    model_dir_name = os.path.join(root_base_path, f'outputs/{data}_{seasonal_patterns}')
    print(f'Reading csv file from model - {model_name}, data - {data}_{seasonal_patterns} ')
else:
    model_dir_name = os.path.join(root_base_path, f'outputs/{data}_')
    print(f'Reading csv file from model - {model_name}, data - {data}_ ')

csv_file_path  = os.path.join(model_dir_name, f'{csv_file_name}')
df = pd.read_csv(csv_file_path)

# df[sort_col_name] = np.sqrt(df[sort_col_name])

sorted_df = df.sort_values(by=sort_col_name, ascending=True, ignore_index=True) 

csv_file_path  = os.path.join(model_dir_name, f'sorted_best_model_metrics.csv')
sorted_df.to_csv(csv_file_path)

print(sorted_df[0:2])


