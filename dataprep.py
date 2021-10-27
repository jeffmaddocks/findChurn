import kaggle # https://github.com/Kaggle/kaggle-api
import pandas as pd

kaggle.api.authenticate()
kaggle.api.dataset_download_files('blastchar/telco-customer-churn', unzip=True) # https://www.kaggle.com/blastchar/telco-customer-churn

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

historical_outcomes=df.sample(frac=0.6,random_state=200) #random state is a seed value
current_customers = current_customers=df.drop(historical_outcomes.index)
current_customers = current_customers.drop(columns='Churn')

historical_outcomes.to_json('Telco_Churn_historical_outcomes.json')
current_customers.to_json('Telco_Churn_current_customers.json')
print()