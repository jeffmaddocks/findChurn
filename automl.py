import h2o
import pandas as pd
# import pandas_profiling
import matplotlib.pyplot as plt
import time

from h2o.automl import H2OAutoML
# from IPython.core.display import display, HTML

h2o.init(log_level="INFO")

current_customers = pd.read_json('Telco_Churn_current_customers.json')
historical_outcomes = pd.read_json('Telco_Churn_historical_outcomes.json')

print("data successfully loaded")

history = h2o.H2OFrame(historical_outcomes)
curr = h2o.H2OFrame(current_customers)

all_columns = history.columns

# create a list of column names that should not be used as predictors (IDs, row counts, and the response column)
ignore_columns = ["customerID"]
for i in all_columns:
    if i[0:12] == "Row Count - ":
        ignore_columns.append(i)
print("Ignore Fields: " + str(ignore_columns))

# define the response (target) field
response = "Churn"

print(history.describe()) 

runtime = int(input("How long should we allow the model to build? (in seconds)  "))

# define the predictors (include factors)
predictors = set(all_columns).difference(ignore_columns)
predictors = list(predictors)

for i in predictors:
    if i == response:
        predictors.remove(i)
        
# define training and validation splits
history[response] = history[response].asfactor()  # for binary classification, response should be cast as a factor
train, valid = history.split_frame(ratios=[.8], seed=1234)

# build the model
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
m = H2OAutoML(max_runtime_secs=runtime, max_models=40, seed=5678)

start_time = time.time()
m.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
finish_time = time.time()
print("model build time (seconds): " + str(int(finish_time - start_time)))

# validate model accuracy using the leaderboard
lb = m.leaderboard.as_data_frame()
lb.sort_values(by="auc", ascending=True)
use_models = lb["model_id"].tolist()
use_performance = lb["auc"].tolist()

# iterate thru the model names for readability
for index,item in enumerate(use_models):
    use_models[index] = use_models[index][:use_models[index].replace("_", " ", 1).find("_")]  # trim the model name starting with the second underscore

plt.barh(use_models, use_performance)
plt.xlim(min(use_performance)-.01, max(use_performance)+.002)
plt.title("Model Accuracy")
plt.xlabel("Area Under Curve")
plt.ylabel("Model")
display(HTML("<a href='http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html#supervised'>Model Descriptions</a>"))
plt.show()

m_performance = m.leader.model_performance()
m_performance.plot(type="roc")

print(m.leader.confusion_matrix())

# score current current customers for churn using the highest performing model
predict = m.leader.predict(curr)

# bind the prediction to the original dataset and convert to a dataframe
curr = curr.cbind(predict)
current_customers_df = pd.DataFrame(curr.as_data_frame(), columns=curr.names)
all_columns = current_customers_df.columns
for i in all_columns:
    if i[0:12] == "Row Count - ":
        del current_customers_df[i]
print(current_customers_df.loc[:10,["customerID","predict","No","Yes"]])

current_customers.to_json('scored_customers.json')

h2o.cluster().shutdown()