# findChurn
Uses [AutoML from H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) to identify customers at risk of attrition by analyzing historical patterns of churn. The result includes:

| customerID | predict | No | Yes |
| --- | --- | --- | --- |
| 3668-QPYBK | Yes | 0.701541 | 0.298459 |
| 7795-CFOCW | No | 0.959560 | 0.040440 |
| 9305-CDSKC | Yes | 0.179969 | 0.820031 |

We will work from a [sample telco customer churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn) from Kaggle. The dataset contains the known, historical outcome in column "Churn." Customers who attrited have a value of "Yes," and customers who did not attrit have a value of "No." Churn will be used as the target (or response field) for the model.

Additional descriptive fields help the machine learning algorithms identify patterns of Churn behavior. These behavior patterns are saved into a model, which can also be thought of as a set of rules that helps classify customers as likely churners or not. The model therefore helps anticipate at-risk customers before they attrit. These descriptive data features include:

* gender
* senior citizen flag
* has a partner
* has dependents
* number of months with the company
* has phone service
* has multiple lines
* type of internet service
* has online security

1. [Prepare the data](https://github.com/jeffmaddocks/findChurn/blob/master/prepare%20data.ipynb). This example leverages a sample telco customer churn dataset from Kaggle. From this starting point, we will split the dataset in two:  one dataset will contain the historical outcomes and the features that describe customers who churn or remain customers; the other dataset will represent current customers, some of whom may churn.

2. [Train & Validate a Classification Model](https://github.com/jeffmaddocks/findChurn/blob/master/build%20churn%20model.ipynb). Once the data is prepared we will train and validate a classification model using the free and open source [H2O python module](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html). This python module provides access to the H2O JVM, as well as its extensions, objects, machine-learning algorithms, and modeling support capabilities, such as basic munging and feature generation. We'll use a technique called [AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2oautoml) to quickly identify the highest-performing model out of many options. 

3. [Score Current Customers](https://github.com/jeffmaddocks/findChurn/blob/master/score%20current%20customers.ipynb). Now that we've built a high-performing classification model and saved it to disk, the final step is to score current customers. The result includes a predicted outcome, and a confidence score. In practice, we would exclude predictions with a low confidence score, and focus customer retention efforts on our most valuable customers who are also most likely to churn.

## Setup

1. Download or clone this project into your working directory.

2. Setting up a virtual environment is a great idea! Let's start by opening a terminal/command prompt, changing directory into our working directory, and using virtualenv to create and activate the virtual environment - here's an example in bash:
    ```
    mkdir env
    virtualenv -p python3 ./env
    source env/bin/activate
    ```
    There's even [instructions for Windows](https://programwithus.com/learn/python/pip-virtualenv-windows)!

3. Install required packages by running pip from the terminal: 
    ```
    pip install -r 'requirements.txt'
    ```

4. Configure your [Kaggle API credentials](https://github.com/Kaggle/kaggle-api).

5. Install OpenJDK. The [installation steps vary based on your operating system](https://openjdk.java.net/install/), but Ubuntu users can install OpenJDK from apt:
    ```
    sudo apt install default-jre
    ```

6. Start JupyterLab: 
    ```
    jupyter-lab
    ```
