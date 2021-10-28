# findChurn
Uses [AutoML from H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) to identify customers at risk of attrition by analyzing historical patterns of churn. Organizations use churn predictions to direct interventions and minimize future churn. The process appends three additional columns of data to the customer record:

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

This example is in three parts:

1. [Prepare the data](https://github.com/jeffmaddocks/findChurn/blob/master/prepare%20data.ipynb). We start by spliting the Kaggle dataset in two:  one dataset will contain the historical outcomes and descriptive features used to train the model; the other dataset will represent current customers, some of whom may churn.

2. [Train & Validate a Classification Model](https://github.com/jeffmaddocks/findChurn/blob/master/build%20churn%20model.ipynb). Once the data is prepared we will train and validate a classification model using the free and open source [H2O python module](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html). This python module provides access to the H2O JVM, machine-learning algorithms, feature generation, etc. We'll use a technique called [AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2oautoml) to quickly identify the highest-performing model out of many options. 

3. [Score Current Customers](https://github.com/jeffmaddocks/findChurn/blob/master/score%20current%20customers.ipynb). Now that we've built a high-performing classification model and saved it to disk, the final step is to score current customers. The result includes a predicted outcome and a confidence score. In practice, we would exclude predictions with a low confidence score, and focus customer retention efforts on our most valuable customers who are also most likely to churn.

## Setup

1. Download or clone this project. If you choose to download, setup a working directory first. Alternatively, the "git clone" command will create a directory with the name 'findChurn' that contains this repository:
    ```
    git clone https://github.com/jeffmaddocks/findChurn
    ```

2. Once you have a working directory, setup a virtual environment and activate it  - here's an example in bash:
    ```
    mkdir env
    virtualenv -p python3 ./env
    source env/bin/activate
    ```

3. Install required packages by running pip from the terminal: 
    ```
    pip install -r 'requirements.txt'
    ```

4. Configure your [Kaggle API credentials](https://github.com/Kaggle/kaggle-api).

5. Install OpenJDK. The [installation steps vary based on your operating system](https://openjdk.java.net/install/).

    Ubuntu:
    ```
    sudo apt install default-jre
    ```
    Manjaro:
    ```
    sudo pacman -S jre-openjdk
    ```

6. Start JupyterLab: 
    ```
    jupyter-lab
    ```
