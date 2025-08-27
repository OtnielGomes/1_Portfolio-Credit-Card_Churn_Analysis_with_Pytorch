<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="images/logo.jpg" alt="Logo" width="700" height="500">
  </a>

<h3 align="center"> Churn Analysis/Prediction </h3>

  <p align="center">
    Customer churn classification of a credit card service with Pytorch as the main classification model.
    <br />
    <a href="https://github.com/OtnielGomes/1_Portfolio-Credit-Card_Churn_Analysis_with_Pytorch/tree/main/src"><strong>Explore the Docs and Functions »</strong></a>
    <br />
    <br />
    <a href="https://github.com/OtnielGomes/1_Portfolio-Credit-Card_Churn_Analysis_with_Pytorch/tree/main/notebooks">View Notebooks</a>
    ·
    <a href="https://github.com/OtnielGomes/1_Portfolio-Credit-Card_Churn_Analysis_with_Pytorch/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/OtnielGomes/1_Portfolio-Credit-Card_Churn_Analysis_with_Pytorch/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <br>
  <summary>Table of Contents</summary>
  <br/>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#pre-requisites">Pre-requisites</a></li>
        <li><a href="#installation-of-librarys">Installation of librarys</a></li>
      </ul>
    </li>
    <li>
      <a href="#the-project">The Project</a></li>
      <ul>
        <li><a href="#1-business-understanding">1-Business understanding</a></li>
        <li><a href="#2-data-understanding">2-Data Understanding</a></li>
        <li><a href="#3-data-preparation">3-Data Preparation</a></li>
        <li><a href="#4-modeling">4-Modeling</a></li>
        <li><a href="#5-evaluation">5-Evaluation</a></li>
        <li><a href="#6-deployment">6-Deployment</a></li>
      </ul>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<br />

<!-- ABOUT THE PROJECT -->
## About The Project

<br />



## Credit Card Churn Analysis - Prediction
---  

## Project Description 
In this project, I will be working with a dataset provided by **Kaggle**, where I will develop a churn-rate analysis. The goal is to identify the causes and reasons for customer churn from a banking institution in relation to credit card services. After understanding these causes and reasons, some machine learning models will be developed to predict potential customers who will be abandoning the credit card service of this institution. With these predictions, I will seek to develop solutions to prevent or reverse the churn of these customers.  

---  

### CRISP-DM Methodology  
The project will follow the CRISP-DM (*Cross-Industry Standard Process for Data Mining*) framework:  

| **Stage** | **Objective** | **Key Actions** |  
|-----------|---------------|------------------|  
| **1. Business Understanding** | Define the impact of churn prediction on customer retention. | - Identify the causes and possible solutions for the business.<br>- Align metrics with business KPIs. |  
| **2. Data Understanding** | Analyze data structure, quality, and variable relationships. | - Exploratory Data Analysis (EDA).<br>- Outlier and correlation detection. |  
| **3. Data Preparation** | Prepare data for model training. | - Split training and test data.<br>- Remove redundant variables. |  
| **4. Modeling** | Train and compare classical models and neural networks. | - Random Forest/Logistic Regression (baseline).<br>- PyTorch neural network (focus on generalization). |  
| **5. Evaluation** | Validate performance with business-oriented metrics. | - AUC-ROC, Recall, confusion matrix.<br>- Simulate financial impact. |  
| **6. Deployment** | Deploy the model for production use. | - Build a final churn prediction model with customer behavior indicators. |  

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<br/>

## Built With
<br/>

- [![Databricks][Databricks Free]][Databricks Free-url]
- [![Language Python][Python]][Python-url]
- [![Apache][Apache Spark]][Apache Spark-url]
- [![PD][Pandas]][Pandas-url]
- [![NP][NumPy]][NumPy-url]
- [![Matplot][Matplotlib]][Matplotlib-url]
- [![Scipy][Scipy]][Scipy-url]
- [![Torch][PyTorch]][PyTorch-url]
- [![Sklearn][scikit-learn]][scikit-learn-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

<br/>

**Clone the repository**
```sh
git clone https://github.com/OtnielGomes/1_Portfolio-Credit-Card_Churn_Analysis_with_Pytorch
```
<br/>

### Pre-requisites

> 📌 **This entire project was built using Databricks Free Edition**.

---

### 🧠 What is Databricks Free Edition?

**Databricks Free Edition** is the free version of the Databricks platform, designed for **students, educators, developers, and data enthusiasts**.  
It replaces the former *Community Edition* and offers a **serverless** environment with limited resources — ideal for **prototyping, learning, and collaboration**.

With it, you can:
- Create interactive notebooks (Python, SQL, Scala, R)
- Use **Databricks Assistant** for code suggestions and corrections
- Train machine learning models and build data pipelines
- Collaborate in real time with other users

---

### 📝 How to Sign Up

#### 1. Go to:  
[Databricks Free Edition – Microsoft Learn](https://learn.microsoft.com/en-us/azure/databricks/getting-started/free-edition)  
#### 2. Sign in with Google, GitHub, Microsoft, or another supported provider.  
#### 3. A **free workspace** will be automatically created for you.

---

### 🧭 First Steps in the Workspace

### 1. **Workspace**
- Organize your notebooks, scripts, and datasets
- Create folders and set sharing permissions

### 2. **Notebook**
- Interactive interface for writing and running code
- Supports **Python, SQL, R, Scala**

### 3. **Databricks Assistant**
- AI-powered helper that explains, suggests, and fixes code
- Works in notebooks and SQL editor

---

### 🔧 What You Can Do in the Free Edition

| Feature                       | Description                                                                |
|--------------------------------|----------------------------------------------------------------------------|
| Create notebooks               | For data analysis, visualizations, and machine learning                   |
| Query data with SQL            | Explore datasets using the SQL editor                                     |
| Build data pipelines           | Using LakeFlow, Auto Loader, and Delta Live Tables                        |
| Train AI models                | With PySpark, MLflow, and foundation models                               |
| Create interactive dashboards  | With natural language-based visualization (Genie)                         |
| Collaborate in real time       | Share and edit notebooks with your team                                   |

---

### ⚠️ Limitations

- **Personal use only** (non-commercial)
- Limited computing resources (no dedicated clusters)
- Some advanced features unavailable (full Unity Catalog, scheduled jobs)

---

### 📚 Learning Resources

- [Databricks Academy](https://www.databricks.com/learn/free-edition) — free courses on:
  - SQL Fundamentals
  - Data Engineering with Delta Lake
  - Machine Learning with PySpark
- [Official Databricks Documentation](https://docs.databricks.com/)

### Installation of Libraries

The installation of the required libraries is performed using the command:

```python
%pip install '..\requirements.txt'
```

This command is present in the first notebook of this project.

---

💡 **Note**:  
- In Jupyter/Databricks notebooks, the `%pip` magic command installs packages directly into the current environment.  
- If your `requirements.txt` file is located in a subdirectory or at a different path, make sure to update the path accordingly (e.g., `../requirements.txt`).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

<br/>

## The Project

### 1 - Business Understanding  
---

### General Problem Context  
#### What is Churn Rate, and What Are the Solutions to This Problem? 
Many companies struggle with customer churn and often find it challenging to reverse this trend. The metric that measures this scenario is called **churn rate**, which indicates when strategic solutions are needed to address the issue.  

In 2020, Bryce Baer published a guide on churn rate on the [Zendesk website](https://www.zendesk.com.br/blog/customer-churn-rate/?_ga=2.155312252.614584228.1623244699-1365810980.1622555740#) – a company specializing in corporate software development. The guide highlights that businesses implementing strategies to reduce churn can increase their **profitability** by nearly 40%.  

---

#### How to Calculate Churn Rate?
##### Churn Rate Formula:  
$$\text{Churn Rate} = \frac{\text{Number of customers lost during a period}}{\text{Total number of customers at the start of the period}} \times 100$$  

---

#### Impacts of a High Churn Rate
While reducing churn to zero is practically impossible, acceptable rates (4% to 5%) minimize financial impacts. Some companies operate at higher rates (5% to 7%) without significant revenue loss, depending on industry dynamics. **Key factors to define "acceptable" churn**:  
- Industry standards (e.g., SaaS vs. retail).  
- Customer lifetime value (CLV).  
- Customer acquisition cost (CAC).  

---

#### Reasons for Customer Churn
1. **Lack of Perceived Value**:  
   - Occurs when there’s a growing gap between customer expectations and actual delivery. Clear communication about product/service benefits is critical.  
2. **Poor Customer Experience**:  
   - Negative interactions (e.g., bad support, complex processes, product failures) drive churn.  
3. **Competitor Offers**:  
   - Attractive promotions or pricing from competitors can lure customers away.  
4. **Changing Customer Needs**:  
   - Failure to adapt products/services to evolving demands leads to turnover.  

---

## Project Challenge: 
The bank’s manager has observed a rising number of customers abandoning credit card services. Stakeholders aim to:  
1. **Analyze historical data** to identify root causes of churn.  
2. **Develop a machine learning model** to predict customer churn probability.  
3. **Implement strategic actions** to retain high-risk customers.  

---

## KPIs for the Churn Prediction Project:  
1. **Churn Rate**:  
   - *Definition*: Percentage of customers who discontinue credit card services within a specific period.  
   - *Goal*: Reduce this metric through targeted retention strategies.  

2. **Retention Rate**:  
   - *Definition*: Percentage of customers retained after a period.  
   - *Importance*: Directly reflects the success of retention efforts.  

3. **Customer Acquisition Cost (CAC) vs. Retention Cost**:  
   - *Definition*: Ratio of costs to acquire new customers vs. retaining existing ones.  
   - *Insight*: Retention is typically **5-7x cheaper** than acquisition.  

4. **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**:  
   - *Definition*: Measures the model’s ability to distinguish between churners and non-churners.  
   - *Target*: AUC-ROC > 0.90.  

5. **Recall**:  
   - *Definition*: Proportion of actual churners correctly identified by the model.  
   - *Importance*: High recall ensures fewer **false negatives** (missed churners), which is critical because a false negative could result in losing a customer. Retaining existing customers through targeted strategies is significantly cheaper than acquiring new ones.  

---

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### 2 - Data Understanding

---

* This dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc.

---

- **Data file**: - BankChurners.csv

---

- **Target dependent variable**: - 'Attrition_Flag', categorical column with binary classification, i.e. 'Existing Customer'(No-churner) or 'Attrited Customer'(Churner).

---

- **The dataset colleted from kaggle**: https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers?sort=votes&select=BankChurners.csv

---
- **The dataset origin from this site**: https://leaps.analyttica.com/home

---

<br />
<div align="left">
    <img src="images/type_variables.png" alt="Type Variables" width="700" height="500">
  </a>
</div>
<br />



### 3 - Data Preparation
---

- In this step, I will initially divide the training and testing data so that the testing data does not interfere in the analyses, so that the model does not have any bias from the testing data, and only with the training data is it capable of generating good classifications with good generalization.
---
- Next, an EDA will be conducted to verify the data and its main characteristics. In this EDA, the main objective will be to understand the relationship of the data with the churn rate of this banking institution.
---
### Exploratory Data Analysis - EDA

The **Exploratory Data Analysis (EDA)** for this project will be carried out in three main stages: **Univariate Analysis**, **Bivariate Analysis**, and **Multivariate Analysis**.  
The goal is to explore and understand the patterns within the dataset, identifying relationships, trends, and potential insights that can guide the development of effective solutions.

---

#### Univariate Analysis  
- **What it is:** Examines **one variable at a time**, without considering its relationship to others.  
- **Purpose:** Understand the distribution, central tendency, and dispersion of the variable, as well as detect possible outliers.  

---

#### Bivariate Analysis  
- **What it is:** Studies **the relationship between two variables**.  
- **Purpose:** Identify correlations, patterns, or dependencies, and assess how one variable may influence the other.  

---

#### Multivariate Analysis  
- **What it is:** Investigates **three or more variables simultaneously**.  
- **Purpose:** Understand complex interactions and multidimensional patterns, identifying combinations of factors that influence the observed behavior.  

---

### Univariate Analysis 

<br />
<div align="left">
    <img src="images/churn_rate_uni.png" alt="Churn Rate train data" width="700" height="500">
  </a>
</div>
<br />


# 4-Modeling

* **Categorical Variables**: 
  
  * For ordinal categorical variables, I will use the Ordinal Encoder. For nominal variables, the Target Encoder will be applied, with the aim of not increasing the dimensionality of our data, and also considering that this approach brings us more information for our training data because the variables are encoded according to their distributions in relation to the target variable, which in this case will be the **loan_status**.

* **Numerical Variables**: 
  
  * For numerical data, we will apply the normalization technique using MinMaxScaler. This choice aims to preserve the distribution of the data, considering that our distributions are almost all asymmetric, so opting for MinMaxScaler makes more sense in this context.

  * The decision to use MinMaxScaler also takes into account the fact that our data has a significant amount of outliers. However, these outliers are part of the natural distribution of the institution's data. Therefore, the objective of this choice is to allow the model to learn from this data.

* **Machine learning algorithms that will be used in this project**:

  * The basis of this project is a model with PyTorch, but first, to compare and verify the most suitable models for our data, a model will be created in each of the algorithms below:

  * Random Forest Classifier

  * KNN

  * Logistic Regression

  * XGBoost

  * Next, a network will be created in PyTorch. This network will undergo an initial training, in which we will evaluate the metrics and the results obtained.

* **How will the metrics for the evaluation stage be chosen?**:

  * The metrics that will be considered as a parameter to determine the best model and its effectiveness will be **AUC-ROC** together with **Accuracy**. Since we are dealing with binary classes and we have a minority class, which are the loans classified as unpaid, the accuracy of the model would not be enough to determine whether the model converged adequately with its classifications.

  * Therefore, I will be using the **AUC-ROC** metric as the first criterion as a main parameter to determine the effectiveness of the model. This metric takes into account the correct classification of the positive and negative classes, since in a dataset where we have an imbalance in the classes, it is common and natural to expect that the model has a tendency to classify most of the training and testing data with the majority class. Therefore, we will use this metric as a fundamental parameter for the evaluation and analysis of our models, so that we have a good control over false positives and false negatives, in order to reduce them as much as possible.

  * In the background, I will use the accuracy of the model to consider whether there is good predictability of the model with training, validation and testing data.

* **Hypertune | Finetune**:

  * After this initial training, we will **hypertune** the model, adjusting the hyperparameters and the number of neurons in the hidden layers.

  * Later, we will **finetune**, adjusting the learning level per epoch of the model.

* **Test data**

  * Finally, we will move on to the model evaluation stage, in which we will check its performance on the test data.

## Separating features and labels 

```py
  train_data['loan_status'] = train_data['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1}).astype(int)
  test_data['loan_status'] = test_data['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1}).astype(int)
```

```py
  # Train
  X_train = train_data.drop(columns = ['loan_status']) 
  y_train =  train_data['loan_status'].copy()
  
  # Test
  X_test = test_data.drop(columns = ['loan_status']) 
  y_test =  test_data['loan_status'].copy()
```

## Preprocessing

### Categorical Features
```py
  # Ordinal features
  ordinal_features = [
       'term', 'sub_grade', 'expen_cr_inc'  
  ]
  # Manual adjustment
  ordinal_emp_length = ['emp_length']
  
  # Nominal Features
  
  nominal_features = ['home_ownership', 'purpose', 'initial_list_status', 'tot_coll_amt', 'delinq_2yrs', 'pub_rec', 'inq_last_6mths',]
```

### Numerical Features
```py
  num_features = [
    'loan_amnt', 'int_rate', 'dti', 'open_acc', 'revol_util', 'total_acc','tot_cur_bal', 'total_rev_hi_lim', 'real_income', 'ability_to_pay', 'score_cr',   'mo_earliest_cr_line',
  ]
```

### Preprocessor

```py
  # Categorical ordinal
  categorical_ordinal = Pipeline(
      steps = [
          ('ordinal_encoder', OrdinalEncoder()),
          ('min_max_scaler', MinMaxScaler()),
      ]
  )
  # emp_length
  emp_length_ordinal = Pipeline(
      steps = [
          ('ordinal_encoder', OrdinalEncoder(categories = [['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']])),
          ('min_max_scaler', MinMaxScaler()),
      ]
  )
  
  
  # Categorical Nominal
  categorical_nominal = Pipeline(
      steps = [
          ('target_encoder', TargetEncoder(cols = nominal_features)),
          ('min_max_scaler', MinMaxScaler()),
      ]
  )
  
  
  # Column Transformer
  preprocessor = ColumnTransformer(
      transformers = [
          ('ordinal', categorical_ordinal, ordinal_features),
          ('ord_emp_length', emp_length_ordinal, ordinal_emp_length),
          ('target', categorical_nominal, nominal_features),
          ('numerical_features', MinMaxScaler(), num_features),
      ],
      remainder = 'passthrough'
  )
```
## Training models

### Confusion Matrix models

#### Random Forest Classifier

<br />
<div align="left">
  <a href="https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch">
    <img src="images/RFC_matrix.png" alt="RFC Confusion Matrix" width="400" height="300">
  </a>
</div>
<br />

#### KNN

<br />
<div align="left">
  <a href="https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch">
    <img src="images/KNN_matrix.png" alt="KNN Confusion Matrix" width="400" height="300">
  </a>
</div>
<br />

#### Logistic Regression

<br />
<div align="left">
  <a href="https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch">
    <img src="images/LG_matrix.png" alt="LG Confusion Matrix" width="400" height="300">
  </a>
</div>
<br />

#### XGBoost 

<br />
<div align="left">
  <a href="https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch">
    <img src="images/XGboost_matrix.png" alt="XGBoost Confusion Matrix" width="400" height="300">
  </a>
</div>
<br />
<br />


### Pytorch
<br />
<div align="center">
  <a href="https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch">
    <img src="images/Pytorch_matrix.png" alt="Pytorch Confusion Matrix" width="400" height="400">
  </a>
</div>
<br />

## Scores Models

<br />
<div align="left">
  <a href="https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch">
    <img src="images/scores_models.png" alt="Scores Models" width="1000" height="400">
  </a>
</div>
<br />

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# 5-Evaluation

### Considerations on initial training:

At this stage of the project, some models were tested to verify the performance of the data in different algorithms.

* I used the following models:

  * Random Forest Classifier - RFC

  * K-Nearest Neighbors - KNN

  * Logistic Regression – LR

  * XGBoost

  * PyTorch

* From the beginning, the project was based on building a model in **PyTorch** for credit risk classifications, but it makes a lot of sense to test different algorithms to compare them and understand the main characteristics of our data.

* The metric that best fits the resolution of our problem will be **ROC-AUC**, because as it is a dataset with minority classes and binary classification, this metric takes into account the true classifications. In datasets with minority classes, it is natural for the models to present a greater number of false positives and false negatives.

* Still talking about the metric that will be used to evaluate the performance of the models, we can also consider the context of the problem to be solved. Since this is a credit risk analysis model, it is important that both classifications, both the **negative classes: 0 - (loans classified as paid)** and the **positive classes: 1 - (loans classified as unpaid)**, are classified using the same criteria, taking into account that an incorrect classification on both sides can reflect a significant loss for the institution. In the background, the **Accuracy** of each model will also be analyzed, aiming to have a satisfactory and adequate forecast for the loans.

* An initial training was carried out on all the models above to compare and analyze the results. In these initial trainings, I considered the imbalance of the classes in the data set and understood that it would be more appropriate to use the resources of each algorithm to deal with this imbalanced data.

##### Random Forest Classifier - RFC: 
  
* The parameter *class_weight='balanced'* was used to balance the imbalance of classes. 

* The model performed well in both **AUC-ROC** and **Accuracy** metrics, both in the *training and validation data*, with a well-balanced confusion matrix.


##### K-Nearest Neighbors - KNN: 
  
* The parameter *weights='distance'* was used to balance the class imbalance. 

* The model performed well in both **AUC-ROC** and **Accuracy** metrics, *but only in the training data*. 

* The performance in the *validation data* did not produce the same result, indicating a slight **Overfitting**. As a result, the *confusion matrix became unbalanced*, with a very good classification for the positive classes, but poor for the negative classes. 

* In our confusion matrix, we can see that we have a very large number of **false negatives**, which is not good for a model, and an *Accuracy of the negative classes close to 11%*, **which indicates that the model is not capable of making predictions in this class**.

##### Logistic Regression: 

* The parameter *class_weight='balanced'* was used to balance the class imbalance. 

* The model performed well in both **AUC-ROC** and **Accuracy metrics**, both in *training and validation data*, with a well-balanced confusion matrix. However, **Accuracy**, although satisfactory, *was slightly lower compared to the other models*.

##### XGBoost: 

* The *scale_pos_weight* parameter was used to balance class imbalance. 

* The model performed well in both **AUC-ROC** and **Accuracy** metrics, both in *training and validation data*, with a well-balanced confusion matrix.

##### PyTorch: 

* I chose a different approach, using the **DataLoader** feature, a tool responsible for organizing the data into training batches. 

* In the *training data*, the **sampler** function was used, which generates a *specific weight for each class according to its distribution*. In other words, minority classes will have higher weights and majority classes, lower weights. Thus, the **DataLoader** will select the data according to the weights, generating a balance in the assembly of batches for training. For example, a batch of size 32 will have 16 negative classes and 16 positive classes. 

* In the *validation data*, the default configuration was used to assemble the batches, with the parameter *shuffle=True*, where the data is shuffled **randomly**, regardless of the classes. This ensures unbiased validation, allowing an assertive and real analysis of the model's performance on the validation data.

* I also considered using **pos_weight** to generate a balance in the minority classes. The pos_weight is a parameter of **BCEWithLogitsLoss** that adjusts the error penalty according to the weight of the minority classes. However, compared to the two methods, using the sampler in DataLoader demonstrated, in the initial training, *better performance and compatibility with the training and validation data*.

* The model performed well in both **AUC-ROC** and **Accuracy metrics**, both in training and validation data, with a well-balanced confusion matrix.

* Of the trained models, I will be choosing PyTorch, as it obtained a satisfactory AUC-ROC and Accuracy score. I believe that, since it is a neural network, it provides us with several tools so that the model can **generalize** all the data in this training set, facilitating predictions of both test data and future data that may be received by the institution.

* I chose to include the **weight decay** (L2 Regularization) and **dropout** parameters to have a more robust network that is more likely to deal well with the *variation of new data and records*.

* The **weight decay** helps prevent very large weight adjustments, regularizing the values ​​directly. *This prevents overfitting and also makes the model have a more stable adjustment with the training data*.

* The **dropout** adds uncertainty to the training, making the model more robust to data variability, as it randomly turns off a portion of the neurons during the training epochs, forcing the network to not depend specifically on just one neuron,*promoting more distributed learning*.

* To better adjust this model, I will be performing **HyperTuning** to adjust the number of neurons in each layer and the other hyperparameters present in this network, aiming to maintain good generalization of the data and good performance of the metrics mentioned above.

## Hypertunning

```print
    Best trial config: {'l1': 2, 'l2': 64, 'l3': 8, 'lr': 0.0006325676320034128, 'batch_size': 256}
    
    Best trial final validation loss: 0.5790479942864063
    Best trial final validation accuracy: 0.7356786727905273
    Best trial final validation auc_roc: 0.7059769034385681
    
    Best trial test set loss: [0.5630040202157155]
    Best trial test set accuracy: [0.7350901961326599]
    Best trial test set auc_roc:[0.7020368576049805]
```

## Fine Tuning

* In this step, all the parameters obtained in **Hypertuning** will be applied. Then, controlled Fine-Tuning will be applied, which allows fine-tuning of the feature layers with a lower learning rate, while adjusting the classification layers more quickly. In this network, the layers were assembled using PyTorch's **'nn.Sequential'**, which allowed us to partition the network into two parts:

* **Features**: These are the layers responsible for capturing the characteristics of the data.

* **Classifier**: Layer responsible for outputting/classifying the data.

## Pytorch with Test data
#### Confusion Matrix

<br />
<div align="center">
  <a href="https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch">
    <img src="images/test_matrix.png" alt="Test Confusion Matrix" width="400" height="400">
  </a>
</div>
<br />

### Considerations on validating the model on test data:

* At this stage of the project, we finally submitted our model to the test data. We can see that our AUC-ROC and Precision scores were satisfactory compared to the training data scores, and the confusion matrix is ​​balanced. The model presented the following scores:

  * **AUC-ROC**: 71.08%
  * **Accuracy**: 66.42%
  * **F1 Score**: 37.41%
  * **Recall**: 63.87%

#### Why were the classifications more balanced between negative and positive classes in the test data compared to the training data?

* Our training and test data were separated in an orderly manner based on the loan start dates. During Exploratory Data Analysis (EDA), we observed that the test data had less null data, which possibly contributed to some variables having greater relevance with the model's target variable, which is **loan_status**. Another factor that directly contributed to this result was the difference in the proportion of defaulted records between the datasets. The training data has approximately **18.50%** of defaulted records, while the test data has approximately **15.70%**.

* This difference in the training and test data, considering the chronological order of the loans, suggests that, over the years, the institution has improved and reduced the number of defaulted loans. Several factors may have positively contributed to this improvement, such as improved data collection during the loan granting process and better classification of borrowers based on their past experiences.

* Therefore, the model had no difficulty in adapting and generalizing the classifications with the test data, since this data, in addition to presenting better quality in terms of cleanliness and quantity of null data, also presented a lower proportion of defaulting customers compared to the training data.

#### Individual classification of each class:

* **Target 0**: Customers who paid their loans: **66.89% Accuracy** >>> **True Positives**: 32428 X **False Positives**: 16047
* **Target 1**: Customers who did not pay their loans: **63.87% Accuracy** >>> **True Negatives**: 5770 X **False Negatives**: 3264

* The model obtained a considerably higher accuracy in loans that were paid, which leads us to conclude that it has good detection for these loans.

#### Conclusion

* Regarding the accuracy of our model, we can conclude that it is satisfactory, considering the limitations of our dataset. We dealt with a considerable amount of null data in some variables that were crucial to the project. Some variables could not be used due to this problem, and they could have contributed positively to our model. Our numerical data has asymmetric distributions and a significant number of outliers, which, as verified in the EDA, are part of the natural distribution of the data in this set.

* In the next and final phase of this project, we will seek a solution to generate a balance in the classification of our model regarding whether or not to pay the requested loan, with the final decision in the organization based on our analysis and results.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# 6-Deployment

* Considering that the model has a prediction **Accuracy of 66.42%** on the test data and the **ACU-ROC of 71.08%**, during the classification process some limits and parameters will be defined using some variables that were fundamental in our data analysis process.

* In addition to these variables helping us and giving us a broader view of our model's classifications, they will also help us give feedback to our potential client in case of approval or disapproval.

* I will use the training data to define these limits so that there is no leakage with the test data.

* To implement our final model, I will first consider the classification of our model and then consider the following variables to validate these classifications:

* **ability_to_pay**: The loan applicant's ability to pay considering his/her monthly salary x the loan installment.

* **dti**: The borrower's debt ratio.

* **score_cr** : The applicant's loan score based on FICO score criteria.

* **sub_grade** : The risk rating of the loan made by Lending Club.

* **expen_cr_inc** : The rating of the applicant's level of revolving credit utilization.

#### So our classifier will be as follows:

* When the model classifies the loans as **fully paid** and the applicant has the following scores and classifications:

  
  * **ability_to_pay**  < 10%

    * OR

  * **dti** < 15%

    * OR

  * **score_cr** > 700

    * OR

  * **expen_cr_inc** == 'A'

    * AND

  * **sub_grade** in ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5'] # **good_grades**

* This loan will be classified as:

  * **very low risk of default**. 

  * *Therefore it will be automatically approved*

* And if the model classifies the loans as fully paid and the applicant **does not have the scores and classifications above**:

* This loan will be classified as:
   
  * **Low risk of default**. 

   * *Therefore, it will be forwarded for analysis by the manager to be approved or not*

* When the model classifies the loans as **defaulters** and the applicant has the following scores and classifications:

  * **ability_to_pay**  > 10%

    * OR

  * **dti** > 15%

    * OR

  * **score_cr** <= 700

    * OR

  * **expen_cr_inc** == 'B' or 'C' or 'D'

    * AND

  * **sub_grade** in ['E1', 'E2', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5'] # **bad_grades**

* This loan will be classified as:
  
  * **Very high risk of default**. 
  
  * *Therefore, it will be automatically rejected*

* And if the model classifies the loans as as defaulters and the applicant **does not have the scores and classifications above**:

* This loan will be classified as:

  * **Medium risk of default**.   

  * *Therefore, it will be forwarded for analysis by the manager to be approved or not*
 
  <br />
<div align="left">
  <a href="https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch">
    <img src="images/hist_deploy.png" alt="Histogran deploy" width="1000" height="350">
  </a>
</div>
<br />

<br />
<div align="left">
  <a href="https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch">
    <img src="images/count_deploy.png" alt="Count Deploy" width="1000" height="350">
  </a>
</div>
<br />

### Good Grades

<br />
<div align="left">
  <a href="https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch">
    <img src="images/good_grades.png" alt="Good Grades" width="1100" height="400">
  </a>
</div>
<br />

### Bad Grades

<br />
<div align="left">
  <a href="https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch">
    <img src="images/bad_grades.png" alt="Bad Grades" width="1100" height="400">
  </a>
</div>
<br />

# Final Classifier

```py

  def loan_checker(loan_for_class, device = 'cpu'):
      
      # Data preprocessing
      data_preprocessed = preprocessor_loaded.transform(loan_for_class)
      
      # Loading Network Trained
      net = model_loaded.to(device)
  
      # Net to eval
      net.eval()
      # Loading Dataset
      X = torch.from_numpy(data_preprocessed.astype(np.float32))
      X = X.to(device)
      pred = net(X)
      pred = torch.sigmoid(pred).cpu()
      
      probabilities  = pred.item()
  
      # Convert probabilities to binary values ​​(0 or 1) using a threshold
      thershold = 0.5
      binary_pred = (pred >= thershold).int().item()
      
      print(f'\nThis loan has a: {round((probabilities * 100), 2)}% chance of defaulting')
    
      model_prediction = binary_pred
      # expen_cr_inc
      bad_expen = ['D']
      good_expen = ['A']
      expen_cr_inc = loan_for_class['expen_cr_inc'].item()
      # sub_grade
      bad_grades = ['E1', 'E2', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5',]
      good_grades = ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',]
      sub_grade = loan_for_class['sub_grade'].item()
  
      # dti
      dti = loan_for_class['dti'].item()
      # score_cr
      score_cr = loan_for_class['score_cr'].item()
      # ability_to_pay
      ability_to_pay = loan_for_class['ability_to_pay'].item()
  
      if model_prediction == 1:
  
          if (ability_to_pay > 10 or score_cr < 700 or dti > 15 or expen_cr_inc in bad_expen) and (sub_grade in bad_grades):
              print(f'Loan denied! ---  High risk of default loan')
              print(f'\nThis loan has been considered high risk because some of the scores below do not meet the criteria required for loan approval.')
              print(f"\nexpen_cr_inc: {expen_cr_inc} >>>> {'Not OK' if expen_cr_inc in bad_expen else 'OK'}")
              print(f"score_cr: {score_cr} >>>> {'Not OK' if score_cr < 700 else 'OK'}")
              print(f"ability_to_pay: {ability_to_pay} >>>> {'Not OK' if ability_to_pay > 10 else 'OK'}")
              print(f"dti: {dti} >>>> {'Not OK' if dti > 15 else 'OK'}")
              print(f"\n### sub_grade ###: {sub_grade} >>>> {'Not OK' if sub_grade in bad_grades else 'OK'}")
          else:
              print(f'Approval subject to manager analysis! --- Medium risk loan of default')
              print(f'\nThis loan has been considered medium risk because some of the scores below meet the criteria required for loan approval.')
              print(f"\nexpen_cr_inc: {expen_cr_inc} >>>> {'Not OK' if expen_cr_inc in bad_expen else 'OK'}")
              print(f"score_cr: {score_cr} >>>> {'Not OK' if score_cr < 700 else 'OK'}")
              print(f"ability_to_pay: {ability_to_pay} >>>> {'Not OK' if ability_to_pay > 10 else 'OK'}")
              print(f"dti: {dti} >>>> {'Not OK' if dti > 15 else 'OK'}")
              print(f"\n### sub_grade ###: {sub_grade} >>>> {'Not OK' if sub_grade in bad_grades else 'OK'}")
      
  
      if model_prediction == 0:
  
          if (ability_to_pay < 10 or score_cr >= 700 or dti < 15 or expen_cr_inc in (good_expen)) and (sub_grade in(good_grades)):
              print(f'Loan approved! ---  Very low default risk loan')
              print(f'\nThis loan has been deemed very low risk because some of the scores below meet the criteria required for loan approval.')
              print(f"\nexpen_cr_inc: {expen_cr_inc} >>>> {'OK' if expen_cr_inc in good_expen else 'Not OK'}")
              print(f"score_cr: {score_cr} >>>> {'OK' if score_cr >= 700 else 'Not OK'}")
              print(f"ability_to_pay: {ability_to_pay} >>>> {'OK' if ability_to_pay <= 10 else 'Not OK'}")
              print(f"dti: {dti} >>>> {'OK' if dti <= 15 else 'Not OK'}")
              print(f"\n### sub_grade ###: {sub_grade} >>>> {'OK' if sub_grade in good_grades else 'Not OK'}")
          else:
              print(f'Approval subject to manager analysis! --- Low risk of default loan')
              print(f'\nThis loan has been deemed low risk because some of the scores below do not meet the criteria required for loan approval.')
              print(f"\nexpen_cr_inc: {expen_cr_inc} >>>> {'OK' if expen_cr_inc in good_expen else 'Not OK'}")
              print(f"score_cr: {score_cr} >>>> {'OK' if score_cr >= 700 else 'Not OK'}")
              print(f"ability_to_pay: {ability_to_pay} >>>> {'OK' if ability_to_pay <= 10 else 'Not OK'}")
              print(f"dti: {dti} >>>> {'OK' if dti <= 15 else 'Not OK'}")
              print(f"\n### sub_grade ###: {sub_grade} >>>> {'OK' if sub_grade in good_grades else 'Not OK'}")
  
```

### Exemple output

```print
    This loan has a: 46.75% chance of defaulting
    Loan approved! ---  Very low default risk loan
    
    This loan has been deemed very low risk because some of the scores below meet the criteria required for loan approval.
    
    expen_cr_inc: D >>>> Not OK
    score_cr: 716.67 >>>> OK
    ability_to_pay: 11.26 >>>> Not OK
    dti: 27.65 >>>> Not OK
    
    ### sub_grade ###: B2 >>>> OK
```

## Final considerations:

## **We can make the following considerations regarding loan classifications**:

### **Very low risk:** 

* These are loans classified as paid. The selected indicators are in accordance with the rules of our classifier, which allows us to offer better interest rates to this borrower and possibly increase the requested loan amount.

* We have an accuracy of **66.89%** for this class of the model. We can consider the rules determined using the variable **'sub_grade'** as the main parameter in the classifier rules. **'good_grades'** have a probability of default that varies from **3.52% to 16.05%**. Therefore, the chances of loans classified under these terms becoming defaulted are very low.

### **Low risk:** 

* These are loans classified as paid. However, the selected indicators are not in accordance with the rules of our classifier. Therefore, we must consider the borrower's scores and classifications to verify the possibility of approval. In case of approval or not, we have the indicators to justify to our clients the reason for the decision. In cases of approval, we can reduce the amount requested to prevent possible fraud. In case of denial, we can present the indicators that the potential client needs to improve in order to have their loan approved in the future.

* We have an accuracy of **66.89%** for this class of the model. We can consider the rules determined using the variable **'sub_grade'** as the main parameter in the classifier rules. Since these loans are outside the **goods_grades** we can understand that these loans that received this classification need a more careful analysis considering the other indicators.

### **Medium risk:** 

* These are loans classified as defaulted. However, the selected indicators are in accordance with the rules of our classifier. Therefore, we must consider the borrower's scores and classifications to verify the possibility of approval. Whether approved or not, we have the indicators to justify the decision to our clients. In cases of approval, we can reduce the amount requested to prevent possible fraud. In cases of denial, we can present the indicators that the potential client needs to improve in order to have their loan approved in the future.

* We have an accuracy of **63.87%** for this class of the model. We can consider the rules determined using the variable **'sub_grade'** as the main parameter in the classifier rules. Since these loans are outside the **bad_grades** we can understand that these loans that received this classification need a more careful analysis considering the other indicators.

### **Very high risk:** 

* These are loans classified as defaulted, and the indicators indicate that they will probably not be paid. Therefore, these loans will be denied, and we will use our indicators to justify the reasons for non-approval.

* We have an accuracy of **63.87%** for this class of the model. We can consider the rules determined using the **'sub_grade'** variable as the main parameter in the classifier rules. **'bad_grades'** have a default probability ranging from **31.25% to 47.66%**. Therefore, the chances of loans classified under these terms becoming defaulted are very high.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [Notebook-1-EDA](https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch/blob/main/1_Notebooks/0_EDA.ipynb)
- [Notebook-2-Modeling](https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch/blob/main/1_Notebooks/1_Modeling.ipynb)


See the [open issues](https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE.txt`](https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch/blob/main/LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Otniel Gomes - [linkedin.com/in/otnielgomes](https://www.linkedin.com/in/otnielgomes/) - otniel.g.andrade@gmail.com

Project Link: [https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch](https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch.svg?style=for-the-badge
[contributors-url]: https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch.svg?style=for-the-badge
[forks-url]: https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch/network/members

[stars-shield]: https://img.shields.io/github/stars/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch.svg?style=for-the-badge
[stars-url]: https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch/stargazers

[issues-shield]: https://img.shields.io/github/issues/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch.svg?style=for-the-badge
[issues-url]: https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch/issues

[license-shield]: https://img.shields.io/github/license/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch.svg?style=for-the-badge
[license-url]: https://github.com/OtnielGomes/0_Portfolio-Credit_Risk_Analysis_with_Pytorch/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/otnielgomes

[Azure Databricks]: https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=Databricks&logoColor=white
[Azure Databricks-url]:  https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account?icid=databricks

[PyTorch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[PyTorch-url]: https://pytorch.org

[scikit-learn]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/

[Apache Spark]: https://img.shields.io/badge/Apache%20Spark-FDEE21?style=flat-square&logo=apachespark&logoColor=black
[Apache Spark-url]: https://spark.apache.org/

[Pandas]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/

[Matplotlib]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org/

[Scipy]: https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white
[Scipy-url]: https://scipy.org/

[NumPy]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/

[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/

[Databricks Free]: https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=Databricks&logoColor=white
[Databricks Free-url]: https://www.databricks.com/br/learn/free-edition
