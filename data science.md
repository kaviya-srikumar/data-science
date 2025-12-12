**DATA SCIENCE**



What is mean by data science?

            collecting a specified data - (insights) processing information

            Collection of amount of data processing, analyzing it and making insights or predictions

            Statistics are mostly used in data science



1st unit

1st day (Data science)

Theoretical over view-Business problem understanding-KPIs

Data cleaning and wrangling -EDA with pandas- numpy with matplotlib(mathematical plotting library)

sea bond

Hands on experience with e-commerce \[the process of buying and selling goods and services over the internet, involving online transactions, payments, and data transfer Eg. Meesho, Myntra, Ajio, Amazon]



Fintech data type(Financial  technology)



KPI(Key Performance Indicator)  in single word measuring a status

  KPI is a measurable value that shows how well a company, a project, a process is performing

  It is a number whether your process(decision) is success(correct) or not



**Uses of KPI**

make goals measurable

help in finding problem

supports decision making

track the process over time



**Types of KPI**

Quantitative KPI (collection of customer reviews)

Qualitative KPI (customer reviews)

Leading KPI (attracts customer)

Lagging KPI (It stores history of kpi)

Operational KPI ()



**BUSINESS PROBLEM**

    A business problem is a challenge or opportunity a company faces which needs data to understand, solve or improve



**DATA ANALYSIS CHECK**: (mostly depends on sales)

sales drop

sales per product

customer buying pattern (eg. Basically timing for the students to go canteen is common for all the years)

competitor price



**DATA CLEANING**(5 types):- (Data preprocessing steps)

Missing values -> Handling missing values and incorrect values(overcome by statistic methods)

Duplicates

Wrong data type

Inconsistent test

Feature engineering



**FEATURE ENGINEERING**

  (Eg. no of customers visited a day, sales increased because of the product selling place (urban) called features which aims to increase the seasonality of sales)

   Features are give high importance to train our model so that we achieve maximum prediction

    Feature engineering is helps to creating better input features for models from raw data.

        numerical raw data (scaling and binding)

        categorical encoding (separate gender wise)

        Day time features(date-month-year , 3x3 total 9 possibilities )

        aggregated features (units consumed by one family)

        Interaction(one data interacts with other data best eg. BMI calculated by height and weight )



**E-COMMERCE DATA**

grocery products

Fashion \& style (dresses)

Home Decor items

Makeup products (including skincare)

Items in cart

Addresses

Liked products

Location

date

receiver phone number

name of the user

reviews and ratings

price

offers

coupon codes

varieties( includes color, size, brand)

tracking data + api (location tracking)

offers

Mode of payment



**EDA(Exploratory Data Analysis)**

How a data get trend, seasonality, varying and deviation

  From fixed data point how the data get deviated(or) varied is known as deviation







**EDA  Libraries**

EDA with Panda

EDA with numpy

EDA with matplotlib

EDA with seaborn



**DAY-2**



Home made Bakes

 can't appoint people for delivery(in order to reduce more expenses) as they have to take care of their own orders

 Fake orders



Statistic method > Mean, median(centre value), mode(repeated values)



  |                /

  |              /

  |            /

10|           |

8  |      /| |

4  |    /  | |

2  |  /    |/

0  |\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_







peak value for short period of time(trend)

long period of time(seasonality)



Matplotlib(pictorial representation):

 1.Line plot

 2.Dotplot

 3.Scatter plot

 4.Bar graph



Legends are small square boxes used to denote different varieties(For eg. in  a map where box in the bottom right corner represents land, water and mountains )



IDE (Integrator Development Environment)



**Characteristics**

1. Systematic process involves repeatable framework from problem define to model deployment
2. data-centric requires quality and accuracy based
3. Iterative nature



**STEPS INVOLVED IN MODEL BUILDING (machine learning model work flow)**

1. Problem definition -> Explain the problem
2. Data collection -> collecting data required for the solution
3. Pre processing -> cleaning data, handle missing values and address outliers
4. Feature engineering -> create meaningful variables that enhance model prediction
5. Model selection -> Algorithms based on problem type and data characteristics
6. Training -> Fit the model to training data and optimise parameter
7. Evaluation -> Performance using validation sets and relevant metrics
8. Deployment -> Integrate the model into production environments
9. Monitoring -> Model performance and detect degradation over time





\[Eg. Training a model is consider as a baby, problem would be help the child to learn things, collecting pics, real apple, 2nd step  clear the damaged one for good learning 3rd one and name it with some variables in our case it would be "APPLE" will be 4th step, 5th step selecting the baby to be learned(trained), next teaching(training) 6th step, testing the baby for memory 7th step, 8th step Allow the baby to wander over environments and finally  check whether the child could find the apple in the environment.]



Day-3

Discussion about the steps 5,6,7 deeply

Machine Learning Models

1. Regression

          1.    Linear

          2.   Logistic

2\.   Classification

          1.  Binary Classification

          2.  Tree Based classification



REGRESSION

Regression defined as the models that predicts continuous numeric outcomes by learning relationship between the input features and output.

 It is continuous predictions

|               /

|\_\_ \_\_  /

|         /|

|      /   |

|   /      |

|/\_\_\_\_|\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_



*y=mx+c*

*y=logx+x*  where m, log are slope



Measurable quantity:

* Time
* Weight
* Money



For eg. Let us consider Dominos they have been started the business at 2020 if their profit at 2021 is 100k then 2022 is 200k ,2023 is 300k like this progressively increasing models will be trained to understand the logic behind the growth it should make algorithmic logic to find next year's(i.e 2024) profit.



Regression types:

1. Linear Regression

                                           y=B0+B1X1+BnXn    where B = beta

2\.  Ridge Regression \[L2] -> regularization

                                           Loss = summation(y-y^2)+lambda(summation(B^2))

3\.  Lasso Regression \[L1] -> regularization

                                           Loss =  summation(y-y^2)+lambda(summation(Modulus(B))



Decision Tree in Classification Models

   A key structure model splitting data into subsets based on conditions are variables

Random forest classifier -> Multiple decision tree( in which different decision tree are trained) are grouped together called ensemble



**ARTIFICIAL INTELLIGENCE:**

  Man-made thinking power

  If we write a program it will take a certain input and give output and then program ends. but artificial intelligence make outputs for infinite set of inputs

**MACHINE LEARNING:**

making the machine to learn the certain patterns

**DATA SCIENCE:**

It is a multi-disciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.



Combination of AI,ML,DS id DL (Deep Learning)



**DEEP LEARNING:**

 Deep Learning is a specialized subset of Machine Learning that utilizes Deep Neural Networks to solve complex problems, especially those involving unstructured data (images, text, audio)



**NEURAL NETWORK:**

A Neural Network ( Artificial Neural Network or ANN) is a computational model inspired by the structure and function of the human brain. It is an algorithm composed of interconnected nodes (called artificial neurons) arranged in layers.

  i. Neuron

       A Neuron is the fundamental computational unit within a neural network. It's often called a node or perceptron.

 ii. Hidden layer

      A Hidden Layer is any layer of neurons in an ANN that is neither the input layer (where the data enters) nor the output layer (where the prediction exits).

 iii. Backtracking Layer

      Backpropagation is the mathematical method a neural network uses to distribute the error it made on a prediction back across all of its internal connections (weights) so it knows exactly how to adjust itself to perform better next time.





**MODEL BUILDING**

*Two libraries SK-Learn (or SciKit Learn) and Tensor Flow \[contains classifier and regression models]*

Data can be given in the form of csv, image which is divided as per rule division:

70%- 30%   where, 70 - Training, 30 - Testing

80% - 20%  where, 80 - Training, 20 - Testing

using the libraries we could separate model

     X           Y

x\_train   y\_train

x\_test    y\_test

Using oops concept

model is called using object

For eg. model = Random Forest()

            model.train

            model.fit



**LIBRARIES:**

1. numpy for *numerical data*
2. pandas for *tabulation*
3. matplotlib.pyplot for *Basic plotting, Customization, Multiple plot types - Supports 2D plots, 3D plots, Figure management - Organize multiple plots in subplots and figures, Output formats - Save plots as PNG, PDF, SVG, and other formats, Low-level control - Fine-grained control over every aspect of a plot*
4. seaborn for *Statistical visualization - Easily create plots that show statistical relationships in data*
5. 

*Enhanced aesthetics - Provides beautiful, publication-ready plots with better default styling than matplotlib*

*Simplified syntax - Makes complex visualizations easier to create with fewer lines of code*

*Data exploration - Great for exploratory data analysis (EDA) with functions like:*

*sns.heatmap() - correlation matrices*

*sns.boxplot() - distribution comparisons*

*sns.scatterplot() - relationships between variables*

*sns.histplot() - distributions*

*sns.barplot() - categorical data*

4\.	import warnings for *rectify temporal warnings from OS*



raw-data -> load using pandas which is concatenation occurs using concat

data.describe().T -> for describing the numerical values



Environment - space for the project to develop

Virtual Environment is created for particular project to have everything that required to develop the project

GIT:

  Version control system used for managing versions of the software or code

          --> GIT CL1   -->GIT Bash



*  Git is a tree like structure. It runs locally on computer, helps to track changes in source code and works offline.

 It also helps developers to collaborate with different versions of files.

* If someone commits the other person's error code then he/she would admit the blame
* GitHub helps in cloud-based hosting service for git repositories. Works on the internet and allow collaboration with people across the world. It is an application for educational purpose not everybody use it for their own job works

In simple words, GitHub where we stores and shares git repositories



Basic commands

--> Push

--> Commit

--> Pull

--> Blame



    Tree structure that stores the code of the software





 Origin/main

   |\_B1

   |\_ B2



**DATA VISUALIZATION**

Representation of processed and cleaned data in a graphical or pictorial representation



**POWER BI**

  power BI is used for visualization and report



**SQL(*Structured Query Language*)**

**two types**

&nbsp;    /\\
   /    \\

DDL  DML

 

DDL - Data definition Language(to define the data(tables, rows and columns)

DML - Data Manipulation Language(used to change the data)



--> Primary Key (initialize defined keys(unique))

--> Foreign Key

