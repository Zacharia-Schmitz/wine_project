[Presentation Link](https://www.canva.com/design/DAFt-C3lqkg/ksw7j4p6XHH49NVsme6OSA/view?utm_content=DAFt-C3lqkg&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink)

## Predicting Wine Quality for California Wine Institute

### Project goals

- To predict the quality of wine while incorporating unsupervised learning techniques. 


### Project description

- We've been tasked to find drivers of wine quality for the California Wine Institute. 
- We've been given 2 datasets, one for red wine and one for white wine and are variants of the Portuguese "Vinho Verde" wine.


### Initial hypotheses

- Higher amounts of Alcohol present determines the quality
- Volatile acidity content in the wine will affect the quality
- Chlorides content in the wine will affect the quality
- Density has an affect on quality 


### Acquire: 
- Acquire the data from Data.World as CSVs
- You will need to use the winequality_red.csv and winequality_white.csv


### Prepare

- Combined red (1599 Rows / 12 columns) and white (4898 Rows / 12 columns) csv's into one dataframe into a variable.
- Shape after Merge: 6497 Rows, 12 Columns
- Null Check: No Null Values
- Created is_red column for feature engineering in modeling 
- We narrowed the dataframe to 6 Columns to work with
    - Quality, Chlorides, Volatile Acidity, Density, Alcohol, Is_red


### Explore: 

**QUESTIONS OF THE DATA**

- **How does volatile_acidity effect quality?**

- **How does chlorides content effect quality?**

- **How does density effect quality?**

- **How does alcohol content effect quality?**


### Modeling: 

- Use drivers in explore to build predictive models of different types
- Evaluate models on train and validate data
- Select the best model based on accuracy
- Evaluate the test data


### Data dictionary:

| Feature | Dtypes | Definition |
|--------|-----------|-----------|
|Volatile Acidity| float | Acidity measured in grams per liter |
|Chlorides | float | Chlorides in mg per liter |
|Density| float | Measured in grams per liter|
|Alcohol| float | Alcohol by volume|
|**Quality**| **int** | **Quality is a target variable in the dataset, indicating the overall quality or rating of the wine**|
|Is red| int | 1 = red / 0 = white binary|

### How to Reproduce
- Clone this repo (both the CSVs)
- Save both csv's into local folder
- Run notebook

### Key findings 

- RandomForest was the best at very surface level tuning
- It became overfit with more hyperparameters
- Even with only 5 features, it performed very well

### Takeaways and Conclusions
- Ended up using volatile acidity, chlorides, density, alcohol, is_red to predict quality in wines.
- After running 594 Models with various iterations and scalers we determined that the best model was: 
    - Total of 3 Clusters
    - Scaled with StandardScaler
    - Hyperparameters: 
        - n_estimators = 300
        - max_depth = 6
        - min_samples_split = 10
        - min_samples_leaf = 1
- Random Forest Classifier output train accuracy of .601 and validate accuracy of .556
- Test Data Ran through the model returned a test accuracy of .562

### Recommendations
- Recommend running the models to predict quality by keeping the color types of wine split.
    - This could enable for a more fine tuned model in determining quality.
