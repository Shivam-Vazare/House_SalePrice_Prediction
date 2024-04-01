# House_SalePrice_Prediction
(PFA. I shared a word file with detail overview of the tasks.)

**EXPLANATION OF MODEL PREDICTION:-**
  Upon evaluating several regression models on the provided dataset, it is evident that the dataset size and the chosen train-test split may have limited the accuracy of the models. Despite these constraints, one model stands out for its relatively superior performance: the Random Forest.
  It's important to note that the performance metrics achieved by the models may not reflect their true potential due to the limited size of the dataset and the simple train-test split employed. Additionally, the absence of missing values and categorical features, as well as the lack of significant correlations between the predictors and the target variable, further constrained the predictive capabilities of the models.
  Nevertheless, the Random Forest demonstrated the best performance among the models evaluated, albeit not achieving optimal accuracy. This model showcases potential promise for further refinement and exploration in larger, more diverse datasets where its strengths may be more pronounced.
	  In summary, while the dataset's limitations hindered the models' accuracy, the Random Forest emerged as the top performer. Further investigation and refinement of this model in more robust datasets may unveil its true capabilities.

**SUGGEST IMPROVEMENTS:-**
•	Increase Dataset Size: Acquire a larger dataset with a more extensive collection of features related to housing attributes. A larger dataset will provide the model with more diverse and representative information, enabling it to learn more robust patterns and relationships.
•	Feature Engineering: Explore additional features or perform feature engineering to create new variables that may better capture the complexities of housing prices. This could include creating interaction terms, polynomial features, or deriving new variables from existing ones.
•	Address Missing Values: Although the provided dataset has no missing values, real-world datasets often contain missing data. Implement strategies such as imputation or advanced techniques like predictive modeling to handle missing values effectively.
•	Consider Additional Features: Introduce new features that may influence house prices, such as neighborhood characteristics, proximity to amenities, school ratings, crime rates, and economic indicators. These additional features can enrich the model's understanding of housing market dynamics.

*“Basically, for better accuracy ,we will need more data and more features, so our model didn't confuse& e get better correlation between dependent and independent features. In given sample dataset there are only 1000 x 10 data, which is not sufficient to predict house prices. Even from the same dataset, we split test train data, so it's obvious model didn't perform well.
By implementing these improvements, we can enhance the accuracy and predictive power of our regression model, enabling more reliable predictions of house sale prices in real-world scenarios.”*

***Tasks Performed To Improve Accuracy :-***
1) Check missing values, outliers and categorical values - Not a single value present
2) Merge old features 'Fullbath' + 'HalfBath' = create new 'TotalBathrooms' feature
3) 'GarageCars' and 'GarageArea' are perfectly correlated with each other so as per the problem statement we need to focus on areas so I drop 'GarageCars' feature'
4) Gaussian Distribution and Logarithmic distribution didn't work becasue features are already uniformly distributed


