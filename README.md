Project Title: Predicting Aircraft Engine Failures Using Sensor Data

Riley Roberts

Boston University

Boston, Massachusetts

rileyr21@bu.edu

**Introduction**

The prediction of Remaining Useful Life (RUL) of aircraft engines is an important predictive maintenance problem. If an engine's health can be determined, its maintenance can be planned to avoid failures and minimize operating costs. In this project, the goal was to develop an engine RUL prediction model in Rust. 
**Data Collection**

The dataset utilized was a publicly available turbofan engine degradation simulation dataset also known as the CMAPSS dataset (Commercial Modular Aero-Propulsion System Simulation). It has run-to-failure data for a number of engines with features including operational conditions and sensor measurements. For the project, only a few features were utilized: Engine ID, cycle number, and three operational conditions. The data was stored in CSV format and read into Rust by manually reading files using standard library functions like BufReader.
**Data Cleaning / Transformation**

Because the dataset was well formatted, cleaning was light. During loading, each row in the CSV file was parsed into a structured Rust object (EngineRecord). There were no missing values because the dataset was complete. For feature transformation, raw data was transformed into numeric matrices with the ndarray crate. Cycle count was normalized against the maximum cycle seen across engines to give a feature between 0 and 1. The feature matrix had the normalized cycle count and the three operating parameters. The target label, RUL, was calculated as the difference between an engine's maximum cycle and its current cycle.
**Data Analysis**

The Random Forest Regressor model was trained for engine RUL prediction. Random Forest is an ensemble learning technique that fits multiple decision trees and returns the average of their predictions, preventing overfitting and enhancing predictive performance. The model was run using the Rust smartcore crate. Smartcore is a set of machine learning algorithms tuned for performance. The feature matrix and label vector for training were transformed into the DenseMatrix format expected by SmartCore. The performance of the model was measured using Root Mean Squared Error (RMSE) and R-squared (R²) score on a separate test set. The final results were: Test RMSE: 18.17, Test R² score: 0.8828. These findings suggest that the model could account for about 88% of the variability in the remaining life of the engine using the features present.

Figure 1: Predicted vs True RUL (found in folder plots)
x-axis: True RUL
y-axis: Predicted RUL
The scatter plot of predicted vs actual RUL values is strongly oriented along the diagonal, indicating that the model predicted very well across the whole range of RUL values. There are slight deviations at the two extreme ends where the prediction errors are comparatively greater. This is common in degradation modeling where the long-term predictions are affected by noise and operating variations.

Figure 2: Cycles vs RUL (found in folder plots)
x-axis: Normalized Cycle Number
y-axis: RUL
The plot of normalized cycle vs. RUL demonstrates a positive linear trend: RUL increases linearly as the normalized cycle number increases. This pattern reflects the structure of the dataset and confirms that cycle count is a strong indicator of RUL in the model’s features.

In general, the results indicate that it is enough to use simple features and a Random Forest model to get good predictive performance on this data. What can be improved is adding more sensor features or utilizing more complicated models for even higher accuracy.

**Discussion & Limitations**
Whereas the Random Forest model achieved success with prediction, several dangers exist. Firstly, a small number of features available were leveraged. Leveraging additional sensor values would likely enhance predictive capabilities. Secondly, no hyperparameter adjustment was implemented on the Random Forest; techniques such as grid search would maximize model performance further. Additionally, Rust also lacks the extensive machine learning and data visualization ecosystem of languages like Python. This project utilized low-level operations and required more direct handling of data types than in higher-level languages.
While this project was able to properly prove the concept behind an RUL prediction model built in Rust, there are significant limitations that must be enumerated. The model was trained against a simplified data set with hardly any engine parameters as features. In actual aerospace use, predictive maintenance solutions would have dozens of sensors and make use of sophisticated flight condition data. Additionally, Random Forest models, although adequate for small-scale issues, may not yield the level of precision necessary for mission-critical aviation systems, which often require deep learning models like LSTMs. Some future directions of work can include enlarging the feature set, experimenting with more complicated models, and optimizing the precision of predictions to match real-world standards of reliability and safety.

**Conclusion & Future Work & Prediction**
This project was able to successfully demonstrate the use of Rust in performing machine learning tasks like aircraft engine remaining useful life prediction. For the CMAPSS dataset, the code attained a high R² with low RMSE through the performance of Random Forest modeling using Rust's smartcore library. Feature engineering is possible to be extended to other sensor variables in the future, and hyperparameter tuning techniques can be employed to further improve model performance. Additionally, employing cross-validation and model ensembling can improve prediction accuracy. Rust's evolving ecosystem means that it will be an increasingly viable option for large-scale machine learning projects in the future.

**References**
•	SmartCore Machine Learning Library. (n.d.). Retrieved from https://smartcorelib.org/ 
•	ndarray - N-dimensional array for Rust. (n.d.). Retrieved from https://docs.rs/ndarray/latest/ndarray/ 
•	Plotters - Rust drawing library. (n.d.). Retrieved from https://docs.rs/plotters/latest/plotters/ 
•	Saxena, Abhinav, and Kai Goebel. “Turbofan Engine Degradation Simulation Data Set.” Kaggle, 2008, https://www.kaggle.com/datasets/behrad3d/turbofan-engine-degradation-simulation-data-set.


