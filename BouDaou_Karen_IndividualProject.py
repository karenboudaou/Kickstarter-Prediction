
####                                     MMA FALL 2024 - INSY 662 individual project - Karen Bou Daou (260957944)

################          hello grader, please go to line 415 to find the grading code with clean preprocessing where you can enter your grading dataset!         ################

### to read through comments and find more details, check out the code below:

####### task 1: classification model

# Load Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Import data
kickstarter_df = pd.read_excel("/Users/kaykaydaou/Desktop/MMA/MMA FALL 24/INSY 662 - data mining and viz/individual project/Kickstarter.xlsx")

# Pre-Processing
kickstarter_df.isnull().sum() #only 'main category' has missing values which is minimal (278 out of 15215) - so i am replacing it with 'unknown'
kickstarter_df['main_category'].fillna('Unknown', inplace=True)
kickstarter_df.describe(datetime_is_numeric=True)
kickstarter_df.duplicated().sum() # there are no duplicated projects
kickstarter_df=kickstarter_df.drop(columns=['name','disable_communication','staff_pick','state_changed_at','pledged', 'spotlight',
                                            'state_changed_at_day','state_changed_at_hr','state_changed_at_month','state_changed_at_yr','state_changed_at_weekday',
                                            'name_len', 'blurb_len','created_at','launched_at','deadline','currency',
                                            'usd_pledged', 'backers_count','staff_pick.1',
                                            'created_at_day','created_at_hr','created_at_month','created_at_weekday','created_at_yr',
                                            'deadline_day','launched_at_day'], errors='ignore')
#dropping name fields => they have a different value for each record so they are not useful in our data mining algorithm
#also dropping disable communication and staff pick as they are unary variable + 'state changed at' and 'pledged' as it refers to post-launch information which we do not need
#dropping name_len and blurb_len as we already have name_len_clean and blurb_len_clean which is more straightforward as it focuses on main keywords
#dropping created at, deadline, launched at, currency since its repetitive, name as it does not affect prediction
#dropping pledged, usd pledged, everything related to state, backers because they introduce introduce future information that aren't realistically available at the point of prediction (when launched)
#also dropping staff pick as its biased and may lead for a project's success and its post-launch too
#dropping everything related to creation as it doesn't matter since everything starts once the project is launched
#dropped deadline day and launched at day as these are very specific to dates and I thought month, years and weekdays would be more insightful than date numbers

kickstarter_df=kickstarter_df.set_index(['id']) #setting the id as index because we wont be using it, its better to use it as project reference
kickstarter_df = kickstarter_df[(kickstarter_df.state == "failed") | (kickstarter_df.state == "successful")] #only keeping failed and successful goals
kickstarter_df["state"].value_counts() #8351 successful and 6112 failed

#feature engineering
kickstarter_df['goal_usd'] = kickstarter_df.goal * kickstarter_df.static_usd_rate #converting goal to usd currency
kickstarter_df = kickstarter_df.drop(columns=['goal', 'static_usd_rate']) #dropping goal and static usd rate because no longer needed

#kickstarter_df['goal_to_backer_ratio'] = kickstarter_df['goal'] / kickstarter_df['backers_count'] #check if the number of backers per goal affects the state
#kickstarter_df['goal_to_backer_ratio'].replace([float('inf'), -float('inf')], 0, inplace=True)
#kickstarter_df['goal_to_backer_ratio'].fillna(0, inplace=True) #in case backers count 0
## not relevant bc we dont have backers at time of launch - removed

#converting the true /false to  0/1
kickstarter_df['show_feature_image'] = kickstarter_df['show_feature_image'].astype(int)
kickstarter_df['video'] = kickstarter_df['video'].astype(int)

#converting week days split into 2 categories - weekday and weekends
def is_weekend(day):
    return 1 if day in ['Saturday', 'Sunday'] else 0
kickstarter_df['deadline_is_weekend'] = kickstarter_df['deadline_weekday'].apply(is_weekend)
kickstarter_df['launched_is_weekend'] = kickstarter_df['launched_at_weekday'].apply(is_weekend)
kickstarter_df = kickstarter_df.drop(columns=['deadline_weekday', 'launched_at_weekday'])

#creating a binary column for US vs. Non-US projects as 57% of the dataset consists of US projects
kickstarter_df['is_us'] = kickstarter_df['country'].apply(lambda x: 1 if x == 'US' else 0)
kickstarter_df = kickstarter_df.drop(columns=['country'])

#set up the variables
#X = kickstarter_df[['launched_at_weekday','deadline_weekday','category', 'main_category', 'country', 'name_len_clean', 'blurb_len_clean','goal_usd','show_feature_image','video']]
X = kickstarter_df.drop(columns=['state'])
y = kickstarter_df['state'].replace({"failed": 0, "successful": 1})

#dummify categorical variables
X_preprocessed = pd.get_dummies(X, columns=['is_us', 'main_category', 'category', 'deadline_is_weekend', 'launched_is_weekend',
                               'deadline_month','deadline_yr','deadline_hr','launched_at_month','launched_at_yr','launched_at_hr'], drop_first=True)
#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.33, random_state=5)

#standardize the features (for models that need it)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, p=2, weights='distance'),
    'Decision Tree': DecisionTreeClassifier(random_state=1,max_depth=3),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=1),
    'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(8), max_iter=1000, random_state=1)
}
#train and evaluate each model
best_model = None
best_accuracy = 0
results = {}
for model_name, model in models.items():
    if model_name in ['Logistic Regression', 'K-Nearest Neighbors', 'Artificial Neural Network']:  #train model and evaluate on the test set
        model.fit(X_train_scaled, y_train)         #models that require standardized data
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)           #models that do not require standardized data
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)         #calculate accuracy and print classification report
    results[model_name] = accuracy
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    if accuracy > best_accuracy:     #update best model if current model's accuracy is higher
        best_accuracy = accuracy
        best_model = model_name

print("Best Model:", best_model)
print("Best Accuracy:", best_accuracy)

#the best model for my features is random forest with accuracy = 78%
#trying to improve my model's performance, i will play with hyperparameters and fine tune my model:
#
# #fine-tune Random Forest since it's the best model
# if best_model == 'Random Forest':
#     hyperparameters = {
#         'n_estimators': [50, 100, 150, 200],
#         'max_features': [3, 4, 5, 6],
#         'min_samples_leaf': [1, 2, 3, 4],
#         'max_depth': [10, 20, 30]}
#     rf = RandomForestClassifier(random_state=1)
#     grid_search_CV = GridSearchCV(estimator=rf, param_grid=hyperparameters, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
#     grid_search_CV.fit(X_train, y_train)
#     best_params = grid_search_CV.best_params_
#     best_score = grid_search_CV.best_score_
#
#     print(f"Best Random Forest model performance (accuracy): {best_score}")
#     print(f"Best combination of hyperparameters: {best_params}")
#
#     #Evaluating on the test set with the best model
#     best_rf_model = grid_search_CV.best_estimator_
#     y_pred_rf = best_rf_model.predict(X_test)
#     test_accuracy_rf = accuracy_score(y_test, y_pred_rf)
#
#     print("Test Accuracy of the best Random Forest model:", test_accuracy_rf)
#     print("Classification Report for Best Random Forest:\n", classification_report(y_test, y_pred_rf))
#
# #fine tuning decreased my accuracy as i got 76% now (2% difference)
# #Best combination of hyperparameters: {'max_depth': 30, 'max_features': 6, 'min_samples_leaf': 2, 'n_estimators': 200}

#FINAL MODEL - random forest - best model
randomforest=RandomForestClassifier(n_estimators=100, random_state=1)
randomforest_model=randomforest.fit(X_train,y_train)
y_test_pred=randomforest_model.predict(X_test)
accuracy_score(y_test,y_test_pred)

##feature importance to see which features affect the model most (best predictors)
feature_importances = randomforest_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns,'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)

#getting the top 20 most important features
top_features = feature_importance_df.head(20)

# Visualize the top 20 features
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.barh(top_features['Feature'], top_features['Importance'], align='center')
plt.gca().invert_yaxis()  # Ensure the most important feature is at the top
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 20 Features by Importance in Random Forest")
plt.show()

##################### task 2: clustering model ########################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

#preprocessing as per task 1, except including state for clustering visualization and getting dummies for less variables as I am performing feature engineering on the rest
X_clustering = kickstarter_df.copy()
X_clustering['state'] = X_clustering['state'].replace({"failed": 0, "successful": 1})

#converting deadline and launch months by seasons (fall, winter, spring, summer) for interpretation
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

X_clustering['deadline_season'] = X_clustering['deadline_month'].apply(get_season)
X_clustering['launch_season'] = X_clustering['launched_at_month'].apply(get_season)

#converting deadline hr and launch hr by AM vs PM for better feature engineering
def get_hour(hour):
    return 'AM' if hour < 12 else 'PM'

X_clustering['deadline_time_period'] = X_clustering['deadline_hr'].apply(get_hour)
X_clustering['launch_time_period'] = X_clustering['launched_at_hr'].apply(get_hour)

#converting launch and deadline years pre covid, during covid, and post covid
def get_covid_period(year):
    if year < 2020:
        return 'Pre-COVID'
    elif year == 2020:
        return 'During-COVID'
    elif year == 2021:
        return 'During-COVID'
    else:
        return 'Post-COVID'

X_clustering['deadline_covid_period'] = X_clustering['deadline_yr'].apply(get_covid_period)
X_clustering['launch_covid_period'] = X_clustering['launched_at_yr'].apply(get_covid_period)

#dropping irrelevant columns
X_clustering = X_clustering.drop(columns=['deadline_month', 'launched_at_month',
                                              'deadline_hr', 'launched_at_hr',
                                              'deadline_yr', 'launched_at_yr'])

#numeric and categorical features
X_cluster_numeric = ['name_len_clean', 'blurb_len_clean', 'goal_usd']
cluster_categorical_features = ['deadline_season', 'launch_season',
                                'deadline_time_period', 'launch_time_period',
                                'deadline_covid_period', 'launch_covid_period','state',
                                'is_us','main_category', 'deadline_is_weekend', 'launched_is_weekend']
#not including category this time as its too broad and to make informative clustering decisions i believe it's best to keep main category for interpretation purposes

#dummifying categorical variables
X_cluster_dummies = pd.get_dummies(X_clustering[cluster_categorical_features], drop_first=True)

#combining numeric and dummy variables
X_cluster_preprocessed = pd.concat([X_clustering[X_cluster_numeric], X_cluster_dummies], axis=1)

#standardizing numeric columns
scaler2 = StandardScaler()
X_cluster_preprocessed[X_cluster_numeric] = scaler2.fit_transform(X_cluster_preprocessed[X_cluster_numeric])

# #performing pca for dimensionality reduction - commented this as PCA was not very insightful and relevant for this dataset given my results
# pca = PCA(n_components=2)
# X_new = pca.fit_transform(X_clustering_scaled)
# explained_variance_ratio = pca.explained_variance_ratio_
#
# # Scree Plot to visualize explained variance by each component
# PC_values = np.arange(len(explained_variance_ratio)) + 1
# plt.figure(figsize=(10, 6))
# plt.plot(PC_values, explained_variance_ratio, marker='o')
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Variance Explained')
# plt.show()
#
# ##scree plot not very insightful - commenting PCA
#
# #K-Means Clustering with PCA-transformed data
# #Elbow Method to find optimal K on PCA data
# withinss = []
# range_clusters = range(2,15)
#
# for k in range_clusters:
#     kmeans = KMeans(n_clusters=k, random_state=5)
#     model = kmeans.fit(X_new)
#     withinss.append(model.inertia_)
#
# # Create a plot
# from matplotlib import pyplot
# pyplot.plot([2,3,4,5,6,7,8,9,10,11,12,13,14],withinss)
# pyplot.title('Elbow Method for Optimal Number of Clusters (PCA)')
# pyplot.xlabel('Number of Clusters')
# pyplot.ylabel('Within-Cluster Sum of Squares (Inertia)')
# pyplot.show()
# #elbow method shows k=4 as the optimal one as the steep declines significantly after k=4
#
# # Silhouette Scores for different K values on PCA data
# for k in range_clusters:
#     kmeans = KMeans(n_clusters=k, random_state=5)
#     labels = kmeans.fit_predict(X_new)
#     silhouette_avg = silhouette_score(X_new, labels)
#     print(f"Silhouette Score for k={k}: {silhouette_avg}")
# #no clear k, as it keeps on increasing
#
# # Calinski-Harabasz Scores for optimal K on PCA data
# for k in range_clusters:
#     kmeans = KMeans(n_clusters=k, random_state=5)
#     labels = kmeans.fit_predict(X_new)
#     f_score = calinski_harabasz_score(X_new, labels)
#     print(f"Calinski-Harabasz Score for k={k}: {f_score}")
# #same for calinski - no clear k
#
# #choosing k = 4 based on elbow method
#
# #clustering using kmeans
# kmeans = KMeans(n_clusters=4, random_state=5)
# pca_labels = kmeans.fit_predict(X_new)
#
# #adding cluster labels to original DataFrame
# X_clustering['Cluster'] = pca_labels
#
# # Distribution of the clusters
# pl = sns.countplot(x=X_clustering["Cluster"])
# pl.set_title("Distribution Of Projects by Clusters")
# plt.show()
#
# # Boxplot for goal USD by cluster
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Cluster', y='goal_usd', data=X_clustering)
# plt.title('Distribution of Goal USD by Cluster')
# plt.show()
#
# # Boxplot for post covid distribution by cluster
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Cluster', y='post_covid', data=X_clustering)
# plt.title('Post-COVID Distribution by Cluster')
# plt.show()
#
# #Look at description instead
# descrip = (X_clustering.groupby('Cluster').describe())
#
# centroids=kmeans.cluster_centers_
# X_clustering['Cluster'] = pca_labels
# cluster_0 = X_clustering[X_clustering['Cluster'] == 0].mean()
# cluster_1 = X_clustering[X_clustering['Cluster'] == 1].mean()
# cluster_2 = X_clustering[X_clustering['Cluster'] == 2].mean()
# cluster_3 = X_clustering[X_clustering['Cluster'] == 3].mean()
#
# #combining my clusters in a single dataframe
# clusters_summary = pd.concat([cluster_0, cluster_1, cluster_2, cluster_3], axis=1)
# clusters_summary.columns = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']


#Elbow Method to find optimal K
withinss = []
range_clusters = range(2,11)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=5)
    k_model = kmeans.fit(X_cluster_preprocessed)
    withinss.append(k_model.inertia_)

# Create a plot
from matplotlib import pyplot
pyplot.plot([2,3,4,5,6,7,8,9,10],withinss)
pyplot.title('Elbow Method for Optimal Number of Clusters')
pyplot.xlabel('Number of Clusters')
pyplot.ylabel('Within-Cluster Sum of Squares (Inertia)')
pyplot.show()
#elbow method shows k=3 as the optimal one as the steep declines significantly after k=3
#the within cluster sum of squares (inertia) drops at k=3

# Silhouette Scores for different K values
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=5)
    labels = kmeans.fit_predict(X_cluster_preprocessed)
    silhouette_avg = silhouette_score(X_cluster_preprocessed, labels)
    print(f"Silhouette Score for k={k}: {silhouette_avg}")

# The silhouette score is highest for k=2 (0.93) but it drops drastically as the number of clusters increases.
# Silhouette Score for k=2: 0.9381025243238422
# Silhouette Score for k=3: 0.12624405490055393
# Silhouette Score for k=4: 0.11188605241516723
# Silhouette Score for k=5: 0.10344508917607728
# Silhouette Score for k=6: 0.09651146612872548
# Silhouette Score for k=7: 0.09254031848074108
# Silhouette Score for k=8: 0.08082417910264761
# Silhouette Score for k=9: 0.08701307438373246
# Silhouette Score for k=10: 0.0844903773291181

# Calinski-Harabasz Scores for optimal K
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=5)
    labels = kmeans.fit_predict(X_cluster_preprocessed)
    f_score = calinski_harabasz_score(X_cluster_preprocessed, labels)
    print(f"Calinski-Harabasz Score for k={k}: {f_score}")

# The Calinski-Harabasz score peaks at k=3 (2240), after which it starts declining steadily as k increases. T
# from both elbow and calinski, i can see that the cluster separation and compactness are best at k=3.

# Calinski-Harabasz Score for k=2: 1975.1292668969995
# Calinski-Harabasz Score for k=3: 2240.7702504383333
# Calinski-Harabasz Score for k=4: 2131.7595967592197
# Calinski-Harabasz Score for k=5: 1829.628415264145
# Calinski-Harabasz Score for k=6: 1657.6513813538959
# Calinski-Harabasz Score for k=7: 1507.6622183688905
# Calinski-Harabasz Score for k=8: 1367.8310375832332
# Calinski-Harabasz Score for k=9: 1274.7313079271094
# Calinski-Harabasz Score for k=10: 1197.4574036579995

kmeans2=KMeans(n_clusters=3)
model1=kmeans2.fit(X_cluster_preprocessed)
labels2=kmeans2.predict(X_cluster_preprocessed)

centroids=kmeans2.cluster_centers_
X_cluster_preprocessed['Cluster']=labels2

cluster_1_pt2=X_cluster_preprocessed[X_cluster_preprocessed['Cluster']==0].mean()
cluster_2_pt2=X_cluster_preprocessed[X_cluster_preprocessed['Cluster']==1].mean()
cluster_3_pt2=X_cluster_preprocessed[X_cluster_preprocessed['Cluster']==2].mean()

clusters_summary=pd.concat([cluster_1_pt2,cluster_2_pt2,cluster_3_pt2],axis=1)
clusters_summary.columns = ['Cluster 0', 'Cluster 1', 'Cluster 2']
clusters_summary


#################################### Grading #############################

# Import Grading Data
kickstarter_grading_df = pd.read_excel("Kickstarter-Grading.xlsx")

# Pre-Process Grading Data
kickstarter_grading_df['main_category'].fillna('Unknown', inplace=True)
kickstarter_grading_df=kickstarter_grading_df.drop(columns=['name','disable_communication','staff_pick','state_changed_at','pledged', 'spotlight',
                                            'state_changed_at_day','state_changed_at_hr','state_changed_at_month','state_changed_at_yr','state_changed_at_weekday',
                                            'name_len', 'blurb_len','created_at','launched_at','deadline','currency',
                                            'usd_pledged', 'backers_count','staff_pick.1',
                                            'created_at_day','created_at_hr','created_at_month','created_at_weekday','created_at_yr',
                                            'deadline_day','launched_at_day'], errors='ignore')

kickstarter_grading_df=kickstarter_grading_df.set_index(['id']) #setting the id as index because we wont be using it, its better to use it as project reference
kickstarter_grading_df = kickstarter_grading_df[(kickstarter_grading_df.state == "failed") | (kickstarter_grading_df.state == "successful")] #only keeping failed and successful goals
kickstarter_grading_df["state"].value_counts()

#feature engineering
kickstarter_grading_df['goal_usd'] = kickstarter_grading_df.goal * kickstarter_grading_df.static_usd_rate #converting goal to usd currency
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['goal', 'static_usd_rate']) #dropping goal and static usd rate because no longer needed

#converting the true /false to  0/1
kickstarter_grading_df['show_feature_image'] = kickstarter_grading_df['show_feature_image'].astype(int)
kickstarter_grading_df['video'] = kickstarter_grading_df['video'].astype(int)

#converting week days split into 2 categories - weekday and weekends
def is_weekend(day):
    return 1 if day in ['Saturday', 'Sunday'] else 0
kickstarter_grading_df['deadline_is_weekend'] = kickstarter_grading_df['deadline_weekday'].apply(is_weekend)
kickstarter_grading_df['launched_is_weekend'] = kickstarter_grading_df['launched_at_weekday'].apply(is_weekend)
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['deadline_weekday', 'launched_at_weekday'])

#creating a binary column for US vs. Non-US projects as 57% of the dataset consists of US projects
kickstarter_grading_df['is_us'] = kickstarter_grading_df['country'].apply(lambda x: 1 if x == 'US' else 0)
kickstarter_grading_df = kickstarter_grading_df.drop(columns=['country'])

# Setup the variables
X_grading = kickstarter_grading_df.drop(columns=['state'])
y_grading = kickstarter_grading_df['state'].replace({"failed": 0, "successful": 1})

#dummify categorical variables
X_grading = pd.get_dummies(X_grading, columns=['is_us', 'main_category', 'category', 'deadline_is_weekend', 'launched_is_weekend',
                                            'deadline_month','deadline_yr','deadline_hr','launched_at_month','launched_at_yr','launched_at_hr'], drop_first=True)
#split the data into training and test sets
X_train_grading, X_test_grading, y_train_grading, y_test_grading = train_test_split(X_grading, y_grading, test_size=0.33, random_state=5)
# note: standardization is not needed for random forest model

# Apply the model previously trained to the grading data
y_grading_pred = randomforest_model.predict(X_test_grading) #using the model 'randomforest_model' developed based on the original data - see line 154 above

# Calculate the accuracy score
accuracy_score(y_test_grading,y_grading_pred)

#extra: feature importance
feature_importances_grading = randomforest_model.feature_importances_
feature_importance_df_grading = pd.DataFrame({'Feature': X_train_grading.columns,'Importance': feature_importances_grading}).sort_values(by='Importance', ascending=False)
print(feature_importance_df_grading)

#plotting the top 20 important features
top_features_grading = feature_importance_df_grading.head(20)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.barh(top_features_grading['Feature'], top_features_grading['Importance'], align='center')
plt.gca().invert_yaxis()  # Ensure the most important feature is at the top
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 20 Features by Importance in Random Forest")
plt.show()
