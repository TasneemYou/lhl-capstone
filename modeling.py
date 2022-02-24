from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

numeric_transform = Pipeline([('scaling', StandardScaler()),])

categorical_transform = Pipeline([('one-hot-encode', OneHotEncoder(sparse=False))])

preprocessing_df = ColumnTransformer([('numeric', numeric_transform, num_feats), 
                                    ('categorical', categorical_transform, cat_feats)])

####################################################################
knn = KNeighborsClassifier()

pipeline = Pipeline([('preprocessing', preprocessing_df),
                     #('pca', PCA()),
                     ('model',knn)])

param_grid = {'model__n_neighbors': [5,10,20],
              #'pca__n_components':[5,10,13],
              }

grid_knn = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
grid_knn.fit(X_train, y_train)

#######################################################################

dummy = DummyClassifier(strategy='most_frequent')

pipeline = Pipeline([('preprocessing', preprocessing_df),
                     #('pca', PCA()),
                     ('model',dummy)])

param_grid = {'model__random_state': [5,10,20],
              #'pca__n_components':[5,10,13],
              }

grid_dummy = GridSearchCV(pipeline,param_grid=param_grid, cv=5)
grid_dummy.fit(X_train, y_train)

######################################################################

LR = LogisticRegression()

pipeline = Pipeline([('preprocessing', preprocessing_df),
                     #('pca', PCA()),
                     ('model',LR)])

param_grid = {'model__C': [1,10,100],
              #'pca__n_components':[5,10,13],
              }

grid_LR = GridSearchCV(pipeline,param_grid=param_grid, cv=5)
grid_LR.fit(X_train, y_train)

##################################################################

RF = RandomForestClassifier()

pipeline = Pipeline([('preprocessing', preprocessing_df),
                     #('pca', PCA()),
                     ('model',RF)])

param_grid = {'model__max_depth':[6,10,13],
              #'pca__n_components':[5,10,13],
              }

grid_RF = GridSearchCV(pipeline,param_grid=param_grid, cv=5)
grid_RF.fit(X_train, y_train)

###############################################################

svc = SVC()

pipeline = Pipeline([('preprocessing', preprocessing_df),
                     #('pca', PCA()),
                     ('model',svc)])

param_grid = {'model__random_state': [5,10,20],
              #'pca__n_components':[5,10,13],
              }

grid_svc = GridSearchCV(pipeline,param_grid=param_grid, cv=5)
grid_svc.fit(X_train, y_train)

################################################################

xgb = xgb.XGBClassifier()

pipeline = Pipeline([('preprocessing', preprocessing_df),
                     #('pca', PCA()),
                     ('model',xgb)])

param_grid = {'model__max_depth': [10, 12, 15],
              'model__n_estimators': [50, 100]}


grid_xgb = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
grid_xgb.fit(X_train, y_train)

