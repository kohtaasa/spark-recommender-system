import sys
import numpy as np
import time
import json
from xgboost import XGBRegressor
from pyspark import SparkContext


def get_args():
    """
    Get command line arguments
    :return: folder_path, test_path, output_path
    """
    folder_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]
    return folder_path, test_path, output_path


folder_path, test_path, output_path = get_args()
# folder_path = 'data'
# test_path = 'data/yelp_val.csv'
# output_path = 'data/model_based_pred.csv.csv'

start_time = time.time()
sc = SparkContext('local[*]', 'model_based')

# Load train data
rdd_train = sc.textFile(folder_path + '/yelp_train.csv')
header_train = rdd_train.first()
rdd_train_clean = rdd_train.filter(lambda l: l != header_train).map(lambda l: l.split(','))

# Load validation data
rdd_validation = sc.textFile(test_path)
header_validation = rdd_validation.first()
rdd_validation_clean = rdd_validation.filter(lambda l: l != header_validation).map(lambda l: l.split(','))

# Load user features
user_rdd = sc.textFile(folder_path + '/user.json').map(lambda x: json.loads(x))
user_features = user_rdd.map(lambda x: (x['user_id'], (x['review_count'], x['average_stars'], x['useful'], x['funny'], x['cool'], x['fans']))).collectAsMap()

# Load business features
business_rdd = sc.textFile(folder_path + '/business.json').map(lambda x: json.loads(x))
business_features = business_rdd.map(lambda x: (x['business_id'], (x['review_count'], x['stars']))).collectAsMap()


def merge_features(data, user_features, business_features, train=True):
    """
    Merge user and business features with train data
    :param data:
    :param user_features: dictionary containing user features
    :param business_features: dictionary containing business features
    :param train: boolean flag to check if data is training data
    :return: (user_id, business_id, rating, user_feature1, user_feature2, ..., business_feature1, business_feature2)
    """
    if train:
        user_id, business_id, rating = data
    else:
        user_id, business_id = data
    user_feature = user_features.get(user_id, (None, None, None, None, None, None))
    business_feature = business_features.get(business_id, (None, None))

    if train:
        return (user_id, business_id, rating, *user_feature, *business_feature)
    else:
        return (user_id, business_id, *user_feature, *business_feature)


# Prepare data for xgboost
train_features_rdd = rdd_train_clean.map(lambda x: merge_features(x, user_features, business_features, train=True))
train_data = np.array(train_features_rdd.collect())
X_train = train_data[:, 3:].astype(np.float32)
y_train = train_data[:, 2].astype(np.float32)

validation_features_rdd = rdd_validation_clean.map(lambda x: (x[0], x[1])).map(lambda x: merge_features(x, user_features, business_features, train=False))
validation_data = np.array(validation_features_rdd.collect())
X_validation = validation_data[:, 2:].astype(np.float32)

# Train XGBoost regressor
# Parameters are chosen based on hyperparameter tuning
model = XGBRegressor(objective='reg:linear', learning_rate=0.1, max_depth=5, n_estimators=200)
model.fit(X_train, y_train)

# Save predictions
validation_preds = model.predict(X_validation)
with open(output_path, 'w') as f:
    f.write('user_id,business_id,prediction\n')
    for i in range(len(validation_data)):
        f.write(f'{validation_data[i][0]},{validation_data[i][1]},{validation_preds[i]}\n')

end_time = time.time()
print("Execution Time:", end_time - start_time)