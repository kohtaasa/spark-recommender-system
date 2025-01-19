import sys
import numpy as np
import time
import json
from math import sqrt
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
# output_path = 'data/hybrid_pred.csv'

start_time = time.time()
sc = SparkContext('local[*]', 'hybrid')

# Load train data
rdd_train = sc.textFile(folder_path + '/yelp_train.csv')
header_train = rdd_train.first()
rdd_train_clean = rdd_train.filter(lambda l: l != header_train).map(lambda l: l.split(',')).cache()

# Load validation data
rdd_validation = sc.textFile(test_path)
header_validation = rdd_validation.first()
rdd_validation_clean = rdd_validation.filter(lambda l: l != header_validation).map(lambda l: l.split(',')).cache()


### Item-item CF Functions
def pearson_correlation(ratings):
    """
    Calculate Pearson correlation between two businesses
    :param ratings: list of tuples containing ratings of two businesses (created in business_rating_pairs)
    :return: Pearson correlation coefficient
    """

    sum_i = sum_j = 0
    for i, j in ratings:
        sum_i += i
        sum_j += j

    mean_i = sum_i / len(ratings)
    mean_j = sum_j / len(ratings)

    numerator = sum((i - mean_i) * (j - mean_j) for i, j in ratings)
    denominator = sqrt(sum((i - mean_i) ** 2 for i, _ in ratings) * sum((j - mean_j) ** 2 for _, j in ratings))

    return numerator / denominator if denominator != 0 else 0


def calculate_item_similarity(business1, business2):
    """
    Calculate Pearson correlation between two businesses
    :param business1: business id 1
    :param business2: business id 2
    :return: Pearson correlation coefficient
    """
    users1 = set(mean_centered_ratings.get(business1, {}).keys())
    users2 = set(mean_centered_ratings.get(business2, {}).keys())
    common_users = users1.intersection(users2)

    if len(common_users) < 6:
        return 0

    ratings = [(mean_centered_ratings[business1][user], mean_centered_ratings[business2][user]) for user in common_users]
    similarity = pearson_correlation(ratings)

    return similarity


def baseline_predict_rating(user_id, target_business_id):
    """
    Base-line prediction for items with non-reliable similarities
    :param user_id:
    :param target_business_id:
    :return: baseline prediction
    """
    baseline = avg_rating + (user_avg_ratings.get(user_id, 0) - avg_rating) + (business_avg_ratings_dict.get(target_business_id, 0) - avg_rating)
    if baseline > 5:
        return 5
    elif baseline < 1:
        return 1
    else:
        return baseline


def predict_rating(user_id, target_business_id, similarity_threshold=0.1):
    """
    Predict rating for a user and business using mean-centered ratings
    :param user_id: user id
    :param target_business_id: business id
    :param similarity_threshold: minimum similarity threshold
    :return: tuple of user id, business id, predicted rating
    """

    # Check if user is missing
    if user_id not in user_item_rating_mapping:
        return (user_id, target_business_id), business_avg_ratings_dict.get(target_business_id, avg_rating)
    if target_business_id not in business_avg_ratings_dict:
        return (user_id, target_business_id), user_avg_ratings.get(user_id, avg_rating)

    rated_business = user_item_rating_mapping[user_id]

    candidate_neighbors = [
        (candidate_business, calculate_item_similarity(target_business_id, candidate_business))
        for candidate_business in rated_business
    ]
    selected_neighbors = [(business_id, similarity) for business_id, similarity in candidate_neighbors if similarity > similarity_threshold]

    if len(selected_neighbors) < 4:
        return (user_id, target_business_id), baseline_predict_rating(user_id, target_business_id)

    numerator = sum(similarity * rated_business[business_id] for business_id, similarity in selected_neighbors)
    denominator = sum(abs(similarity) for business_id, similarity in selected_neighbors)

    predicted_rating = numerator / denominator

    if denominator == 0:
        return (user_id, target_business_id), baseline_predict_rating(user_id, target_business_id)
    if numerator / denominator > 5:
        predicted_rating = 5
    elif numerator / denominator < 1:
        predicted_rating = 1

    return (user_id, target_business_id), predicted_rating


### Model-based Functions
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


def train_xgbreg(X_train, y_train):
    """
    Train XGBoost Regressor model
    :param X_train: training features
    :param y_train: training labels
    :return: trained model
    """
    model = XGBRegressor(objective='reg:linear', learning_rate=0.1, max_depth=5, n_estimators=200)
    model.fit(X_train, y_train)

    return model


# Make predictions using Item-item CF
business_user_ratings = (
    rdd_train_clean
    .map(lambda x: (x[1], (x[0], float(x[2]))))
    .groupByKey()
    .mapValues(dict)
    .collectAsMap()
)
user_item_rating_mapping = (
    rdd_train_clean
    .map(lambda x: (x[0], (x[1], float(x[2]))))
    .groupByKey()
    .mapValues(dict)
    .collectAsMap()
)
business_avg_ratings_dict = (
    rdd_train_clean
    .map(lambda x: (x[1], (float(x[2]), 1)))
    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    .mapValues(lambda x: x[0] / x[1])
    .collectAsMap()
)
# Global Average
avg_rating = sum(business_avg_ratings_dict.values()) / len(business_avg_ratings_dict)
user_avg_ratings = (
    rdd_train_clean
    .map(lambda x: (x[0], (float(x[2]), 1)))
    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    .mapValues(lambda x: x[0] / x[1])
    .collectAsMap()
)

mean_centered_ratings = rdd_train_clean.map(lambda x: (x[1], (x[0], float(x[2]) - user_avg_ratings[x[0]]))) \
    .groupByKey().mapValues(dict).collectAsMap()

# print("item-item CF prediction running....")
item_based_predictions = (
    rdd_validation_clean
    .map(lambda x: (x[0], x[1]))
    .map(lambda x: predict_rating(x[0],
                                  x[1],
                                  0.1)
         )
    .filter(lambda x: x[1] is not None)
    .collectAsMap()
)

# Make predictions using Model-based CF
# Load user features
user_rdd = sc.textFile(folder_path + '/user.json').map(lambda x: json.loads(x))
user_features = user_rdd.map(lambda x: (
x['user_id'], (x['review_count'], x['average_stars'], x['useful'], x['funny'], x['cool'], x['fans']))).collectAsMap()

# Load business features
business_rdd = sc.textFile(folder_path + '/business.json').map(lambda x: json.loads(x))
business_features = business_rdd.map(lambda x: (x['business_id'], (x['review_count'], x['stars']))).collectAsMap()

# Prepare data for xgboost
train_features_rdd = rdd_train_clean.map(lambda x: (x[0], x[1], float(x[2]))).map(
    lambda x: merge_features(x, user_features, business_features, train=True))
train_data = np.array(train_features_rdd.collect())
X_train = train_data[:, 3:].astype(np.float32)
y_train = train_data[:, 2].astype(np.float32)

validation_features_rdd = rdd_validation_clean.map(lambda x: (x[0], x[1])).map(
    lambda x: merge_features(x, user_features, business_features, train=False))
validation_data = np.array(validation_features_rdd.collect())
X_validation = validation_data[:, 2:].astype(np.float32)

# print("Model-based prediction running....")
model = train_xgbreg(X_train, y_train)
xgb_predictions = model.predict(X_validation)

# print("Making final predictions....")
# Combine predictions for hybrid model
final_predictions = []
for i in range(len(validation_data)):
    user_id, business_id = validation_data[i][0], validation_data[i][1]

    # Check if CF prediction exists in cf_dict
    cf_prediction = item_based_predictions.get((user_id, business_id))

    combined_prediction = cf_prediction * 0.1 + xgb_predictions[i] * 0.9
    final_predictions.append((user_id, business_id, combined_prediction))

# Save predictions
with open(output_path, 'w') as f:
    f.write('user_id,business_id,prediction\n')
    for user_id, business_id, combined_prediction in final_predictions:
        f.write(f"{user_id},{business_id},{combined_prediction}\n")

print("Execution Time:", time.time() - start_time)
