import sys
import time
from math import sqrt
from pyspark import SparkContext


def get_args():
    """
    Get command line arguments
    :return: train_path, test_path, output_path
    """
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]
    return train_path, test_path, output_path


train_path, test_path, output_path = get_args()
# train_path = 'data/yelp_train.csv'
# test_path = 'data/yelp_val.csv'
# output_path = 'data/item_cf_pred.csv'

# Initialize Spark Context
start_time = time.time()
sc = SparkContext('local[*]', 'item_based_cf')

# Load and preprocess training data
rdd_train = sc.textFile(train_path)
header_train = rdd_train.first()
rdd_train_clean = rdd_train.filter(lambda l: l != header_train).map(lambda l: l.split(',')).cache()

# Load and preprocess testing data
rdd_test = sc.textFile(test_path)
header_test = rdd_test.first()
rdd_test_clean = rdd_test.filter(lambda l: l != header_test).map(lambda l: l.split(','))


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


# Create dictionaries for user and business ratings
business_user_ratings = rdd_train_clean.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()

# Calculate user average ratings
user_avg_ratings = (
    rdd_train_clean
    .map(lambda x: (x[0], (float(x[2]), 1)))
    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    .mapValues(lambda x: x[0] / x[1])
    .collectAsMap()
)

# Mean-center the ratings for each user by subtracting user's average rating
mean_centered_ratings = rdd_train_clean.map(lambda x: (x[1], (x[0], float(x[2]) - user_avg_ratings[x[0]]))) \
    .groupByKey().mapValues(dict).collectAsMap()


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


# Dictionaries for user-item and business average ratings
user_item_rating_mapping = rdd_train_clean.map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()

# Calculate business average ratings
business_avg_ratings_dict = (
    rdd_train_clean
    .map(lambda x: (x[1], (float(x[2]), 1)))
    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    .mapValues(lambda x: x[0] / x[1])
    .collectAsMap()
)
# global average rating
avg_rating = sum(business_avg_ratings_dict.values()) / len(business_avg_ratings_dict)


def baseline_predict_rating(user_id, target_business_id):
    """
    Base-line prediction for items with non-reliable similarities
    Cap the baseline prediction between 1 and 5
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
        return user_id, target_business_id, business_avg_ratings_dict.get(target_business_id, avg_rating)
    if target_business_id not in business_avg_ratings_dict:
        return user_id, target_business_id, user_avg_ratings.get(user_id, avg_rating)

    rated_business = user_item_rating_mapping[user_id]

    candidate_neighbors = [
        (candidate_business, calculate_item_similarity(target_business_id, candidate_business))
        for candidate_business in rated_business
    ]
    selected_neighbors = [(business_id, similarity) for business_id, similarity in candidate_neighbors if similarity > similarity_threshold]

    if len(selected_neighbors) < 4:
        return user_id, target_business_id, baseline_predict_rating(user_id, target_business_id)

    numerator = sum(similarity * rated_business[business_id] for business_id, similarity in selected_neighbors)
    denominator = sum(abs(similarity) for business_id, similarity in selected_neighbors)

    if denominator == 0:
        return user_id, target_business_id, baseline_predict_rating(user_id, target_business_id)
    elif numerator / denominator > 5:
        return user_id, target_business_id, 5
    elif numerator / denominator < 1:
        return user_id, target_business_id, 1
    else:
        return user_id, target_business_id, numerator / denominator


# Predict ratings for validation data
predictions = (
    rdd_test_clean
    .map(lambda x: (x[0], x[1]))
    .map(lambda x: predict_rating(x[0], x[1], similarity_threshold=0))
    .collect()
)

# Save predictions to output file
with open(output_path, 'w') as f:
    f.write('user_id,business_id,prediction\n')
    for user_id, business_id, prediction in predictions:
        f.write(f'{user_id},{business_id},{prediction}\n')

# Report execution time
print("Execution Time: ", time.time() - start_time)
