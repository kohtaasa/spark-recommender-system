# Spark-Based Recommendation Systems

## Overview
This repository contains the implementation of various recommendation systems using Apache Spark. The systems are built to predict star ratings for user-business pairs based on Yelp datasets.
The Yelp dataset can be found here https://www.yelp.com/dataset.

*`business.json` and `user.json` files are not included in this repository due to their large size. Please download them from the Yelp dataset link above.*


## Implemented Cases
### Case 1: Item-Based Collaborative Filtering (CF) with Pearson Similarity
- Implements an item-based CF recommendation system using Pearson similarity.
- Handles cold-start problems with a custom default rating mechanism.
- Execution format:
  ```bash
  spark-submit item_based_cf.py <train_file_name> <test_file_name> <output_file_name>
  ```

### Case 2: Model-Based Recommendation System
- Uses XGBRegressor (version 0.72) to train a model with features like:
  - Average user/business ratings
  - Number of user/business reviews
  - Other features derived from the dataset.
- Execution format:
  ```bash
  spark-submit model_based.py <data_filder_path> <test_file_name> <output_file_name>
  ```
  
### Case 3: Hybrid Recommendation System
- Combines the item-based CF and model-based recommendation systems.
- Execution format:
  ```bash
  spark-submit hybrid.py <data_folder_path> <test_file_name> <output_file_name>
   ```
## Environment Setup
1. Build the Docker image:
   ```bash
   docker build -t <image_name> .
   ```
2. Run the Docker container:
   ```bash
   docker run -it  --name <container_name> -p 8888:8888 -v <absolute-path-to-project-folder>:/workspace <image_name> /bin/bash
   ```
3. Run the script:
   ```bash
   spark-submit <script_name>.py <arguments>
   ```

