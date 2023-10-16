import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from surprise import NMF
from surprise import KNNBasic
from surprise.model_selection import train_test_split as train_test_split_sup
from surprise import Dataset, Reader

from tensorflow import keras
from keras import layers

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding",
          "Classification with Embedding")


def load_ratings():
    return pd.read_csv("Data/ratings.csv")


def load_course_sims():
    return pd.read_csv("Data/sim.csv")


def load_courses():
    df = pd.read_csv("Data/course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("Data/courses_bows.csv")


def load_genre():
    course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
    course_genres_df = pd.read_csv(course_genre_url)
    return course_genres_df


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("Data/ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


def model_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, model, data):
    all_courses = set(idx_id_dict.values())
    # Create a list of candidate courses that the user hasn't interacted with
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                sim = model.predict(data)
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


def create_user_profile(user_id, id_idx_dict, course_genres_df, user_ratings_df):
    user_profile = pd.DataFrame(0, index=[user_id], columns=course_genres_df.columns[1:])

    for index, row in user_ratings_df.iterrows():
        course_id = row['item']
        genre_columns = course_genres_df.columns[1:]

        # Check if the course has a genre and update the user-genre matrix accordingly
        for genre in genre_columns:
            if course_genres_df.loc[course_genres_df['COURSE_ID'] == course_id, genre].values[0] == 1:
                user_profile.at[user_id, genre] += row['rating']

    # get user vector for the current user id
    test_user_vector = user_profile.iloc[0, 1:].values

    # get the unknown course ids for the current user id
    unknown_courses = id_idx_dict
    unknown_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
    unknown_course_ids = unknown_course_df['COURSE_ID'].values

    # Element-wise multiplication to get the recommendation scores for each course
    recommendation_scores = np.dot(unknown_course_df.iloc[:, 2:].values, test_user_vector)

    return recommendation_scores, unknown_course_ids, user_profile


def combine_cluster_labels(user_ids_df, labels):
    labels_df = pd.DataFrame(labels)
    cluster_df = pd.merge(user_ids_df, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    return cluster_df


def users_labelled(cluster_df):
    ratings_df = load_ratings()
    test_users_df = ratings_df[['user', 'item']]
    test_users_labelled = pd.merge(test_users_df, cluster_df, left_on='user', right_on='user')
    courses_cluster = test_users_labelled[['item', 'cluster']]
    courses_cluster['count'] = [1] * len(courses_cluster)
    courses_cluster = courses_cluster.groupby(['cluster', 'item']).agg(enrollments=('count', 'sum')).reset_index()

    return test_users_labelled, courses_cluster


class RecommenderNet(keras.Model):

    def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
        """
           Constructor
           :param int num_users: number of users
           :param int num_items: number of items
           :param int embedding_size: the size of embedding vector
        """
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        # Define a user_embedding vector
        self.user_embedding_layer = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            name='user_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define a user bias layer
        self.user_bias = layers.Embedding(
            input_dim=num_users,
            output_dim=1,
            name="user_bias")

        # Define an item_embedding vector
        self.item_embedding_layer = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            name='item_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define an item bias layer
        self.item_bias = layers.Embedding(
            input_dim=num_items,
            output_dim=1,
            name="item_bias")

    def call(self, inputs):
        """
           method to be called during model fitting

           :param inputs: user and item one-hot vectors
        """
        # Compute the user embedding vector
        user_vector = self.user_embedding_layer(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding_layer(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        # Add all the components (including bias)
        x = dot_user_item + user_bias + item_bias
        # Sigmoid output layer to output the probability
        return tf.nn.relu(x)


def process_dataset(raw_data):
    encoded_data = raw_data.copy()

    # Mapping user ids to indices
    user_list = encoded_data["user"].unique().tolist()
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}

    # Mapping course ids to indices
    course_list = encoded_data["item"].unique().tolist()
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)}

    # Convert original user ids to idx
    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    # Convert original course ids to idx
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    # Convert rating to int
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")

    return encoded_data, user_idx2id_dict, course_idx2id_dict


def generate_train_test_datasets(dataset, scale=True):

    min_rating = min(dataset["rating"])
    max_rating = max(dataset["rating"])

    dataset = dataset.sample(frac=1)
    x = dataset[["user", "item"]].values
    if scale:
        y = dataset["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    else:
        y = dataset["rating"].values

    # Assuming training on 80% of the data and validating on 10%, and testing 10%
    train_indices = int(0.8 * dataset.shape[0])
    test_indices = int(0.9 * dataset.shape[0])

    x_train, x_val, x_test, y_train, y_val, y_test = (
        x[:train_indices],
        x[train_indices:test_indices],
        x[test_indices:],
        y[:train_indices],
        y[train_indices:test_indices],
        y[test_indices:],
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


# Model training
def train(model_name, params):

    if model_name == models[0] or model_name == models[1]:
        pass

    # Clustering training
    elif model_name == models[2]:

        cluster_no = params["Number of Clusters"]

        user_profile_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/user_profile.csv"
        user_profile_df = pd.read_csv(user_profile_url)

        feature_names = list(user_profile_df.columns[1:])

        scaler = StandardScaler()
        user_profile_df[feature_names] = scaler.fit_transform(user_profile_df[feature_names])

        features = user_profile_df.loc[:, user_profile_df.columns != 'user']
        user_ids = user_profile_df.loc[:, user_profile_df.columns == 'user']

        # Create a K-Means model with the optimized K value
        kmeans = KMeans(n_clusters=cluster_no)

        # Fit the K-Means model to your features
        kmeans.fit(features)

        model_filename = 'models/K_means_model.joblib'
        joblib.dump(kmeans, model_filename)

        cluster_labels = kmeans.labels_

        cluster_df = combine_cluster_labels(user_ids, cluster_labels)
        cluster_df.to_csv('Data/cluster_data.csv', index=False)

    # Clustering with PCA training
    elif model_name == models[3]:

        cluster_no = params["Number of Clusters"]
        n_components = params["n_components"]

        user_profile_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/user_profile.csv"
        user_profile_df = pd.read_csv(user_profile_url)

        features = user_profile_df.loc[:, user_profile_df.columns != 'user']

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(features)

        model = KMeans(n_clusters=cluster_no)

        # Fit KMeans to the PCA-transformed features
        model.fit(components)

        model_filename = 'models/PCA_model.joblib'
        joblib.dump(model, model_filename)

    # KNN Training
    elif model_name == models[4]:

        k = params["K"]

        # Load the ratings data into a Pandas DataFrame
        rating_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"
        rating_df = pd.read_csv(rating_url)

        rating_df.to_csv("Data/course_ratings.csv", index=False)

        reader = Reader(
            line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))

        course_dataset = Dataset.load_from_file("Data/course_ratings.csv", reader=reader)

        train_set, test_set = train_test_split_sup(course_dataset, test_size=.00001)

        model = KNNBasic(k=k, min_k=1, sim_options={'name': 'cosine', 'user_based': False}, verbose=True)
        model.fit(train_set)

        model_filename = 'models/KNN_model.joblib'
        joblib.dump(model, model_filename)

    # NMF training
    elif model_name == models[5]:

        rating_df = load_ratings()
        rating_df.to_csv("Data/course_ratings.csv", index=False)
        # Read the course rating dataset with columns user item rating
        reader = Reader(
            line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))

        course_dataset = Dataset.load_from_file("Data/course_ratings.csv", reader=reader)

        train_set, test_set = train_test_split_sup(course_dataset, test_size=.00001)

        model = NMF()
        model.fit(train_set)

        model_filename = 'models/NMF_model.joblib'
        joblib.dump(model, model_filename)

    # Neural Network Training
    elif model_name == models[6]:

        embedding_size = params["embedding_size"]
        batch_size = params["batch_size"]
        epochs = params["epochs"]
        lr = params["learning rate"]

        rating_df = load_ratings()
        encoded_data, user_idx2id_dict, course_idx2id_dict = process_dataset(rating_df)

        X_train, X_val, X_test, y_train, y_val, y_test = generate_train_test_datasets(encoded_data)

        num_users = len(rating_df['user'].unique())
        num_items = len(rating_df['item'].unique())

        model = RecommenderNet(num_users, num_items, embedding_size)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=tf.keras.metrics.RootMeanSquaredError())

        model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

        model_filename = 'models/NN_model.joblib'
        joblib.dump(model, model_filename)

    # Regression with embedding features training
    elif model_name == models[7]:

        split = params["split"]
        alpha = params["alpha"]

        rating_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"
        user_emb_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/user_embeddings.csv"
        item_emb_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_embeddings.csv"
        rating_df = pd.read_csv(rating_url)
        user_emb = pd.read_csv(user_emb_url)
        item_emb = pd.read_csv(item_emb_url)

        # Merge user embedding features
        user_emb_merged = pd.merge(rating_df, user_emb, how='left', left_on='user', right_on='user').fillna(0)
        # Merge course embedding features
        merged_df = pd.merge(user_emb_merged, item_emb, how='left', left_on='item', right_on='item').fillna(0)

        u_features = [f"UFeature{i}" for i in range(16)]
        c_features = [f"CFeature{i}" for i in range(16)]

        user_embeddings = merged_df[u_features]
        course_embeddings = merged_df[c_features]
        ratings = merged_df['rating']

        # Aggregate the two feature columns using element-wise add
        regression_dataset = user_embeddings + course_embeddings.values
        regression_dataset.columns = [f"Feature{i}" for i in range(16)]
        regression_dataset['rating'] = ratings

        X = regression_dataset.iloc[:, :-1]
        y = regression_dataset.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

        model = linear_model.Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        model_filename = 'models/Regression_with_embedding_model.joblib'
        joblib.dump(model, model_filename)

    # Classification with embedding features training
    elif model_name == models[8]:

        max_depth = params["max_depth"]
        split = params["split"]

        user_emb_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/user_embeddings.csv"
        item_emb_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_embeddings.csv"

        rating_df = load_ratings()
        user_emb = pd.read_csv(user_emb_url)
        item_emb = pd.read_csv(item_emb_url)

        # Merge user embedding features
        merged_df = pd.merge(rating_df, user_emb, how='left', left_on='user', right_on='user').fillna(0)
        # Merge course embedding features
        merged_df = pd.merge(merged_df, item_emb, how='left', left_on='item', right_on='item').fillna(0)

        u_features = [f"UFeature{i}" for i in range(16)]
        c_features = [f"CFeature{i}" for i in range(16)]

        user_embeddings = merged_df[u_features]
        course_embeddings = merged_df[c_features]
        ratings = merged_df['rating']

        # Aggregate the two feature columns using element-wise add
        interaction_dataset = user_embeddings + course_embeddings.values
        interaction_dataset.columns = [f"Feature{i}" for i in range(16)]
        interaction_dataset['rating'] = ratings

        X = interaction_dataset.iloc[:, :-1]
        y_raw = interaction_dataset.iloc[:, -1]

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.values.ravel())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

        model = RandomForestClassifier(max_depth=max_depth)
        model.fit(X_train, y_train)

        model_filename = 'models/Classification_with_embedding_model.joblib'
        joblib.dump(model, model_filename)

    else:
        pass


# Prediction
def predict(model_name, user_ids, params):

    top_courses = params["top_courses"]

    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:

        # Course Similarity predictions
        if model_name == models[0]:

            sim_threshold = params["sim_threshold"] / 100.0
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()

            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)

            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)

        # User profile predictions
        elif model_name == models[1]:

            score_threshold = params["score_threshold"]
            ratings_df = load_ratings()
            course_genres_df = load_genre()
            user_ratings_df = ratings_df[ratings_df['user'] == user_id]

            recommendation_scores, unknown_course_ids, user_profile = create_user_profile(user_id, id_idx_dict,
                                                                                          course_genres_df,
                                                                                          user_ratings_df)

            # Append the results into the users, courses, and scores list
            for i in range(0, len(unknown_course_ids)):
                score = recommendation_scores[i]
                # Only keep the courses with high recommendation score
                if score >= score_threshold:
                    users.append(user_id)
                    courses.append(unknown_course_ids[i])
                    scores.append(recommendation_scores[i])

        # Clustering predictions
        elif model_name == models[2]:

            model = joblib.load('models/K_means_model.joblib')

            course_genres_df = load_genre()
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]

            user_profile_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/user_profile.csv"
            user_profile_df = pd.read_csv(user_profile_url)

            recommendation_scores, unknown_course_ids, user_profile = create_user_profile(user_id,
                                                                                          id_idx_dict,
                                                                                          course_genres_df,
                                                                                          user_ratings)

            user_profile = user_profile.rename(columns={'TITLE': 'user'})
            user_profile['user'] = user_id  # Add the user ID to the 'user' column
            last_index = user_profile_df.index.max()
            user_profile = user_profile.reset_index(drop=True)
            user_profile.index += last_index + 1

            user_profile_df = pd.concat([user_profile_df, user_profile], axis=0)

            user_ids_df = user_profile_df.loc[:, user_profile_df.columns == 'user']

            feature_names = list(user_profile_df.columns[1:])

            scaler = StandardScaler()
            user_profile_df[feature_names] = scaler.fit_transform(user_profile_df[feature_names])

            features = user_profile_df.loc[:, user_profile_df.columns != 'user']

            model.fit(features)

            cluster_labels = model.labels_

            cluster_df = combine_cluster_labels(user_ids_df, cluster_labels)

            test_users_df = ratings_df.drop('rating', axis=1)

            test_users_labelled = pd.merge(test_users_df, cluster_df, left_on='user', right_on='user')

            courses_cluster = test_users_labelled[['item', 'cluster']]
            courses_cluster['count'] = [1] * len(courses_cluster)
            courses_cluster = courses_cluster.groupby(['cluster', 'item']).agg(
                enrollments=('count', 'sum')).reset_index()

            # Create a user subset for the current test user
            user_subset = test_users_labelled[test_users_labelled['user'] == user_id]

            # Get the enrolled courses of the current user
            enrolled_courses = set(user_subset['item'])

            # Find the cluster label of the current user (assuming it's the same for all rows)
            cluster_id = user_subset['cluster'].iloc[0]

            # Find all courses in the same cluster
            cluster_courses = set(test_users_labelled[test_users_labelled['cluster'] == cluster_id]['item'])

            # Calculate the set of new/unseen courses
            unseen_courses = cluster_courses.difference(enrolled_courses)

            # Filter popular courses (e.g., enrollments beyond a threshold like 10)
            popular_courses = [course for course in unseen_courses if courses_cluster[
                (courses_cluster['cluster'] == cluster_id) & (courses_cluster['item'] == course)]['enrollments'].values[
                0] > 10]
            popular_courses.sort(reverse=True)
            user_recommendations = popular_courses[:top_courses]

            score = [courses_cluster[
                          (courses_cluster['cluster'] == cluster_id) & (courses_cluster['item'] == course)][
                          'enrollments'].values[0] for course in popular_courses]

            for i in range(0, len(user_recommendations)):
                users.append(user_id)
                courses.append(user_recommendations[i])
                scores.append(score[i])

        # Clustering with PCA predictions
        elif model_name == models[3]:

            n_components = params["n_components"]

            model = joblib.load('models/PCA_model.joblib')

            ratings_df = load_ratings()
            course_genres_df = load_genre()
            user_ratings = ratings_df[ratings_df['user'] == user_id]

            user_profile_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/user_profile.csv"
            user_profile_df = pd.read_csv(user_profile_url)

            recommendation_scores, unknown_course_ids, user_profile = create_user_profile(user_id,
                                                                                          id_idx_dict,
                                                                                          course_genres_df,
                                                                                          user_ratings)

            user_profile = user_profile.rename(columns={'TITLE': 'user'})
            user_profile['user'] = user_id  # Add the user ID to the 'user' column
            last_index = user_profile_df.index.max()
            user_profile = user_profile.reset_index(drop=True)
            user_profile.index += last_index + 1

            user_profile_df = pd.concat([user_profile_df, user_profile], axis=0)

            user_ids_df = user_profile_df.loc[:, user_profile_df.columns == 'user']

            feature_names = list(user_profile_df.columns[1:])

            scaler = StandardScaler()
            user_profile_df[feature_names] = scaler.fit_transform(user_profile_df[feature_names])

            features = user_profile_df.loc[:, user_profile_df.columns != 'user']

            pca = PCA(n_components=n_components)
            components = pca.fit_transform(features)

            model.fit(components)

            cluster_labels = model.labels_

            cluster_df = combine_cluster_labels(user_ids_df, cluster_labels)

            test_users_df = ratings_df.drop('rating', axis=1)

            test_users_labelled = pd.merge(test_users_df, cluster_df, left_on='user', right_on='user')

            courses_cluster = test_users_labelled[['item', 'cluster']]
            courses_cluster['count'] = [1] * len(courses_cluster)
            courses_cluster = courses_cluster.groupby(['cluster', 'item']).agg(
                enrollments=('count', 'sum')).reset_index()

            # Create a user subset for the current test user
            user_subset = test_users_labelled[test_users_labelled['user'] == user_id]

            # Get the enrolled courses of the current user
            enrolled_courses = set(user_subset['item'])

            # Find the cluster label of the current user (assuming it's the same for all rows)
            cluster_id = user_subset['cluster'].iloc[0]

            # Find all courses in the same cluster
            cluster_courses = set(test_users_labelled[test_users_labelled['cluster'] == cluster_id]['item'])

            # Calculate the set of new/unseen courses
            unseen_courses = cluster_courses.difference(enrolled_courses)

            # Filter popular courses (e.g., enrollments beyond a threshold like 10)
            popular_courses = [course for course in unseen_courses if courses_cluster[
                (courses_cluster['cluster'] == cluster_id) & (courses_cluster['item'] == course)]['enrollments'].values[
                0] > 10]
            popular_courses.sort(reverse=True)
            user_recommendations = popular_courses[:top_courses]

            score = [courses_cluster[
                          (courses_cluster['cluster'] == cluster_id) & (courses_cluster['item'] == course)][
                          'enrollments'].values[0] for course in popular_courses]

            for i in range(0, len(user_recommendations)):
                users.append(user_id)
                courses.append(user_recommendations[i])
                scores.append(score[i])

        # KNN predictions
        elif model_name == models[4]:

            model = joblib.load('models/KNN_model.joblib')

            ratings_df = load_ratings()
            all_course_ids = ratings_df['item'].unique()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()

            # Generate predictions for courses that the user hasn't interacted with
            other_courses = [
                course_id for course_id in all_course_ids if course_id not in enrolled_course_ids
            ]

            # Create a test-set for the user with unrated courses
            testset = [(user_id, course_id, 0) for course_id in other_courses]

            # Use the KNN model to predict ratings for unrated courses
            predictions = model.test(testset)

            # Sort the predictions by predicted rating in descending order
            predictions.sort(key=lambda x: x.est, reverse=True)

            # Extract the top recommended courses with predicted ratings
            for rec in predictions[:top_courses]:
                course_id, score = rec.iid, rec.est
                users.append(user_id)
                courses.append(course_id)
                scores.append(score)

        # NMF predictions
        elif model_name == models[5]:

            model = joblib.load('models/NMF_model.joblib')

            ratings_df = load_ratings()
            all_course_ids = ratings_df['item'].unique()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()

            # Generate predictions for courses that the user hasn't interacted with
            other_courses = [
                course_id for course_id in all_course_ids if course_id not in enrolled_course_ids
            ]

            # Create a test-set for the user with unrated courses
            testset = [(user_id, course_id, 0) for course_id in other_courses]

            # Use the KNN model to predict ratings for unrated courses
            predictions = model.test(testset)

            # Sort the predictions by predicted rating in descending order
            predictions.sort(key=lambda x: x.est, reverse=True)

            # Extract the top recommended courses with predicted ratings
            for rec in predictions[:top_courses]:
                course_id, score = rec.iid, rec.est
                users.append(user_id)
                courses.append(course_id)
                scores.append(score)

        # NN predictions
        elif model_name == models[6]:

            sim_threshold = params["sim_threshold"] / 100.0

            model = joblib.load('models/NN_model.joblib')

            ratings_df = load_ratings()
            all_course_ids = ratings_df['item'].unique()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            print(user_ratings)
            enrolled_course_ids = user_ratings['item'].to_list()
            encoded_data, user_idx2id_dict, course_idx2id_dict = process_dataset(ratings_df)

            user_id_to_index = {v: k for k, v in user_idx2id_dict.items()}
            user_index = user_id_to_index.get(user_id, None)

            other_courses = [
                course_id for course_id in all_course_ids if course_id not in enrolled_course_ids
            ]

            # Create the testset with mapped user and item indices
            testset = list(zip([user_index] * len(other_courses),
                               course_idx2id_dict, [0] * len(other_courses)))

            # Reshape the testset
            testset = np.array(testset)
            user_input = testset[:, 0]
            item_input = testset[:, 1]

            testset = np.vstack((user_input, item_input)).T
            print(testset)

            preds = model.predict(testset)
            print(preds)
            '''
            user_list = user_ratings["user"].unique().tolist()
            user_idx2id_dict = {i: x for i, x in enumerate(user_list)}

            course_list = user_ratings["item"].unique().tolist()
            course_idx2id_dict = {i: x for i, x in enumerate(course_list)}

            # Encode the user_id and get the corresponding user index
            user_idx = user_idx2id_dict.get(user_id, None)
            print(user_idx)

            if user_idx is not None:
                # Create input data for predictions
                user_indices = np.full(top_courses, user_idx, dtype=int)
                item_indices = np.arange(top_courses)  # You can adapt this based on your requirements
                input_data = np.column_stack((user_indices, item_indices))

                # Make predictions
                preds = model.predict(input_data)

                # Loop through the predictions and filter by the sim_threshold
                for item_idx, score in enumerate(preds):
                    if score >= sim_threshold:
                        users.append(user_id)
                        # Decode the item index back to item_id
                        item_id = course_idx2id_dict.get(item_idx, None)
                        if item_id is not None:
                            courses.append(item_id)
                            scores.append(score)
            else:
                print(f"User with ID {user_id} not found in training data")
            '''
        else:
            pass

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    print(res_df)

    # Get the top N courses based on the SCORE column
    top_courses_df = res_df.nlargest(top_courses, 'SCORE')

    return top_courses_df
