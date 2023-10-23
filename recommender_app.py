import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup
st.set_page_config(
    page_title="Course Recommender System",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data
def load_course_sims():
    return backend.load_course_sims()


@st.cache_data
def load_courses():
    return backend.load_courses()


@st.cache_data
def load_bow():
    return backend.load_bow()


# Initialize the app by first loading datasets
def init__recommender_app():
    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results


def train(model_name, params):
    if model_name in backend.models:
        # Start training course similarity model
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_name, params)
        st.success('Done!')


def predict(model_name, user_ids, params):
    if model_name in backend.models:
        # Start making predictions based on model name, test user ids, and parameters
        with st.spinner('Generating course recommendations: '):
            time.sleep(0.5)
            res = backend.predict(model_name, user_ids, params)
        st.success('Recommendations generated!')
        return res


# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection select-box
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyperparameters for each model
params = {'top_courses': None, 'sim_threshold': None, 'Number of Clusters': None, 'n_components': None, 'K': None,
          'split': None, 'embedding_size': None, 'batch_size': None, 'epochs': None, 'alpha': None, 'max_depth': None,
          'score_threshold': None, 'learning rate': None}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:

    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=25,
                                    value=10, step=1)

    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=5)

    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold

# User profile model
elif model_selection == backend.models[1]:

    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=25,
                                    value=10, step=1)

    score_threshold = st.sidebar.slider('Score Threshold',
                                        min_value=0, max_value=20,
                                        value=8, step=1)

    params['top_courses'] = top_courses
    params['score_threshold'] = score_threshold

# Clustering model
elif model_selection == backend.models[2]:

    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=25,
                                    value=10, step=1)

    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=35,
                                   value=20, step=1)

    params['top_courses'] = top_courses
    params['Number of Clusters'] = cluster_no

# Clustering with PCA
elif model_selection == backend.models[3]:

    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=25,
                                    value=10, step=1)

    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=35,
                                   value=20, step=1)

    n_components = st.sidebar.slider('Number of Components',
                                     min_value=1, max_value=20,
                                     value=9, step=1)

    params['top_courses'] = top_courses
    params['Number of Clusters'] = cluster_no
    params['n_components'] = n_components

# KNN
elif model_selection == backend.models[4]:

    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=25,
                                    value=10, step=1)

    k = st.sidebar.slider('K',
                          min_value=1, max_value=35,
                          value=11, step=1)

    params['top_courses'] = top_courses
    params['K'] = k

# NMF
elif model_selection == backend.models[5]:

    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=25,
                                    value=10, step=1)

    params['top_courses'] = top_courses

# NN
elif model_selection == backend.models[6]:

    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=25,
                                    value=10, step=1)

    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=5)

    embedding_size = st.sidebar.slider('Embedding Size',
                                       min_value=1, max_value=30,
                                       value=16, step=1)

    batch_size = st.sidebar.slider('Batch Size',
                                   min_value=8, max_value=192,
                                   value=64, step=8)

    epochs = st.sidebar.slider('Epochs',
                               min_value=1, max_value=20,
                               value=10, step=1)

    lr = st.sidebar.slider('Learning rate',
                           min_value=1e-5, max_value=1.0,
                           value=1e-3, step=1e-3)

    params['learning rate'] = lr
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
    params['embedding_size'] = embedding_size
    params['batch_size'] = batch_size
    params['epochs'] = epochs

elif model_selection == backend.models[7]:

    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=25,
                                    value=10, step=1)

    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=5)

    split = st.sidebar.slider('Split',
                              min_value=0.0, max_value=1.0,
                              value=0.2, step=0.05)

    alpha = st.sidebar.slider('Alpha',
                              min_value=0.0, max_value=2.0,
                              value=0.1, step=0.1)

    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
    params['split'] = split
    params['alpha'] = alpha

elif model_selection == backend.models[8]:

    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=25,
                                    value=10, step=1)

    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=5)

    split = st.sidebar.slider('Split',
                              min_value=0.0, max_value=1.0,
                              value=0.2, step=0.05)

    max_depth = st.sidebar.slider('Max depth',
                                  min_value=0, max_value=20,
                                  value=8, step=1)

    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
    params['split'] = split
    params['max_depth'] = max_depth

else:
    pass

# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')
# Start training process
if training_button:
    train(model_selection, params)

# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [new_id]
    res_df = predict(model_selection, user_ids, params)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
    st.table(res_df)
