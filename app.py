import os
from http.client import HTTPException
import fastapi
import pandas as pd
from sqlalchemy import create_engine, text
from catboost import CatBoostClassifier
from typing import List
from schema import PostGet, Response
import hashlib
from loguru import logger


#Загрузка модели
def get_model_path(model_version: str) -> str:
    """
    Модификация функции для возможности загрузить обе модели
    """
#    print(os.environ)
    if (
        os.environ.get("IS_LMS") == "1"
    ):  
        model_path = f"/workdir/user_input/model_{model_version}"
    else:
        model_path = f"T:\\Data\\mo22 final\\model_{model_version}"
    return model_path



def load_models(model_version: str):
    model_path = get_model_path(model_version)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(fname=model_path)
    return loaded_model



#Выгрузка признаков из БД

DATABASE_URI = "postgresql://robot-startml-ro:*******@postgres.lab.karpov.courses:6432/startml"

engine = create_engine(DATABASE_URI)

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(DATABASE_URI)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_u_features() -> pd.DataFrame:
    query = '''SELECT * FROM o_levitskaja_u_features_lesson_22'''
    features_df = batch_load_sql(query)
    return features_df

def load_p_features() -> pd.DataFrame:
    query = '''SELECT * FROM o_levitskaja_p_features_lesson_22'''
    features_df = batch_load_sql(query)
    return features_df

def load_posts() -> pd.DataFrame:
    query = '''SELECT * FROM public.post_text_df'''
    posts = batch_load_sql(query)
    return posts


#Сервис (переделать)
# Теперь мы загружаем сразу 2 модели
model_control = load_models("control")
model_test = load_models("test")

u_features = load_u_features()
p_features = load_p_features()
df_posts = load_posts()
app = fastapi.FastAPI()
### USER SPLITTING

SALT = "my_salt"


def get_user_group(id: int) -> str:
    value_str = str(id) + SALT
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num % 100
    if percent < 50:
        return "control"
    elif percent < 100:
        return "test"
    return "unknown"


def get_recommendations(user_id):
    # Выбираем группу пользователи
    user_group = get_user_group(id=user_id)
    logger.info(f"user group {user_group}")

    # Выбираем нужную модель
    if user_group == "control":
        model = model_control
    elif user_group == "test":
        model = model_test
    else:
        raise ValueError("unknown group")

    # Добавляем столбец с заданным user_id к признакам постов
    p_features['user_id'] = user_id
    # Отбираем признаки заданного пользователя
    user_features = u_features.loc[u_features['user_id'] == user_id]
    # Создаем датафрейм с признаками пользователя для каждой записи признаков постов
    user_values = user_features.iloc[0].to_dict()
    # Добавляем столбцы из user_features в p_features
    df_pp = p_features.assign(**user_values)
    # Отбираем колонки для обучения модели
    model_column = ['gender', 'age', 'country', 'city', 'exp_group', 'topic', 'tfidf_sum', 'tfidf_max', 'first_pop_top',
                    'second_pop_top', 'user_total_likes']

    df_pp['pred_prob'] = model.predict_proba(df_pp[model_column])[:, 1]
    df_pp = df_pp.sort_values(by='pred_prob', ascending=False)
    df_pp = df_pp[['post_id']]
    df_pp.drop_duplicates(inplace=True)

    recommended_posts_idx = df_pp['post_id'].head(5).tolist()
    recommended_posts = df_posts.loc[df_posts['post_id'].isin(recommended_posts_idx), ['post_id', 'text', 'topic']]
    recommended_posts = recommended_posts.rename(columns={'post_id': 'id'})
    recommended_posts_list = recommended_posts.to_dict(orient='records')

    return Response(exp_group=user_group, recommendations=recommended_posts_list)



@app.get("/post/recommendations/", response_model=Response)
def get_recommend(id: int) -> Response:
    recommendations = get_recommendations(id)
    return recommendations

