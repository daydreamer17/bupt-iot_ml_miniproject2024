import tensorflow as tf
from keras import regularizers  # 添加此行以导入正则化器模块
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate
from keras.models import Model

def create_cnn_with_language_embedding(audio_input_shape, num_languages, embedding_dim=8):
    # 音频输入和处理
    audio_input = Input(shape=audio_input_shape, name='audio_input')
    audio_features = Flatten()(audio_input)

    # 语言类别输入和嵌入层
    language_input = Input(shape=(1,), name='language_input')
    language_embedding = Embedding(input_dim=num_languages, output_dim=embedding_dim, name='language_embedding')(language_input)
    language_features = Flatten()(language_embedding)

    # 特征融合
    combined_features = Concatenate()([audio_features, language_features])

    # 全连接层和分类输出
    dense_1 = Dense(64, activation='relu')(combined_features)
    dense_2 = Dense(32, activation='relu')(dense_1)
    output = Dense(1, activation='sigmoid', name='output')(dense_2)

    # 构建模型
    model = Model(inputs=[audio_input, language_input], outputs=output)
    return model





