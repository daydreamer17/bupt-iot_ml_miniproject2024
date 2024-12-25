
import pandas as pd
import librosa
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences

# 加载标签并对语言列进行编码
def load_labels_with_language_encoding(csv_path):
    labels_df = pd.read_csv(csv_path)
    labels_df['Story_type'] = labels_df['Story_type'].str.lower().str.replace(" ", "_")
    labels_df['label'] = labels_df['Story_type'].apply(lambda x: 1 if x == 'true_story' else 0)

    # 对语言类别进行编码
    label_encoder = LabelEncoder()
    labels_df['language_encoded'] = label_encoder.fit_transform(labels_df['Language'])
    return labels_df, label_encoder

# 提取音频特征
def extract_mel_spectrogram(audio_path, n_mels=128, hop_length=512, n_fft=2048, target_length=256):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 填充或裁剪到目标长度
        if mel_spec_db.shape[1] < target_length:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_length - mel_spec_db.shape[1])), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :target_length]

        # 增加通道维度
        mel_spec_db = mel_spec_db[..., np.newaxis]
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# 数据准备函数
def prepare_data_without_bert(audio_folder, labels_df, feature_extractor, audio_target_length=256):
    audio_features = []
    language_labels = []
    labels = []

    for index, row in labels_df.iterrows():
        # 获取音频文件路径和标签
        audio_path = os.path.join(audio_folder, row['filename'])
        label = row['label']
        language_label = row['language_encoded']  # 使用编码后的语言类别

        # 提取音频特征
        feature = feature_extractor(audio_path, target_length=audio_target_length)
        if feature is None:
            continue  # 跳过处理失败的音频文件

        # 将特征和标签加入列表
        audio_features.append(feature)
        language_labels.append(language_label)
        labels.append(label)

    # 转换为 NumPy 数组
    audio_features = np.array(audio_features)
    language_labels = np.array(language_labels, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)

    return audio_features, language_labels, labels





