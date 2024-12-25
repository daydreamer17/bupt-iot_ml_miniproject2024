# train.py

import numpy as np
from keras.optimizers.legacy import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.optimizers import Adam
from keras.callbacks import Callback
from datapreprocess import load_labels_with_language_encoding, prepare_data_without_bert, extract_mel_spectrogram
from model import create_cnn_with_language_embedding
from tqdm import tqdm  # 导入tqdm进度条

# 自定义TQDM进度条回调类
class TQDMProgressBar(Callback):
    def __init__(self, val_data, val_labels):
        super(TQDMProgressBar, self).__init__()
        self.val_data = val_data
        self.val_labels = val_labels

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.epochs_progress = tqdm(total=self.epochs, desc="Training Progress", position=0, ncols=100)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_progress.update(1)

        # 计算并打印F1分数, 精确率, 召回率
        y_pred = self.model.predict(self.val_data)
        y_pred_binary = (y_pred > 0.5).astype(int)  # 二分类阈值设为0.5
        f1 = f1_score(self.val_labels, y_pred_binary, zero_division=0)
        precision = precision_score(self.val_labels, y_pred_binary, zero_division=0)
        recall = recall_score(self.val_labels, y_pred_binary, zero_division=0)

        print(f"Epoch {epoch + 1} - F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    def on_batch_end(self, batch, logs=None):
        if 'loss' in logs:
            tqdm.write(f"Batch {batch} - Loss: {logs['loss']:.4f}")

    def on_train_end(self, logs=None):
        self.epochs_progress.close()

def main():
    # 路径设置
    audio_folder = './Deception-main/CBU0521DD_stories'
    csv_path = './Deception-main/CBU0521DD_stories_attributes.csv'

    # 加载标签和语言编码器
    labels_df, label_encoder = load_labels_with_language_encoding(csv_path)

    # 数据准备
    X_audio, X_language, y = prepare_data_without_bert(
        audio_folder, labels_df, extract_mel_spectrogram, audio_target_length=256
    )

    # 标准化音频特征
    mean = np.mean(X_audio, axis=0)
    std = np.std(X_audio, axis=0)
    X_audio = (X_audio - mean) / (std + 1e-9)

    # 数据划分
    X_train_audio, X_temp_audio, X_train_language, X_temp_language, y_train, y_temp = train_test_split(
        X_audio, X_language, y, test_size=0.4, random_state=42
    )
    X_val_audio, X_test_audio, X_val_language, X_test_language, y_val, y_test = train_test_split(
        X_temp_audio, X_temp_language, y_temp, test_size=0.25, random_state=42
    )

    # 类别权重计算
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    # 创建模型
    num_languages = len(label_encoder.classes_)
    cnn_model_1 = create_cnn_with_language_embedding(
        audio_input_shape=(128, 256, 1),
        num_languages=num_languages,
        embedding_dim=8  # 嵌入维度
    )
    cnn_model_2 = create_cnn_with_language_embedding(
        audio_input_shape=(128, 256, 1),
        num_languages=num_languages,
        embedding_dim=8  # 嵌入维度
    )
    cnn_model_3 = create_cnn_with_language_embedding(
        audio_input_shape=(128, 256, 1),
        num_languages=num_languages,
        embedding_dim=8  # 嵌入维度
    )

    # 编译每个模型
    optimizer_1= Adam(learning_rate=1e-4)
    optimizer_2= Adam(learning_rate=1e-4)
    optimizer_3= Adam(learning_rate=1e-4)
    cnn_model_1.compile(optimizer=optimizer_1, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    cnn_model_2.compile(optimizer=optimizer_2, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    cnn_model_3.compile(optimizer=optimizer_3, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

    # 训练回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

    # 模型训练
    history_1= cnn_model_1.fit(
        [X_train_audio, X_train_language], y_train,
        validation_data=([X_val_audio, X_val_language], y_val),
        epochs=5,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=[early_stopping, lr_scheduler]
    )
    # 模型训练
    history_2= cnn_model_2.fit(
        [X_train_audio, X_train_language], y_train,
        validation_data=([X_val_audio, X_val_language], y_val),
        epochs=4,
        batch_size=16,
        class_weight=class_weights_dict,
        callbacks=[early_stopping, lr_scheduler]
    )
    # 模型训练
    history_3= cnn_model_3.fit(
        [X_train_audio, X_train_language], y_train,
        validation_data=([X_val_audio, X_val_language], y_val),
        epochs=3,
        batch_size=8,
        class_weight=class_weights_dict,
        callbacks=[early_stopping, lr_scheduler]
    )
    # 测试集预测（多个模型）
    y_pred_1 = cnn_model_1.predict([X_test_audio, X_test_language])
    y_pred_2 = cnn_model_2.predict([X_test_audio, X_test_language])
    y_pred_3 = cnn_model_3.predict([X_test_audio, X_test_language])
    
    print("Test Set Predictions:", y_pred_1)
    print("Test Set Predictions:", y_pred_2)
    print("Test Set Predictions:", y_pred_3)
    
    # 简单平均集成
    ensemble_predictions = (y_pred_1 + y_pred_2 + y_pred_3) / 3
    
    # 转换为二分类标签
    y_pred_binary = (ensemble_predictions > 0.5).astype(int)
    
    accuracy = np.mean(y_pred_binary.flatten() == y_test.flatten()) * 100
    f1 = f1_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)


    print(f"Ensemble Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


if __name__ == "__main__":
    main()





