import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# 保存模型
def save_model(model, model_name='model.h5'):
    """保存训练好的模型到指定文件"""
    if not os.path.exists('models'):
        os.makedirs('models')
    model_path = os.path.join('models', model_name)
    model.save(model_path)
    print(f"Model saved at {model_path}")
    
# 加载模型
def load_trained_model(model_path='models/model.h5'):
    """加载已经训练好的模型"""
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"No model found at {model_path}")
        return None

# 评估模型
def evaluate_model(model, x_test, y_test):
    """评估给定的模型在测试集上的表现"""
    if model is not None:
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        print(f"Test accuracy: {test_acc}")
        return test_loss, test_acc
    else:
        print("Model is not loaded.")
        return None, None

# 模型摘要
def model_summary(model):
    """打印模型的结构摘要"""
    if model is not None:
        model.summary()
    else:
        print("Model is not loaded.")
