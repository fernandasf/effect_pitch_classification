import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


class UtilsMetrics():
    def __init__(self):
        pass


    @staticmethod
    def get_histogram(exp_path, config, bins=100, density=True):
        df_tr = pd.read_csv(config['database']['train'])
        df_te = pd.read_csv(config['database']['test'])
        plt.figure(figsize=(10, 4))
        df_tr["f0"].hist(bins=bins, density=density)
        df_tr["f0"].plot.kde()
        df_te["f0"].hist(bins=bins, density=density)
        df_te["f0"].plot.kde()
        plt.title(f"F0 Histogram")
        plt.savefig(f"{exp_path}/Histogram_f0.png")


    @staticmethod
    def get_results(test_audio, test_labels, model):
        y_pred = np.argmax(model.predict(test_audio), axis=1)
        y_true = test_labels        
        test_acc = sum(y_pred == y_true) / len(y_true)
        print(f'Test set accuracy: {test_acc:.0%}')
        return y_true, y_pred, test_acc  


    @staticmethod
    def plot_curve(metrics, path, history):
        plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.savefig(f"{path}/curve.png")


    @staticmethod
    def conf_matrix(y_true, y_pred, path, LABELS):
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx,
                    xticklabels=LABELS,
                    yticklabels=LABELS,
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.savefig(f"{path}/confusion_matrix.png")


    @staticmethod
    def save_results(df_test, test_files, ground_truth, predicted, name):
        genres = []
        F0 = []
        keywords = []
        for path in test_files:
            genre = df_test[df_test['filepaths'].str.contains(path)]["genders"].iloc[0]
            f0 = df_test[df_test['filepaths'].str.contains(path)]["f0"].iloc[0]
            keyword = df_test[df_test['filepaths'].str.contains(path)]["words"].iloc[0]
            genres.append(genre)
            F0.append(f0)
            keywords.append(keyword)
        
        dict_ = {
            "path": test_files,
            "keyword": keywords,
            "genres": genres,
            "F0": F0,
            "ground_truth": ground_truth,
            "predicted": predicted            
        }
        df = pd.DataFrame(dict_)
        df.to_csv(name, index=False)
    