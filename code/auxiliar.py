from generate_datasets import evaluate_model_on_test_set
import matplotlib.pyplot as plt
import time
import os


def get_time_stamp():
    return str(time.time())


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_scores_from_models(**models):
    models_scores = {}
    for model in models.keys():
        scores = evaluate_model_on_test_set(models[model])
        scores_range = list(scores.keys())
        scores_list = list(scores.values())
        models_scores[model] = scores_list
    return scores_range, models_scores


def plot_from_model_scores(scores_range, model_scores):
    fig = plt.figure(figsize=(6, 5))
    for model in model_scores.keys():
        plt.plot(scores_range, model_scores[model], label=model)
    plt.ylim((0, 1))
    plt.xlim((min(scores_range), max(scores_range)))
    plt.legend()
    plt.show()
    return fig

def clone_model():
    pass

def plot_from_history():
    pass


    
