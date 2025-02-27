import pandas as pd
import json
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from sklearn import metrics

"""
Ensure file has "Mode" and "Actual value" columns from the Retool.
"""

MAX_MODE = 11
FILE_NAME = "ensemble_reinfections_ft.csv"

def main():
    df = pd.read_csv(FILE_NAME)
    results = []
    results_tuples = []
    for threshold in range(1, MAX_MODE + 1):
        df = df[["Mode", "Actual value"]]
        df[["Mode"]] = df[["Mode"]]
        # convert to tuples
        tuples = [tuple(x) for x in df.to_numpy()]
        raw_predicted_and_actual = [(Counter(json.loads(tuple[0]))["included"], tuple[1]) for tuple in tuples]
        results_tuples.append(raw_predicted_and_actual)
        tuples = [("included" if tuple[0] >= threshold else "excluded", tuple[1]) for tuple in raw_predicted_and_actual]
        
        accuracy = sum([1 if tuple[0] == tuple[1] else 0 for tuple in tuples]) / len(tuples)
        sensitivity = sum([1 if tuple[0] == "included" and tuple[1] == "included" else 0 for tuple in tuples]) / sum([1 if tuple[1] == "included" else 0 for tuple in tuples])
        specificity = sum([1 if tuple[0] == "excluded" and tuple[1] == "excluded" else 0 for tuple in tuples]) / sum([1 if tuple[1] == "excluded" else 0 for tuple in tuples])
        results.append((threshold, accuracy, sensitivity, specificity))
        
        
    results_tuples = [(tuple[0] / 11, tuple[1]) for tuple_array in results_tuples for tuple in tuple_array]
    y = [1.0 if tuple[1] == "included" else 0.0 for tuple in results_tuples]
    pred = [tuple[0] for tuple in results_tuples]
    print(list(zip(y, pred))[:20])
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    print(metrics.auc(fpr, tpr))
    print(thresholds)
    
    # 1. AUC (uncomment)
    # plt.plot(fpr, tpr)
    # plt.show()
    
    # 2. Accuracy, Sensitivity, Specificity
    results = np.array(results)
    plt.plot(results[:, 0], results[:, 1], label="Accuracy")
    plt.plot(results[:, 0], results[:, 2], label="Sensitivity")
    plt.plot(results[:, 0], results[:, 3], label="Specificity")
    plt.legend()
    plt.show()
    
    
    
    
    



main()