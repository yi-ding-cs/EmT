import pandas as pd
from pathlib import Path
import numpy as np
import ast
import os
import sys
import argparse


def csv_generator(path_to_read_result, num_folds=10, **kwargs):

    fold_wise_csv_filename = "training_logs.csv"
    csv_filename = os.path.join(path_to_read_result, "result.csv")

    num_folds = num_folds
    metrics = ["rmse", "pcc", "ccc"]
    validate_result = np.zeros((len(metrics), num_folds + 2))
    test_result = np.zeros((len(metrics), num_folds + 2))

    for path in Path(path_to_read_result).rglob(fold_wise_csv_filename):
        fold = int(path.parts[-2].split('_')[-2])

        print(path)
        df = pd.read_csv(path, skiprows=4)
        last_row = df.iloc[-1, :]

        best_epoch = str(df["best_epoch"].iloc[-2])
        result_of_best_epoch = df.loc[df['epoch'] == best_epoch]

        test_rmse = float(last_row.iloc[2])
        test_pcc = float(last_row.iloc[4])
        test_ccc = float(last_row.iloc[7])

        val_rmse = float(result_of_best_epoch["val_rmse"].values[0])
        val_pcc = result_of_best_epoch["val_pcc_v"].values[0]
        val_ccc = float(result_of_best_epoch["val_ccc"].values[0])

        # test_rmse = float(last_row.iloc[2])
        # test_pcc = float(last_row.iloc[4])
        # test_ccc = float(last_row.iloc[7])
        #
        # val_rmse = float(result_of_best_epoch["val_rmse_v"].values[0])
        # val_pcc = result_of_best_epoch["val_pcc_v_v"].values[0]
        # val_ccc = float(result_of_best_epoch["val_ccc_v"].values[0])

        test_result[0, fold] = test_rmse
        test_result[1, fold] = test_pcc
        test_result[2, fold] = test_ccc

        validate_result[0, fold] = val_rmse
        validate_result[1, fold] = val_pcc
        validate_result[2, fold] = val_ccc

    test_result[:, -2] = np.mean(test_result[:, :-2], axis=1)
    test_result[:, -1] = np.std(test_result[:, :-2], axis=1)

    validate_result[:, -2] = np.mean(validate_result[:, :-2], axis=1)
    validate_result[:, -1] = np.std(validate_result[:, :-2], axis=1)

    # result_csv = pd.DataFrame(validate_result, columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'mean', 'std'], index=metrics).rename_axis("val")
    # result_csv.to_csv(csv_filename)
    #
    # result_csv = pd.DataFrame(test_result,
    #                           columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'mean', 'std'], index=metrics).rename_axis("test")
    # result_csv.to_csv(csv_filename, mode='a')

    result_csv = pd.DataFrame(validate_result,
                              columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14','15','16','17','18','19','20','21','22','23','mean', 'std'],
                              index=metrics).rename_axis("val")
    result_csv.to_csv(csv_filename)

    result_csv = pd.DataFrame(test_result,
                              columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14','15','16','17','18','19','20','21','22','23','mean', 'std'],
                              index=metrics).rename_axis("test")
    result_csv.to_csv(csv_filename, mode='a')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument(
        '-path_to_read_result',
        default=r'D:\DingYi\EmT-release\regression\save',
        type=str, help='The root directory of the dataset.')
    args = parser.parse_args()
    csv_generator(args.path_to_read_result, num_folds=24)
