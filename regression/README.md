# Prepare the data
Download MAHNOB-HCI dataset [here](https://mahnob-db.eu/hci-tagging/). And set the data folder as the root_directory in configs.py, e.g., /home/dingyi/MAHNOB/. This folder should contains two subfolders, ./Sessions/ and ./Subjects/.

Get the continuous label in this [repo](https://github.com/soheilrayatdoost/ContinuousEmotionDetection). Put the lable_continous_Mahnob.mat at the data folder, e.g., /home/dingyi/MAHNOB/lable_continous_Mahnob.mat

Note that it might pop some error messages when you create the dataset by using generate_dataset.py. It is because there are some format errors in the original data. You can identify the file according to the error message and correct the format error in that file.

The exact folder/files to be edit include:

A. Sessions/1200/P10-Rec1-All-Data-New_Section_30.tsv - `Remove Line 3169-3176 as their format is broken`.

B. Sessions/1854 - `Remove this trial folder as it does not contain EEG recordings`.

C. Sessions/1984 - `Remove this trial folder as it does not contain EEG recordings`.

# Run the code
Step 1: Check the config.py file first and change the parameters accordingly. Mainly, update the `"root_directory"` and `"output_root_directory"` according to your data location.

Step 2: Run generate_dataset.py.

Step 3: Check the parameters in the main.py file and change them accordingly. Mainly, update the `"-dataset_path"`, `"-load_path"`, `"-save_path"`, and `"-python_package_path"` according to your local directory.

Step 4: Run main.py to train and evaluate the network.

Step 5: Using generate_results_csv.py to get the summarized results.

Please add `pip install chardet` if you received an error saying "ImportError: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS' from 'charset_normalizer.constant'" when running `main.py`.

