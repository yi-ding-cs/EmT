import sys
import argparse

if __name__ == '__main__':
    frame_size = 48
    crop_size = 40

    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-experiment_name', default="pr_rev", help='The experiment name.')
    parser.add_argument('-gpu', default=0, type=int, help='Which gpu to use?')
    parser.add_argument('-cpu', default=1, type=int, help='How many threads are allowed?')
    parser.add_argument('-high_performance_cluster', default=0, type=int, help='On high-performance server or not?')
    parser.add_argument('-stamp', default='DY-RP', type=str, help='To indicate different experiment instances')
    parser.add_argument('-dataset', default='mahnob_hci', type=str, help='The dataset name.')
    parser.add_argument('-modality', default=["eeg_bandpower", "continuous_label"], nargs="*", help='eeg_raw, eeg_bandpower, landmark')
    parser.add_argument('-resume', default=0, type=int, help='Resume from checkpoint?')
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-num-eeg-chan', default=32, type=int)
    parser.add_argument('-num-f', default=6, type=int)


    parser.add_argument('-case', default='loso', type=str, help='trs: trial-wise shuffling, loso.: leave-one-subject-out.')
    parser.add_argument('-task', default='reg', type=str, help='reg: regression, cls: classification.')

    parser.add_argument('-debug', default=0, type=str, help='When debug=1, the fold number will be fixed to 3, because there are three subjects'
                                                            'in the debug dataset.')
    parser.add_argument('-test', default=0, type=int, help='Run test using test=1, run training using test=0')
    # Calculate mean and std for each modality?
    parser.add_argument('-calc_mean_std', default=0, type=int, help='Calculate the mean and std and save to a pickle file.')
    parser.add_argument('-cross_validation', default=1, type=int)

    # For trial-level shuffling and 10 fold subject independent
    # Not used, just a place-holder for initialization.
    # If case = loso, then num_folds = 24. If case = trs, then num_folds = 10.
    # It will be automatically set in the experiment according to the case argument. So leave it be here.
    parser.add_argument('-num_folds', default=0, type=int, help="How many folds to consider?")

    parser.add_argument('-folds_to_run', default=["all"], nargs="+", type=int, help='Which fold(s) to run in this session? If all, then run all the folds.')

    parser.add_argument('-dataset_path', default=r'D:\DingYi\Dataset\MAHNOB-P', type=str,
                        help='The root directory of the dataset.')     # change this
    parser.add_argument('-dataset_folder', default='compacted_{:d}'.format(frame_size), type=str,
                        help='The root directory of the dataset.')
    parser.add_argument('-load_path', default=r'D:\DingYi\EmT-release\regression\load\model_load_path', type=str, help='The path to load the trained model.')  # change this
    parser.add_argument('-save_path', default=r'D:\DingYi\EmT-release\regression\save', type=str, help='The path to save the trained model ')  # change this
    parser.add_argument('-python_package_path', default=r'D:\DingYi\EmT-release\regression', type=str, help='The path to the entire repository.')   # change this
    parser.add_argument('-save_model', default=1, type=int, help='Whether to save the model?')

    parser.add_argument('-normalize_eeg_raw', default=0, type=int, help='Whether to normalize eeg raw data?')

    # Models
    parser.add_argument('-bandpower_dim', default=192, type=int, help='electrodes x interest bands')
    parser.add_argument('-model_name', default="EmT", help='Model: tcn, lstm, satcn, masatcn')
    parser.add_argument('-backbone_mode', default="ir", help='Mode for resnet50 backbone: ir, ir_se')
    parser.add_argument('-backbone_state_dict_frame', default="model_state_dict_0.874", help='The filename for the backbone state dict.')
    parser.add_argument('-backbone_state_dict_eeg', default="mahnob_reg_v", help='The filename for the backbone state dict.')
    parser.add_argument('-cnn1d_embedding_dim', default=512, type=int, help='Dimensions for temporal convolutional networks feature vectors.')
    parser.add_argument('-cnn1d_channels', default=[64, 64], nargs="+", type=int, help='The specific epochs to do something.')
    parser.add_argument('-cnn1d_kernel_size', default=[3, 5, 15], help='The size of the 1D kernel for temporal convolutional networks.')
    parser.add_argument('-cnn1d_dropout', default=0.5, type=float, help='The dropout rate.')
    parser.add_argument('-lstm_embedding_dim', default=68, type=int, help='Dimensions for LSTM feature vectors.')
    parser.add_argument('-lstm_hidden_dim', default=68, type=int, help='The size of the 1D kernel for temporal convolutional networks.')
    parser.add_argument('-lstm_dropout', default=0.4, type=float, help='The dropout rate.')

    parser.add_argument('-learning_rate', default=1e-4, type=float, help='The initial learning rate.')
    parser.add_argument('-min_learning_rate', default=1e-6, type=float, help='The minimum learning rate.')

    # Groundtruth settings
    parser.add_argument('-num_classes', default=1, type=int, help='The number of classes for the dataset.')
    parser.add_argument('-emotion', default="valence", nargs="*", help='The emotion dimension to analysis.')
    parser.add_argument('-metrics', default=["rmse", "pcc", "ccc"], nargs="*", help='The evaluation metrics.')

    # Dataloader settings
    parser.add_argument('-window_length', default=96, type=int, help='The length in second to windowing the data.')
    parser.add_argument('-hop_length', default=32, type=int, help='The step size or stride to move the window.')
    parser.add_argument('-continuous_label_frequency', default=4, type=int,
                        help='The frequency of the continuous label.')
    parser.add_argument('-frame_size', default=frame_size, type=int, help='The size of the images.')
    parser.add_argument('-crop_size', default=crop_size, type=int, help='The size to conduct the cropping.')
    parser.add_argument('-batch_size', default=2, type=int)

    # Scheduler and Parameter Control
    parser.add_argument('-scheduler', default='plateau', type=str, help='plateau, cosine')
    parser.add_argument('-patience', default=5, type=int, help='Patience for learning rate changes.')
    parser.add_argument('-factor', default=0.5, type=float, help='The multiplier to decrease the learning rate.')
    parser.add_argument('-gradual_release', default=0, type=int, help='Whether to gradually release some layers?')
    parser.add_argument('-release_count', default=0, type=int, help='How many layer groups to release?')
    parser.add_argument('-milestone', default=[], nargs="+", type=int, help='The specific epochs to do something.')

    parser.add_argument('-save_plot', default=0, type=int,
                        help='Whether to plot the session-wise output/target or not?')

    args = parser.parse_args()
    sys.path.insert(0, args.python_package_path)

    from experiment import Experiment

    exp = Experiment(args)
    exp.prepare()
    exp.run()
