eval "$(conda shell.bash hook)"
conda activate torch_zoo

cd ../video_features
# extract r(2+1)d features for the sample videos
python main.py \
    feature_type='i3d'\
    device="cuda:0" \
    file_with_video_paths="../captioning_datasets/data.txt" on_extraction=save_numpy output_path="../captioning_datasets/data_extract/msrvtt"