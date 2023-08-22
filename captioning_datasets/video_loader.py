import pandas as pd
from pytube import YouTube, exceptions
from moviepy.editor import *
import subprocess
import os
import torch
import numpy as np
import http
import glob
import json
import ffmpeg

# pytube.exceptions.RegexMatchError: __init__: could not find match for ^\w+\W
# -> solution: var_regex = re.compile(r"^\$*\w+\W")
def extract(feature_type, val=False, preprocessed_file=None, type='vatex'):
    p1 = None
    files = glob.glob('./vids/*')

    for f in files:
        os.remove(f)
    batch_size = 50
    print(torch.cuda.device_count())
    path_dict = 'vids'
    if preprocessed_file is None:
        vatex_meta_dataset = pd.read_json("../data/vatex_training.json") if not val \
            else pd.read_json("../data/vatex_validation.json")
        df = vatex_meta_dataset
        df['video_id'] = df.videoID.str[:-14]
        df['caption'] = df['enCap']
        df['start'] = df.videoID.str[-13:-7].astype(int)
        df['end'] = df.videoID.str[-6:].astype(int)
    else:
        type = 'msrvtt'
        df = preprocessed_file
    video_batch = []
    counter = 0
    old_percentage = None
    leng = len(df.index)
    # df = df.sample(frac=1)
    for index, row in df.iterrows():
        percentage = "{:.0%}".format(counter / len(df.index))
        if percentage != old_percentage:
            print(percentage, 'done')
        print(counter / len(df.index), 'progress')
        if len(df.index) - counter <= batch_size:
            batch_size = 1
        old_percentage = percentage
        counter += 1
        prefix = row['video_id'] + '_{:06d}'.format(row['start']) + \
                 '_{:06d}'.format(row['end'])
        if feature_type == "vatex_i3d" or feature_type == "msrvtt_i3d":
            path_dict_flow = './data_extract/' + type + '/i3d/' + prefix + '_flow.npy'
        elif feature_type == "vatex_vggish" or feature_type == "msrvtt_vggish":
            path_dict_flow = './data_extract/' + type + '/vggish/' + prefix + '_vggish.npy'
        if os.path.exists(path_dict_flow):
            print(path_dict_flow, '  exists')
            continue
        filename = row['video_id'] + '_{:06d}'.format(row['start']) + \
                   '_{:06d}'.format(row['end'])
        filename = filename + '.mp4' if "i3d" in feature_type else filename + '.wav'
        tmp_filename = path_dict + '/' + 'tmp_' + filename
        try:
            yt = YouTube('http://youtube.com/watch?v=' + row['video_id'], use_oauth=True, allow_oauth_cache=True)
            yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if not os.path.exists(path_dict):
                print('exists')
                os.makedirs(path_dict)
            yt.download(path_dict, filename='tmp_' + filename)

            if feature_type == 'vatex_i3d' or feature_type == "msrvtt_i3d":

                clip = VideoFileClip(tmp_filename)
                clip = clip.subclip(row['start'], row['end'])
                clip.write_videofile(path_dict + '/' + filename, fps=25, audio=False)
            else:
                sound = AudioFileClip(tmp_filename)
                newsound = sound.subclip(row['start'], row['end'])  # audio from 13 to 15 seconds
                newsound.write_audiofile(path_dict + '/' + filename, 44100, 2, 2000, "pcm_s32le")
            video_batch.append("../captioning_datasets/vids/" + filename)
            print(len(video_batch))


        except (
        exceptions.AgeRestrictedError, exceptions.VideoPrivate, exceptions.VideoUnavailable, http.client.IncompleteRead,
        Exception):
            print('fail')

        if len(video_batch) >= batch_size:
            if p1 is not None:
                p1.wait()
                with open('data.txt') as f:
                    lines = f.readlines()
                    for l in lines:
                        os.remove(l[:-1])
            file_list = pd.DataFrame(video_batch, columns=['file_names'])
            np.savetxt('data.txt', file_list.values, fmt='%s')
            if feature_type == "vatex_i3d":
                p1 = subprocess.Popen("../extract_video.sh", shell=True)
            elif feature_type == "vatex_vggish":
                p1 = subprocess.Popen("../extract_video_vggish.sh", shell=True)
            elif feature_type == "msrvtt_vggish":
                p1 = subprocess.Popen("../extract_video_vggish_msr.sh", shell=True)
            elif feature_type == "msrvtt_i3d":
                p1 = subprocess.Popen("../extract_video_msr.sh", shell=True)
            video_batch = []


def get_unavailable(pre):
    vatex_dev_dataset = pd.read_json("../data/vatex_training.json")
    vatex_val_dataset = pd.read_json("../data/vatex_validation.json")
    meta_data = vatex_val_dataset.loc[:, ['videoID', 'enCap', 'start', 'end']].reset_index()
    meta_data['videoID'] = meta_data.videoID.str[:-14]
    meta_data['enCap'] = meta_data['enCap'].apply(lambda x: x[0])
    meta_data['duration'] = meta_data['end'] - meta_data['start']
    meta_data['phase'] = 'vatex_val'
    meta_data['idx'] = meta_data.index
    meta_data = meta_data.rename(columns={"videoID": "video_id", "enCap": "caption"})

    meta_data.drop(columns=["index"]).to_csv("vatex_val.csv", sep='\t', index=False)

    pre.rename(columns={"enCap": "caption"})
    dataset = pd.concat([vatex_dev_dataset, vatex_val_dataset])
    dataset['video_id'] = dataset.videoID.str[:-14]
    dataset['caption'] = dataset['enCap']
    dataset['start'] = dataset.videoID.str[-13:-7].astype(int)
    dataset['end'] = dataset.videoID.str[-6:].astype(int)
    dataset = pd.concat(([dataset, pre]))
    not_available = []
    for index, row in dataset.iterrows():
        path_dict_i3d = 'data_extract/vatex/i3d/'
        path_dict_vggish = './data_extract/vatex/vggish/'
        filename = row['video_id'] + '_{:06d}'.format(row['start']) + \
                   '_{:06d}'.format(row['end'])
        flow = path_dict_i3d + filename + '_flow.npy'
        rgb = path_dict_i3d + filename + ('_rgb.npy')
        vggish = path_dict_vggish + filename + ('_vggish.npy')
        if os.path.exists(flow):
            not_available.append(flow)
        if os.path.exists(flow):
            not_available.append(rgb)
        if os.path.exists(vggish):
            not_available.append(vggish)
    a = pd.DataFrame(not_available)
    a.to_csv("not_available_files_vatex.csv")


def remove_unnecessary():
    dirs = ['data_extract/vatex/i3d/', 'data_extract/msrvtt/i3d/']

    counter = 1
    for dir in dirs:
        test = os.listdir(dir)
        for item in test:
            counter += 1
            if item.endswith("fps.npy") or item.endswith("ms.npy"):
                os.remove(os.path.join(dir, item))


def preprocess(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        val_data = pd.read_csv('msrvtt.txt', sep=" ", header=None)
        val_data = val_data.values.squeeze()
        print(data['videos'][0])
        df = [[el['url'][32:], int(el['start time']), int(el['end time']), el['video_id'], el['split']] for el in data['videos']]
        df = pd.DataFrame(df, columns=['video_id', 'start', 'end', 'id', 'split'])
        df2 = [[el['video_id'], el['caption']] for el in data['sentences']]
        df2 = pd.DataFrame(df2, columns=['id', 'enCap', ])
        df = df.join(df2, lsuffix='id', rsuffix='id2')
        print(df.keys())
        df = df.drop(columns=["idid"])
        print(df.keys())
        val_data = df.loc[df['idid2'].isin(val_data)]
        val_data.to_csv("msrvtt_val_data.csv")
        train_data = df.loc[~df['idid2'].isin(val_data)]
        train_data.to_csv("msrvtt_train_data.csv")
        meta_data = val_data.loc[:, ['video_id', 'enCap', 'start', 'end']].reset_index()
        meta_data['duration'] = meta_data['end'] - meta_data['start']
        meta_data['phase'] = 'msrvtt_val'
        meta_data['idx'] = meta_data.index
        meta_data = meta_data.rename(columns={"enCap": "caption"})
        meta_data.drop(columns=["index"]).to_csv("msrvtt_val.csv", sep='\t', index=False)
        return df


if __name__ == "__main__":
    prepro_file = preprocess('MSRVTT_data.json')
    for i in range(5):
        extract('msrvtt_i3d', preprocessed_file=prepro_file)
        extract('msrvtt_vggish', preprocessed_file=prepro_file)
        extract('vatex_i3d', val=True)
        extract('vatex_vggish', val=True)
        extract('vatex_i3d', val=False)
        extract('vatex_vggish', val=False)
        remove_unnecessary()
        get_unavailable(prepro_file)
