import pandas as pd
from pytube import YouTube, exceptions
from moviepy.editor import *
import subprocess
import os
import torch
import numpy as np
import glob
import ffmpeg


def extract(feature_type, val):
    files = glob.glob('./vids/*')
    for f in files:
        os.remove(f)
    batch_size = 50
    print(torch.cuda.device_count())
    path_dict = 'vids'
    vatex_meta_dataset = pd.read_json("../data/vatex_training.json") if not val\
        else pd.read_json("../data/vatex_validation.json")
    df = vatex_meta_dataset
    df['video_id'] = df.videoID.str[:-14]
    df['caption'] = df['enCap']
    df['start'] = df.videoID.str[-13:-7].astype(int)
    df['end'] = df.videoID.str[-6:].astype(int)
    not_available = []
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
        if feature_type == "i3d":
            path_dict_flow = './data_extract/i3d/' + prefix + '_flow.npy'
        elif feature_type == "vggish":
            path_dict_flow = './data_extract/vggish/' + prefix + '_vggish.npy'
        if os.path.exists(path_dict_flow):
            continue
        filename = row['video_id'] + '_{:06d}'.format(row['start']) + \
                   '_{:06d}'.format(row['end'])
        filename = filename + '.mp4' if feature_type == "i3d" else filename + '.wav'
        tmp_filename = path_dict + '/' + 'tmp_' + filename
        try:
            yt = YouTube('http://youtube.com/watch?v=' + row['video_id'], use_oauth=True, allow_oauth_cache=True)
            yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if not os.path.exists(path_dict):
                print('exists')
                os.makedirs(path_dict)
            yt.download(path_dict, filename='tmp_' + filename)

            if feature_type == 'i3d':

                clip = VideoFileClip(tmp_filename)
                clip = clip.subclip(row['start'], row['end'])
                clip.write_videofile(path_dict + '/' + filename, fps=25, audio=False)
            else:
                sound = AudioFileClip(tmp_filename)
                newsound = sound.subclip(row['start'], row['end'])  # audio from 13 to 15 seconds
                newsound.write_audiofile(path_dict + '/' + filename, 44100, 2, 2000, "pcm_s32le")
            if os.path.exists(path_dict + '/tmp_' + filename):
                os.remove(path_dict + '/tmp_' + filename)
            video_batch.append("../captioning_datasets/vids/" + filename)
            print(len(video_batch))


        except (exceptions.AgeRestrictedError, exceptions.VideoPrivate, exceptions.VideoUnavailable, KeyError, Exception):
            print('fail')
        if len(video_batch) >= batch_size:
            file_list = pd.DataFrame(video_batch, columns=['file_names'])
            np.savetxt('data.txt', file_list.values, fmt='%s')
            if feature_type == "i3d":
                p1 = subprocess.Popen("../extract_video.sh", shell=True)
            else:
                p1 = subprocess.Popen("../extract_video_vggish.sh", shell=True)
            p1.wait()
            for name in video_batch:
                os.remove(name)
            video_batch = []
def get_unavailable():
    vatex_dev_dataset = pd.read_json("../data/vatex_training.json")
    vatex_val_dataset = pd.read_json("../data/vatex_validation.json")
    dataset = pd.concat([vatex_dev_dataset, vatex_val_dataset])
    dataset['video_id'] = dataset.videoID.str[:-14]
    dataset['caption'] = dataset['enCap']
    dataset['start'] = dataset.videoID.str[-13:-7].astype(int)
    dataset['end'] = dataset.videoID.str[-6:].astype(int)
    not_available = []
    for index, row in dataset.iterrows():
        path_dict_i3d = './data_extract/i3d/'
        path_dict_vggish = './data_extract/vggish/'
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
    dir = './data_extract/i3d/'
    test = os.listdir(dir)
    counter = 1
    for item in test:
        print(counter/len(test), "%")
        counter += 1
        if item.endswith("fps.npy") or item.endswith("ms.npy"):
            os.remove(os.path.join(dir, item))
if __name__ == "__main__":
    for i in range(3):
        extract('i3d', val=True)
        extract('vggish', val=True)
        extract('i3d', val=False)
        extract('vggish', val=False)
        remove_unnecessary()
        get_unavailable()