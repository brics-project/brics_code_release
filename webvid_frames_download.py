import cv2
import pandas as pd
import os
import numpy as np
import argparse
import requests
import concurrent.futures
from mpi4py import MPI
import warnings 
from pathlib import Path

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
RES=os.getenv('RES', 256)
SAVE_IMG_DIR=os.getenv('SAVE_IMG_DIR', 'data/save_clips')
os.makedirs(SAVE_IMG_DIR, exist_ok=True)

def request_save(url, save_fp):
    # Request and write the video
    img_data = requests.get(url, timeout=5).content
    with open(save_fp, 'wb') as handler:
        handler.write(img_data)

    # Get the image path
    save_img_path = os.path.join(SAVE_IMG_DIR, os.path.basename(save_fp).rstrip('.mp4'))

    # Iterate the video to select a random frame
    cap = cv2.VideoCapture(save_fp)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = min(48, length)
    count = 0
    while count < frame_num and cap.isOpened():
        _, frame = cap.read()
        if count == 0:
            w, h, _ = frame.shape
            crop_s = min(min(w, h), RES)
            w_start = np.random.randint(w - crop_s) if crop_s < w else 0
            h_start = np.random.randint(h - crop_s) if crop_s < h else 0
        cropped_frame = frame[h_start:h_start+crop_s, w_start:w_start+crop_s, :]
        cropped_frame = cv2.resize(cropped_frame, (RES, RES), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(save_img_path + f'_{count:07d}.png', cropped_frame)
        count += 1

    # Delete the file
    os.remove(save_fp)
    # Touch a dummy file there
    Path(save_fp).touch()

    cap.release()

def main(args):
    ### preproc
    video_dir = os.path.join(args.data_dir, 'videos')
    if RANK == 0:
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
    
    COMM.barrier()

    # ASSUMES THE CSV FILE HAS BEEN SPLIT INTO N PARTS
    partition_dir = args.csv_path.replace('.csv', f'_{args.partitions}')

    # if not, then split in this job.
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)
        full_df = pd.read_csv(args.csv_path)
        df_split = np.array_split(full_df, args.partitions)
        for idx, subdf in enumerate(df_split):
            subdf.to_csv(os.path.join(partition_dir, f'{idx}.csv'), index=False)

    relevant_fp = os.path.join(args.data_dir, 'relevant_videos_exists.txt')
    if os.path.isfile(relevant_fp):
        exists = pd.read_csv(os.path.join(args.data_dir, 'relevant_videos_exists.txt'), names=['fn'])
    else:
        exists = []

    # ASSUMES THE CSV FILE HAS BEEN SPLIT INTO N PARTS
    # data_dir/results_csvsplit/results_0.csv
    # data_dir/results_csvsplit/results_1.csv
    # ...
    # data_dir/results_csvsplit/results_N.csv


    df = pd.read_csv(os.path.join(partition_dir, f'{args.part}.csv'))

    df['rel_fn'] = df.apply(lambda x: os.path.join(str(x['page_dir']), str(x['videoid'])),
                            axis=1)

    df['rel_fn'] = df['rel_fn'] + '.mp4'

    df = df[~df['rel_fn'].isin(exists)]

    # remove nan
    df.dropna(subset=['page_dir'], inplace=True)

    playlists_to_dl = np.sort(df['page_dir'].unique())

    for page_dir in playlists_to_dl:
        vid_dir_t = os.path.join(video_dir, page_dir)
        pdf = df[df['page_dir'] == page_dir]
        if len(pdf) > 0:
            if not os.path.exists(vid_dir_t):
                os.makedirs(vid_dir_t)

            urls_todo = []
            save_fps = []

            for idx, row in pdf.iterrows():
                video_fp = os.path.join(vid_dir_t, str(row['videoid']) + '.mp4')
                if not os.path.isfile(video_fp):
                    urls_todo.append(row['contentUrl'])
                    save_fps.append(video_fp)

            print(f'Spawning {len(urls_todo)} jobs for page {page_dir}')
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.processes) as executor:
                future_to_url = {executor.submit(request_save, url, fp) for url, fp in zip(urls_todo, save_fps)}
            # request_save(urls_todo[0], save_fps[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shutter Image/Video Downloader')
    parser.add_argument('--partitions', type=int, default=4,
                        help='Number of partitions to split the dataset into, to run multiple jobs in parallel')
    parser.add_argument('--part', type=int, required=True,
                        help='Partition number to download where 0 <= part < partitions')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory where webvid data is stored.')
    parser.add_argument('--csv_path', type=str, default='results_2M_train.csv',
                        help='Path to csv data to download')
    parser.add_argument('--processes', type=int, default=8)
    args = parser.parse_args()

    if SIZE > 1:
        warnings.warn("Overriding --part with MPI rank number")
        args.part = RANK

    if args.part >= args.partitions:
        raise ValueError("Part idx must be less than number of partitions")
    main(args)