## Create conda environment

You need anaconda/miniconda to create a proper environment by:
```
conda env create --file=environment.yml && conda activate brics_release
```


## Dataset

To prepare dataset of FFHQ and LSUN-Church, please refer to [this link](https://github.com/NVlabs/stylegan2-ada-pytorch). We would construct the dataset to a zip file.

For the Webvid-Frames200k dataset we construct from [Webvid-10M](https://github.com/m-bain/webvid). We use a modified script `webvid_frames_download.py` to download frames.

```bash
RES=256 SAVE_IMG_DIR=$SAVE_FOLDER_NAME python webvid_frames_download.py --csv_path $CSV_FILE --partitions 1 --part 0 --data_dir $ROOT_OF_DATA_DIR --processes 8
```
Please check the original [Webvid-10M](https://github.com/m-bain/webvid) repository to get the csv file for different partions.

## Training

### Reconstruction Stage
Run following script to train the autoencoder model of reconstruction.
```bash
bash run.sh $EXP_NAME $NUM_GPU $BATCH_SIZE $DATASET_FILE --gamma=4 --table_size_log2=18 --level_dim=4 --feat_coord_dim=4 --img_snap=2 --init_res=64 --style_dim=512 --img_size=256 --table_num=16 --res_min=16 --init_dim=512 --tile_coord=true --encoder_flag=true --mini_linear_n_layers=3 --disable_patch_gan=true --feat_coord_dim_per_table=1 --num_downsamples=2 --additional_decoder_conv=true --use_kl_reg=false --noise_perturb=true --attn_resolutions 64 --grid_type="tile"
```

| Args      | Symbol in the paper |
| ----------- | ----------- |
| level_dim      | $C_r$       |
| table_num | $N_r$ | 
| feat_coord_dim   | $C_z$        |
| init_res | $H_d,W_d$ | 
| init_dim | $C_d$ |
| feat_coord_dim_per_table | $C_{key}$ |

For options like resuming, VQGAN decoder or MoVQ decoder from a third-party implementation, please refer to `train.py`.

During training, all the logs, intermediate and network checkpoints results will be saved under `training_runs/$EXP_NAME`.

### Diffusion Model Training

Run the following scripts for diffusion training
```bash
python diffusions/train.py --exp_id=$EXP_ID --batch_size=64 --encoder_decoder_network=$AUTOENCODER_PKL --dataset=$DATASET_ZIP_FILE --dim=256 --sample_num=16 --record_k=1 --train_lr=8e-5 --feat_spatial_size=$KEY_CODE_SPATIAL_SIZE --num_resnet_blocks='2,2,2,2' --no_noise_perturb=true --use_min_snr false --noise_scheduler cosine_variant_v2 --cosine_decay_max_steps=1000000 --dim_mults '1,2,3,4' --atten_layers '2,3,4' --snap_k 1280 --sample_k 1280
```
 The `KEY_CODE_SPATIAL_SIZE` is the spatial size of the dumped key codes. Note that it is the size before tiling operation (namely $H_z,W_z$). For more information please refer to `diffusions/train.py`.


## Calculate Metrics
All the metrics calculation scripts are run in the root of this repository.

### Reconstruction Metrics
Run following command to test the reconstruction metrics 

```bash
python measure_recon_error.py --network $NETWORK_PKL_PATH --dataset $VALIDATION_DATASET_ZIP_FILE --outdir $EXPORTED_FOLDER
```
After running, a file `${EXPORTED_FOLDER}/metric.txt` will be dumped that record the LPIPS/PSNR/SSIM metrics. Note that, in default, all reconstruction results will be saved. Use `--max_save` to specify how many images wil be saved and `--runs` to specify if running repeated for more than one time. For more information please refer to the script itself.

### FID/CLIP-FID/Inception Score and Generate Images
Run the following command to test the FID/CLIP-FID/Inception Score.

```bash
python measure_gen_metrics.py --real_data $REAL_IMAGE_PATH --input_folder $GENERATED_IMAGE_PATH --exp_name $EXP_NAME --which_fid [fid,clip_fid]
```

At each run, the script will try to find real statistics cache file under `metrics_cache` folder (`clean-fid` will find the cache in a different local folder, see this [link](https://github.com/GaParmar/clean-fid/blob/main/cleanfid/features.py#L61-L69)). If it cannot find it, it will calculate that. After running, a file `metrics_cache/${EXP_NAME}_metric_result.txt` will be dumped.

You can also generate samples while calculating metrics by providing `--network_ae` and `--network_diff`. For more options, please refer to the script itself.


### Precision/Recall Score

We modified the original [third-party implementation](https://github.com/youngjung/improved-precision-and-recall-metric-pytorch/tree/master) by python multiprocessing and [CuPy](https://github.com/cupy/cupy) to solve the memory hungary issue in the original implementation (although you may still need > 32GB memory.). **You need to first install CuPy**. Note that since we would spawn multiple processes where CuPy is used, there will be multiple VGG network being copied and run. Please decrease the variable `$PR_PROCESS_NUM` if you find you have exceeded GPU memory limitation. Run the following command to calculate the precision/recall score.

```bash
# Calculate the statistics of the whole real dataset
PR_PROCESS_NUM=4 python improved_precision_recall.py $REAL_IMAGE_PATH $GENERATED_IMAGE_PATH --only_precalc

# Calculate the Precision/Recall scores on 50000 generated images
PR_PROCESS_NUM=4 python improved_precision_recall.py $REAL_IMAGE_PATH $GENERATED_IMAGE_PATH --num_samples 50000
```

The `$REAL_IMAGE_PATH` and `$GENERATED_IMAGE_PATH` can either be the root folder of images or a zip file. At the first run, the script will cache the statistics of real images under `metrics_cache` for reuse. The name of this cache will be determined by the real image path/file name. For more options please refer to the script itself.
