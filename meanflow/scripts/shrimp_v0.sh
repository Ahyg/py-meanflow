export CUDA_VISIBLE_DEVICES=1
torchrun --standalone --nproc_per_node=1 --master_port=12345 \
    train.py \
    --output_dir "./tmp_2" \
    --dataset "shrimp" \
    --batch_size "4" \
    --lr 0.0006 \
    --eval_frequency 2000 \
    --epochs 500 \
    --compute_fid \
    --log_per_step 100 \
    --tr_sampler "v1" \
    --P_mean_t "-0.6" \
    --P_std_t 1.6 \
    --P_mean_r -4.0 \
    --P_std_r 1.6 \
    --warmup_epochs 50 \
    --norm_p 0.75 \
    --ratio 0.75 \
    --dropout 0.2 \
    --use_edm_aug \
    --fid_samples 1000 \
    --sat-files-path "/mnt/ssd_1/yghu/Data/71" \
    --radar-files-path "/mnt/ssd_1/yghu/Data/71" \
    --coverage-threshold 0.1 \
    --seed 42 \
    --block-size 96 \
    --split-ratio "(0.7,0.2,0.1)" \
    --retrieve-dataset