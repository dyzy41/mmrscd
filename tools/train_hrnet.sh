#!/usr/bin/env bash

# 函数：执行指定的命令，如果执行时间小于30分钟，则重试
run_command() {
    local cmd=$1
    local min_time=$2 # 最小时间，以秒为单位

    while true; do
        local start_time=$(date +%s)

        eval $cmd

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        if [ $duration -ge $min_time ]; then
            echo "命令执行成功，耗时 $(($duration / 60)) 分钟."
            break
        else
            echo "命令执行时间少于指定的 $(($min_time / 60)) 分钟，重试..."
            sleep 5  # 短暂休眠后重试
        fi
    done
}

# 使用 run_command 函数执行每个实验
# 参数：完整的命令字符串和最小执行时间（秒）
train_time=$((10 * 60))  # 10分钟
test_time=$((10))  # 20秒

# 使用 run_command 函数执行每个实验
# 参数：完整的命令字符串和重试间隔时间（秒）


run_command "bash tools/dist_train.sh configs/0mmcd_backbone_hrnet/hrnet.py 2 \
                        --work-dir work_dirs_LEVIR/hrnet_w18 \
                        --cfg-options model.model_name='hrnet_w18' \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options train_cfg.max_iters=30000 \
                        --cfg-options train_cfg.val_interval=2000 \
                        --resume
                        " $train_time
run_command "bash tools/test.sh LEVIR configs/0mmcd_backbone_hrnet/hrnet.py 2 work_dirs_LEVIR/hrnet_w18 \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options model.model_name='hrnet_w18'" $test_time

run_command "bash tools/dist_train.sh configs/0mmcd_backbone_hrnet/hrnet.py 2 \
                        --work-dir work_dirs_LEVIR/hrnet_w32 \
                        --cfg-options model.model_name='hrnet_w32' \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options train_cfg.max_iters=30000 \
                        --cfg-options train_cfg.val_interval=2000
                        " $train_time
run_command "bash tools/test.sh LEVIR configs/0mmcd_backbone_hrnet/hrnet.py 2 work_dirs_LEVIR/hrnet_w32 \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options model.model_name='hrnet_w32'" $test_time

run_command "bash tools/dist_train.sh configs/0mmcd_backbone_hrnet/hrnet.py 2 \
                        --work-dir work_dirs_LEVIR/hrnet_w48 \
                        --cfg-options model.model_name='hrnet_w48' \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options train_cfg.max_iters=30000 \
                        --cfg-options train_cfg.val_interval=2000
                        " $train_time
run_command "bash tools/test.sh LEVIR configs/0mmcd_backbone_hrnet/hrnet.py 2 work_dirs_LEVIR/hrnet_w48 \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options model.model_name='hrnet_w48'" $test_time



run_command "bash tools/dist_train.sh configs/0mmcd_backbone_hrnet/hrnet.py 2 \
                        --work-dir work_dirs_SYSU/hrnet_w18 \
                        --cfg-options model.model_name='hrnet_w18' \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options train_cfg.max_iters=20000 \
                        --cfg-options train_cfg.val_interval=2000
                        " $train_time
run_command "bash tools/test.sh SYSU configs/0mmcd_backbone_hrnet/hrnet.py 2 work_dirs_SYSU/hrnet_w18 \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options model.model_name='hrnet_w18'" $test_time

run_command "bash tools/dist_train.sh configs/0mmcd_backbone_hrnet/hrnet.py 2 \
                        --work-dir work_dirs_SYSU/hrnet_w32 \
                        --cfg-options model.model_name='hrnet_w32' \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options train_cfg.max_iters=20000 \
                        --cfg-options train_cfg.val_interval=2000
                        " $train_time
run_command "bash tools/test.sh SYSU configs/0mmcd_backbone_hrnet/hrnet.py 2 work_dirs_SYSU/hrnet_w32 \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options model.model_name='hrnet_w32'" $test_time

run_command "bash tools/dist_train.sh configs/0mmcd_backbone_hrnet/hrnet.py 2 \
                        --work-dir work_dirs_SYSU/hrnet_w48 \
                        --cfg-options model.model_name='hrnet_w48' \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options train_cfg.max_iters=20000 \
                        --cfg-options train_cfg.val_interval=2000
                        " $train_time
run_command "bash tools/test.sh SYSU configs/0mmcd_backbone_hrnet/hrnet.py 2 work_dirs_SYSU/hrnet_w48 \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
                        --cfg-options model.model_name='hrnet_w48'" $test_time



run_command "bash tools/dist_train.sh configs/0mmcd_backbone_hrnet/hrnet.py 2 \
                        --work-dir work_dirs_WHUCD/hrnet_w18 \
                        --cfg-options model.model_name='hrnet_w18' \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options train_cfg.max_iters=20000 \
                        --cfg-options train_cfg.val_interval=2000
                        " $train_time
run_command "bash tools/test.sh WHUCD configs/0mmcd_backbone_hrnet/hrnet.py 2 work_dirs_WHUCD/hrnet_w18 \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options model.model_name='hrnet_w18'" $test_time

run_command "bash tools/dist_train.sh configs/0mmcd_backbone_hrnet/hrnet.py 2 \
                        --work-dir work_dirs_WHUCD/hrnet_w32 \
                        --cfg-options model.model_name='hrnet_w32' \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options train_cfg.max_iters=20000 \
                        --cfg-options train_cfg.val_interval=2000
                        " $train_time
run_command "bash tools/test.sh WHUCD configs/0mmcd_backbone_hrnet/hrnet.py 2 work_dirs_WHUCD/hrnet_w32 \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options model.model_name='hrnet_w32'" $test_time

run_command "bash tools/dist_train.sh configs/0mmcd_backbone_hrnet/hrnet.py 2 \
                        --work-dir work_dirs_WHUCD/hrnet_w48 \
                        --cfg-options model.model_name='hrnet_w48' \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options train_cfg.max_iters=20000 \
                        --cfg-options train_cfg.val_interval=2000
                        " $train_time
run_command "bash tools/test.sh WHUCD configs/0mmcd_backbone_hrnet/hrnet.py 2 work_dirs_WHUCD/hrnet_w48 \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
                        --cfg-options model.model_name='hrnet_w48'" $test_time



run_command "bash tools/dist_train.sh configs/0mmcd_backbone_hrnet/hrnet_slide.py 2 \
                        --work-dir work_dirs_CLCD/hrnet_w18 \
                        --cfg-options model.model_name='hrnet_w18' \
                        --cfg-options train_cfg.max_iters=20000 \
                        --cfg-options train_cfg.val_interval=2000
                        " $train_time
run_command "bash tools/test.sh CLCD configs/0mmcd_backbone_hrnet/hrnet_slide.py 2 work_dirs_CLCD/hrnet_w18 \
                        --cfg-options model.model_name='hrnet_w18'" $test_time

run_command "bash tools/dist_train.sh configs/0mmcd_backbone_hrnet/hrnet_slide.py 2 \
                        --work-dir work_dirs_CLCD/hrnet_w32 \
                        --cfg-options model.model_name='hrnet_w32' \
                        --cfg-options train_cfg.max_iters=20000 \
                        --cfg-options train_cfg.val_interval=2000
                        " $train_time
run_command "bash tools/test.sh CLCD configs/0mmcd_backbone_hrnet/hrnet_slide.py 2 work_dirs_CLCD/hrnet_w32 \
                        --cfg-options model.model_name='hrnet_w32'" $test_time

run_command "bash tools/dist_train.sh configs/0mmcd_backbone_hrnet/hrnet_slide.py 2 \
                        --work-dir work_dirs_CLCD/hrnet_w48 \
                        --cfg-options model.model_name='hrnet_w48' \
                        --cfg-options train_cfg.max_iters=20000 \
                        --cfg-options train_cfg.val_interval=2000
                        " $train_time
run_command "bash tools/test.sh CLCD configs/0mmcd_backbone_hrnet/hrnet_slide.py 2 work_dirs_CLCD/hrnet_w48 \
                        --cfg-options model.model_name='hrnet_w48'" $test_time