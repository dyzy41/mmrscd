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
# run_command "bash tools/dist_train.sh configs/0mmcd_dsifn/cd5net.py 2 --work-dir work_dirs_DSIFN/cd5net" $train_time
# run_command "bash tools/test.sh DSIFN configs/0mmcd_dsifn/cd5net.py 2 work_dirs_DSIFN/cd5net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_dsifn/cd5net_be.py 2 --work-dir work_dirs_DSIFN/cd5net_be" $train_time
# run_command "bash tools/test.sh DSIFN configs/0mmcd_dsifn/cd5net_be.py 2 work_dirs_DSIFN/cd5net_be" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_dsifn/cd6net.py 2 --work-dir work_dirs_DSIFN/cd6net" $train_time
# run_command "bash tools/test.sh DSIFN configs/0mmcd_dsifn/cd6net.py 2 work_dirs_DSIFN/cd6net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_dsifn/cd6net_be.py 2 --work-dir work_dirs_DSIFN/cd6net_be" $train_time
# run_command "bash tools/test.sh DSIFN configs/0mmcd_dsifn/cd6net_be.py 2 work_dirs_DSIFN/cd6net_be" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_clcd/cd8net.py 2 --work-dir work_dirs_CLCD/cd8net" $train_time
# run_command "bash tools/test.sh CLCD configs/0mmcd_clcd/cd8net.py 2 work_dirs_CLCD/cd8net" $test_time
# run_command "bash tools/dist_train.sh configs/0mmcd_clcd/cd8net_be.py 2 --work-dir work_dirs_CLCD/cd8net_be" $train_time
# run_command "bash tools/test.sh CLCD configs/0mmcd_clcd/cd8net_be.py 2 work_dirs_CLCD/cd8net_be" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_sysu/cd8net.py 2 --work-dir work_dirs_SYSU/cd8net" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_sysu/cd8net.py 2 work_dirs_SYSU/cd8net" $test_time
# run_command "bash tools/dist_train.sh configs/0mmcd_sysu/cd10net_be.py 2 --work-dir work_dirs_SYSU/cd10net_be" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_sysu/cd10net_be.py 2 work_dirs_SYSU/cd10net_be" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd8net.py 2 --work-dir work_dirs_LEVIR/cd8net" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd8net.py 2 work_dirs_LEVIR/cd8net" $test_time
# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd10net_be.py 2 --work-dir work_dirs_LEVIR/cd10net_be" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd10net_be.py 2 work_dirs_LEVIR/cd10net_be" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_cdd/cd8net.py 2 --work-dir work_dirs_CDD/cd8net" $train_time
# run_command "bash tools/test.sh CDD configs/0mmcd_cdd/cd8net.py 2 work_dirs_CDD/cd8net" $test_time
# run_command "bash tools/dist_train.sh configs/0mmcd_cdd/cd10net_be.py 2 --work-dir work_dirs_CDD/cd10net_be" $train_time
# run_command "bash tools/test.sh CDD configs/0mmcd_cdd/cd10net_be.py 2 work_dirs_CDD/cd10net_be" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd11net.py 2 --work-dir work_dirs_LEVIR/cd11net_v3" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd11net.py 2 work_dirs_LEVIR/cd11net_v3" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_sysu/cd11net.py 2 --work-dir work_dirs_SYSU/cd11net" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_sysu/cd11net.py 2 work_dirs_SYSU/cd11net" $test_time
# run_command "bash tools/dist_train.sh configs/0mmcd_cdd/cd11net.py 2 --work-dir work_dirs_CDD/cd11net" $train_time
# run_command "bash tools/test.sh CDD configs/0mmcd_cdd/cd11net.py 2 work_dirs_CDD/cd11net" $test_time
# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd11net.py 2 --work-dir work_dirs_LEVIR/cd11net" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd11net.py 2 work_dirs_LEVIR/cd11net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_hrcus/cd9net.py 2 --work-dir work_dirs_HRCUS/cd9net --resume" $train_time
# run_command "bash tools/test.sh HRCUS configs/0mmcd_hrcus/cd9net.py 2 work_dirs_HRCUS/cd9net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_hrcus/cd9net_base.py 2 --work-dir work_dirs_HRCUS/cd9net_base" $train_time
# run_command "bash tools/test.sh HRCUS configs/0mmcd_hrcus/cd9net_base.py 2 work_dirs_HRCUS/cd9net_base" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_hrcus/cd9net_ab1.py 2 --work-dir work_dirs_HRCUS/cd9net_ab1" $train_time
# run_command "bash tools/test.sh HRCUS configs/0mmcd_hrcus/cd9net_ab1.py 2 work_dirs_HRCUS/cd9net_ab1" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_hrcus/cd9net_ab2.py 2 --work-dir work_dirs_HRCUS/cd9net_ab2" $train_time
# run_command "bash tools/test.sh HRCUS configs/0mmcd_hrcus/cd9net_ab2.py 2 work_dirs_HRCUS/cd9net_ab2" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_sysu/cd11net.py 2 --work-dir work_dirs_SYSU/cd11net" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_sysu/cd11net.py 2 work_dirs_SYSU/cd11net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levirplus/cd11net.py 2 --work-dir work_dirs_LEVIRPLUS/cd11net" $train_time
# run_command "bash tools/test.sh LEVIRPLUS configs/0mmcd_levirplus/cd11net.py 2 work_dirs_LEVIRPLUS/cd11net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_cdd/cd11net.py 2 --work-dir work_dirs_CDD/cd11net" $train_time
# run_command "bash tools/test.sh CDD configs/0mmcd_cdd/cd11net.py 2 work_dirs_CDD/cd11net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_sysu/cd12net.py 2 --work-dir work_dirs_SYSU/cd12net" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_sysu/cd12net.py 2 work_dirs_SYSU/cd12net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd12net.py 2 --work-dir work_dirs_LEVIR/cd12net" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd12net.py 2 work_dirs_LEVIR/cd12net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_clcd/cd13net.py 2 --work-dir work_dirs_CLCD/cd13net" $train_time
# run_command "bash tools/test.sh CLCD configs/0mmcd_clcd/cd13net.py 2 work_dirs_CLCD/cd13net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd15net.py 2 --work-dir work_dirs_LEVIR/cd15net" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd15net.py 2 work_dirs_LEVIR/cd15net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd18net.py 2 --work-dir work_dirs_LEVIR/cd18net" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd18net.py 2 work_dirs_LEVIR/cd18net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd19net.py 2 --work-dir work_dirs_LEVIR/cd19net" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd19net.py 2 work_dirs_LEVIR/cd19net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd20net.py 2 --work-dir work_dirs_LEVIR/cd20net_v2" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd20net.py 2 work_dirs_LEVIR/cd20net_v2" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd21net.py 2 --work-dir work_dirs_LEVIR/cd21net" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd21net.py 2 work_dirs_LEVIR/cd21net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd22net.py 2 --work-dir work_dirs_LEVIR/cd22net" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd22net.py 2 work_dirs_LEVIR/cd22net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd23net.py 2 --work-dir work_dirs_LEVIR/cd23net" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd23net.py 2 work_dirs_LEVIR/cd23net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd24net.py 2 --work-dir work_dirs_LEVIR/cd24net" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd24net.py 2 work_dirs_LEVIR/cd24net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd25net.py 2 --work-dir work_dirs_LEVIR/cd25net" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd25net.py 2 work_dirs_LEVIR/cd25net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/cd26net.py 2 --work-dir work_dirs_LEVIR/cd26net" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/cd26net.py 2 work_dirs_LEVIR/cd26net" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/EfficientCD.py 2 --work-dir work_dirs_LEVIR/EfficientCD" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/EfficientCD.py 2 work_dirs_LEVIR/EfficientCD" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/EfficientCD0.py 2 --work-dir work_dirs_LEVIR/EfficientCD0" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/EfficientCD0.py 2 work_dirs_LEVIR/EfficientCD0" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/EfficientCD1.py 2 --work-dir work_dirs_LEVIR/EfficientCD1" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/EfficientCD1.py 2 work_dirs_LEVIR/EfficientCD1" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/EfficientCD2.py 2 --work-dir work_dirs_LEVIR/EfficientCD2" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/EfficientCD2.py 2 work_dirs_LEVIR/EfficientCD2" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_levir/EfficientCD01.py 2 --work-dir work_dirs_LEVIR/EfficientCD01" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_levir/EfficientCD01.py 2 work_dirs_LEVIR/EfficientCD01" $test_time


# run_command "bash tools/dist_train.sh configs/0mmcd_sysu/EfficientCD.py 2 --work-dir work_dirs_SYSU/EfficientCD" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_sysu/EfficientCD.py 2 work_dirs_SYSU/EfficientCD" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_sysu/EfficientCD0.py 2 --work-dir work_dirs_SYSU/EfficientCD0" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_sysu/EfficientCD0.py 2 work_dirs_SYSU/EfficientCD0" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_sysu/EfficientCD1.py 2 --work-dir work_dirs_SYSU/EfficientCD1" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_sysu/EfficientCD1.py 2 work_dirs_SYSU/EfficientCD1" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_sysu/EfficientCD2.py 2 --work-dir work_dirs_SYSU/EfficientCD2" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_sysu/EfficientCD2.py 2 work_dirs_SYSU/EfficientCD2" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_sysu/EfficientCD01.py 2 --work-dir work_dirs_SYSU/EfficientCD01" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_sysu/EfficientCD01.py 2 work_dirs_SYSU/EfficientCD01" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/cgnet_clcd.py 2 --work-dir work_dirs_compare/cgnet_clcd" $train_time
# run_command "bash tools/test.sh CLCD configs/0mmcd_compare/cgnet_clcd.py 2 work_dirs_compare/cgnet_clcd" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/cgnet_levir.py 2 --work-dir work_dirs_compare/cgnet_levir" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_compare/cgnet_levir.py 2 work_dirs_compare/cgnet_levir" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/cgnet_LEVIR.py 2 --work-dir work_dirs_compare/cgnet_LEVIR" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_compare/cgnet_LEVIR.py 2 work_dirs_compare/cgnet_LEVIR" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/hcgmnet_clcd.py 2 --work-dir work_dirs_compare/hcgmnet_clcd" $train_time
# run_command "bash tools/test.sh CLCD configs/0mmcd_compare/hcgmnet_clcd.py 2 work_dirs_compare/hcgmnet_clcd" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/hcgmnet_levir.py 2 --work-dir work_dirs_compare/hcgmnet_levir" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_compare/hcgmnet_levir.py 2 work_dirs_compare/hcgmnet_levir" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/hcgmnet_LEVIR.py 2 --work-dir work_dirs_compare/hcgmnet_whucd" $train_time
# run_command "bash tools/test.sh WHUCD configs/0mmcd_compare/hcgmnet_whucd.py 2 work_dirs_compare/hcgmnet_whucd" $test_time


# run_command "bash tools/dist_train.sh configs/0mmcd_compare/cdnet_clcd.py 2 --work-dir work_dirs_compare/cdnet_clcd" $train_time
# run_command "bash tools/test.sh CLCD configs/0mmcd_compare/cdnet_clcd.py 2 work_dirs_compare/cdnet_clcd" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/cdnet_levir.py 2 --work-dir work_dirs_compare/cdnet_levir" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_compare/cdnet_levir.py 2 work_dirs_compare/cdnet_levir" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/cdnet_whucd.py 2 --work-dir work_dirs_compare/cdnet_whucd" $train_time
# run_command "bash tools/test.sh WHUCD configs/0mmcd_compare/cdnet_whucd.py 2 work_dirs_compare/cdnet_whucd" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/cdnet_sysu.py 2 --work-dir work_dirs_compare/cdnet_sysu" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_compare/cdnet_sysu.py 2 work_dirs_compare/cdnet_sysu" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/mscanet_clcd.py 2 --work-dir work_dirs_compare/mscanet_clcd" $train_time
# run_command "bash tools/test.sh CLCD configs/0mmcd_compare/mscanet_clcd.py 2 work_dirs_compare/mscanet_clcd" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/mscanet_levir.py 2 --work-dir work_dirs_compare/mscanet_levir" $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_compare/mscanet_levir.py 2 work_dirs_compare/mscanet_levir" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/mscanet_whucd.py 2 --work-dir work_dirs_compare/mscanet_whucd" $train_time
# run_command "bash tools/test.sh WHUCD configs/0mmcd_compare/mscanet_whucd.py 2 work_dirs_compare/mscanet_whucd" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/mscanet_sysu.py 2 --work-dir work_dirs_compare/mscanet_sysu" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_compare/mscanet_sysu.py 2 work_dirs_compare/mscanet_sysu" $test_time


# run_command "bash tools/dist_train.sh configs/0mmcd_compare/cdnet.py 2 \
#                         --work-dir work_dirs_compare/cdnet_levir \
#                         --cfg-options train_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
#                         --cfg-options val_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
#                         --cfg-options test_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
#                         --cfg-options train_cfg.max_iters=30000 \
#                         --cfg-options train_cfg.val_interval=2000 \
#                         " $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_compare/cdnet.py 2 work_dirs_compare/cdnet_levir" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/cdnet.py 2 \
#                         --work-dir work_dirs_compare/cdnet_whucd \
#                         --cfg-options train_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
#                         --cfg-options val_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
#                         --cfg-options test_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
#                         --cfg-options train_cfg.max_iters=20000 \
#                         --cfg-options train_cfg.val_interval=2000 \
#                         " $train_time
# run_command "bash tools/test.sh WHUCD configs/0mmcd_compare/cdnet.py 2 work_dirs_compare/cdnet_whucd" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/cdnet.py 2 \
#                         --work-dir work_dirs_compare/cdnet_sysu \
#                         --cfg-options train_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
#                         --cfg-options val_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
#                         --cfg-options test_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
#                         --cfg-options train_cfg.max_iters=20000 \
#                         --cfg-options train_cfg.val_interval=2000 \
#                         " $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_compare/cdnet.py 2 work_dirs_compare/cdnet_sysu" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/cdnet_slide.py 2 \
#                         --work-dir work_dirs_compare/cdnet_clcd \
#                         --cfg-options data_root="$CDPATH/CLCD" \
#                         --cfg-options train_cfg.max_iters=20000 \
#                         --cfg-options train_cfg.val_interval=2000 \
#                         " $train_time
# run_command "bash tools/test.sh CLCD configs/0mmcd_compare/cdnet_slide.py 2 work_dirs_compare/cdnet_clcd --cfg-options data_root="$CDPATH/CLCD"" $test_time



# run_command "bash tools/dist_train.sh configs/0mmcd_compare/mscanet.py 2 \
#                         --work-dir work_dirs_compare/mscanet_levir \
#                         --cfg-options data_root="$CDPATH/LEVIR-CD/cut_data" \
#                         --cfg-options train_cfg.max_iters=30000 \
#                         --cfg-options train_cfg.val_interval=2000 \
#                         " $train_time
# run_command "bash tools/test.sh LEVIR configs/0mmcd_compare/mscanet.py 2 work_dirs_compare/mscanet_levir --cfg-options data_root="$CDPATH/LEVIR-CD/cut_data"" $test_time


# run_command "bash tools/dist_train.sh configs/0mmcd_compare/mscanet.py 2 \
#                         --work-dir work_dirs_compare/mscanet_whucd \
#                         --cfg-options train_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
#                         --cfg-options val_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
#                         --cfg-options test_dataloader.dataset.data_root="$CDPATH/WHUCD/cut_data" \
#                         --cfg-options train_cfg.max_iters=20000 \
#                         --cfg-options train_cfg.val_interval=2000 \
#                         " $train_time
# run_command "bash tools/test.sh WHUCD configs/0mmcd_compare/mscanet.py 2 work_dirs_compare/mscanet_whucd" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/mscanet.py 2 \
#                         --work-dir work_dirs_compare/mscanet_sysu \
#                         --cfg-options train_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
#                         --cfg-options val_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
#                         --cfg-options test_dataloader.dataset.data_root="$CDPATH/SYSU-CD" \
#                         --cfg-options train_cfg.max_iters=20000 \
#                         --cfg-options train_cfg.val_interval=2000 \
#                         " $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_compare/mscanet.py 2 work_dirs_compare/mscanet_sysu" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_compare/mscanet_slide.py 2 \
#                         --work-dir work_dirs_compare/mscanet_clcd \
#                         --cfg-options data_root="$CDPATH/CLCD" \
#                         --cfg-options train_cfg.max_iters=20000 \
#                         --cfg-options train_cfg.val_interval=2000 \
#                         " $train_time
# run_command "bash tools/test.sh CLCD configs/0mmcd_compare/mscanet_slide.py 2 work_dirs_compare/mscanet_clcd --cfg-options data_root="$CDPATH/CLCD"" $test_time


# run_command "bash tools/dist_train.sh configs/0mmcd_backbone/EfficientCD_B0.py 2 --work-dir work_dirs_SYSU/EfficientCD_B0" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_backbone/EfficientCD_B0.py 2 work_dirs_SYSU/EfficientCD_B0" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_backbone/EfficientCD_B1.py 2 --work-dir work_dirs_SYSU/EfficientCD_B1" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_backbone/EfficientCD_B1.py 2 work_dirs_SYSU/EfficientCD_B1" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_backbone/EfficientCD_B2.py 2 --work-dir work_dirs_SYSU/EfficientCD_B2" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_backbone/EfficientCD_B2.py 2 work_dirs_SYSU/EfficientCD_B2" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_backbone/EfficientCD_B3.py 2 --work-dir work_dirs_SYSU/EfficientCD_B3" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_backbone/EfficientCD_B3.py 2 work_dirs_SYSU/EfficientCD_B3" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_backbone/EfficientCD_B4.py 2 --work-dir work_dirs_SYSU/EfficientCD_B4" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_backbone/EfficientCD_B4.py 2 work_dirs_SYSU/EfficientCD_B4" $test_time

# run_command "bash tools/dist_train.sh configs/0mmcd_backbone/EfficientCD_B5.py 2 --work-dir work_dirs_SYSU/EfficientCD_B5" $train_time
# run_command "bash tools/test.sh SYSU configs/0mmcd_backbone/EfficientCD_B5.py 2 work_dirs_SYSU/EfficientCD_B5" $test_time


run_command "bash tools/dist_train.sh configs/0mmcd_backbone_resnet/EffiCDResnet18.py 2 \
                        --work-dir work_dirs_LEVIR/EffiCDResnet18 \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options train_cfg.max_iters=30000 \
                        --cfg-options train_cfg.val_interval=2000 \
                        --resume
                        " $train_time
run_command "bash tools/test.sh LEVIR configs/0mmcd_backbone_resnet/EffiCDResnet18.py 2 work_dirs_LEVIR/EffiCDResnet18" $test_time

run_command "bash tools/dist_train.sh configs/0mmcd_backbone_resnet/EffiCDResnet34.py 2 \
                        --work-dir work_dirs_LEVIR/EffiCDResnet34 \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options train_cfg.max_iters=30000 \
                        --cfg-options train_cfg.val_interval=2000 \
                        --resume
                        " $train_time
run_command "bash tools/test.sh LEVIR configs/0mmcd_backbone_resnet/EffiCDResnet34.py 2 work_dirs_LEVIR/EffiCDResnet34" $test_time

run_command "bash tools/dist_train.sh configs/0mmcd_backbone_resnet/EffiCDResnet50.py 2 \
                        --work-dir work_dirs_LEVIR/EffiCDResnet50 \
                        --cfg-options train_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options val_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options test_dataloader.dataset.data_root="$CDPATH/LEVIR-CD/cut_data" \
                        --cfg-options train_cfg.max_iters=30000 \
                        --cfg-options train_cfg.val_interval=2000 \
                        --resume
                        " $train_time
run_command "bash tools/test.sh LEVIR configs/0mmcd_backbone_resnet/EffiCDResnet50.py 2 work_dirs_LEVIR/EffiCDResnet50" $test_time
