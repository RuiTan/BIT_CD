#!/usr/bin/env bash
gpus=0
checkpoint_root=checkpoints
data_name=LEVIR
loss=ce

img_size=256
batch_size=16
lr=0.01
max_epochs=300
net_G=base_transformer_pos_s4_dd8_dedim8
#base_resnet18
#base_transformer_pos_s4_dd8
#base_transformer_pos_s4_dd8_dedim8
lr_policy=linear

split=train
split_val=val
output_sigmoid=0
project_name=CD_${loss}loss_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}_sig${output_sigmoid}

python main_cd.py --loss ${loss} --output_sigmoid ${output_sigmoid} --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}