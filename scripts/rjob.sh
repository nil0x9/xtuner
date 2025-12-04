set -ex

# gpu_group=puyuvlm_gpu
# namespace=ailab-puyuvlm
# num_gpus=48
# num_nodes=$((num_gpus / 8))
# # job_name=sft-internvl35-8b-tiny-old-cap
# # config_file=examples/v1/internvl/sft_internvl3.5_8B_config_tiny_old_cap.py
# # job_name=sft-internvl35-8b-tiny-new-cap
# # config_file=examples/v1/internvl/sft_internvl3.5_8B_config_tiny_new_cap.py
# # job_name=sft-internvl35-8b-tiny-old-new-mix-cap
# # config_file=examples/v1/internvl/sft_internvl3.5_8B_config_tiny_old_new_mix_cap.py
# # job_name=sft-internvl35-8b-tiny-old-old-mix-cap
# # config_file=examples/v1/internvl/sft_internvl3.5_8B_config_tiny_old_old_mix_cap.py
# job_name=cpt-internvl35-8b-tiny-old-caption-resume
# config_file=examples/v1/internvl/cpt_internvl3.5_8B_config_tiny_caption.py

# gpu_group=puyullmgpunew_gpu
# namespace=ailab-puyullmgpunew
# num_gpus=512
# num_nodes=$((num_gpus / 8))
# job_name=cpt-internvl35-8b-resume-try
# config_file=examples/v1/internvl/cpt_internvl3.5_8B_config.py

# gpu_group=puyuvlm_gpu
# namespace=ailab-puyuvlm
# num_gpus=64
# num_nodes=$((num_gpus / 8))
# # job_name=cpt-internvl35-8b-tiny-based-mlp
# # config_file=examples/v1/internvl/cpt_internvl3.5_8B_config_tiny.py
# job_name=cpt-internvl35-8b-tiny-lr-decay
# config_file=examples/v1/internvl/cpt_internvl3.5_8B_config_tiny_lr_decay.py

# gpu_group=puyuvlm_gpu
# namespace=ailab-puyuvlm
# num_gpus=32
# num_nodes=$((num_gpus / 8))
# job_name=sft-internvl35-8b-tiny-llavaonevision-data5
# config_file=examples/v1/internvl/sft_internvl3.5_8B_config_tiny_llavaonevision.py

# gpu_group=puyuvlm_gpu
# namespace=ailab-puyuvlm
# num_gpus=32
# num_nodes=$((num_gpus / 8))
# # job_name=sft-internvl35-8b-tiny-based-cpt-tiny-resume
# # config_file=examples/v1/internvl/sft_internvl3.5_8B_config_tiny_based_cpt_tiny.py
# # job_name=sft-internvl35-8b-tiny-based-cpt-tiny-caption1
# # config_file=examples/v1/internvl/sft_internvl3.5_8B_config_tiny_based_cpt_tiny_caption.py
# job_name=sft-internvl35-8b-tiny-based-cpt-tiny-mlp-resume2
# config_file=examples/v1/internvl/sft_internvl3.5_8B_config_tiny_based_cpt_tiny.py

# gpu_group=puyuvlm_gpu
# namespace=ailab-puyuvlm
# num_gpus=32
# num_nodes=$((num_gpus / 8))
# job_name=mlp-internvl35-8b-resume
# config_file=examples/v1/internvl/mlp_internvl3.5_8B_config_tiny.py

# gpu_group=puyullmgpunew_gpu
# namespace=ailab-puyullmgpunew
gpu_group=puyuvlm_gpu
namespace=ailab-puyuvlm
num_gpus=8
num_nodes=$((num_gpus / 8))
# job_name=cpt-tiny-v2
# config_file=examples/v1/internvl/cpt_internvl3.5_8B_config_tiny.py
# job_name=sft-tiny-v2
# config_file=examples/v1/internvl/sft_internvl3.5_8B_config_tiny_based_cpt_tiny.py
job_name=qwen3vl-8b-cpt-tiny
config_file=examples/v1/qwenvl/cpt_qwen3vl_8B_config_tiny.py
# job_name=sft-qwen3vl-8b-tiny-based-cpt-tiny
# config_file=examples/v1/qwenvl/sft_qwen3vl_8B_config_tiny_based_cpt_tiny.py

rjob submit \
    --name=${job_name} \
    --gpu=8 --memory=1200000 --cpu=96 \
    --charged-group=${gpu_group} \
    --namespace ${namespace} \
    --private-machine=group \
    -P ${num_nodes} \
    --image registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/xtuner:pt28_20250911_6652194 \
    --mount=gpfs://gpfs1/intern-multi-modal-delivery:/mnt/shared-storage-user/intern-multi-modal-delivery/ \
    --mount=gpfs://gpfs1/puyullmgpu-shared:/mnt/shared-storage-user/puyullmgpu-shared \
    --mount=gpfs://gpfs1/gaozhangwei:/mnt/shared-storage-user/gaozhangwei \
    --mount=gpfs://gpfs1/intern7shared:/mnt/shared-storage-user/intern7shared \
    --mount=gpfs://gpfs1/chensitao:/mnt/shared-storage-user/chensitao \
    --host-network=true \
    --gang-start=true \
    --custom-resources rdma/mlnx_shared=8  \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    -e DISTRIBUTED_JOB=true \
    -- bash -c '
    pip install transformers==4.57.0 -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
    pip install decord boto3 -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
    pip install /mnt/shared-storage-user/gaozhangwei/workspace_glx/petrel-oss-sdk-2.3.24.tar.gz -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
    cd /mnt/shared-storage-user/gaozhangwei/workspace_glx/xtuner
    bash scripts/sft_intern_s1_vl_entrypoint.sh $0' "${config_file}"
