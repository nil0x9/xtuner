set -ex

# gpu_group=puyullmgpunew_gpu
# namespace=ailab-puyullmgpunew
gpu_group=puyuvlm_gpu
namespace=ailab-puyuvlm

rlaunch --gpu=8 --memory=1600000 --cpu=128 \
    --charged-group=${gpu_group} \
    --namespace ${namespace} \
    --private-machine=yes \
    --image registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/xtuner:pt28_20250911_6652194 \
    --mount=gpfs://gpfs1/intern-multi-modal-delivery:/mnt/shared-storage-user/intern-multi-modal-delivery/ \
    --mount=gpfs://gpfs1/puyullmgpu-shared:/mnt/shared-storage-user/puyullmgpu-shared \
    --mount=gpfs://gpfs1/intern7shared:/mnt/shared-storage-user/intern7shared \
    --mount=gpfs://gpfs1/puyudelivery:/mnt/shared-storage-user/puyudelivery \
    --mount=gpfs://gpfs1/gaozhangwei:/mnt/shared-storage-user/gaozhangwei \
    --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
    --mount=gpfs://gpfs1/chensitao:/mnt/shared-storage-user/chensitao \
    -- bash -c "
        pip install decord boto3 -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
        pip install /mnt/shared-storage-user/gaozhangwei/workspace_glx/petrel-oss-sdk-2.3.24.tar.gz -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
        cd /mnt/shared-storage-user/gaozhangwei/workspace_glx/xtuner
        bash
    "