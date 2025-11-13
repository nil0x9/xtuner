from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import InternS1VLTokenizeFnConfig, PretrainTokenizeFunctionConfig
from xtuner.v1.model import InternVL3P5Dense8BConfig
from xtuner.v1.model.compose.internvl import InternVLVisionConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
import json
import os
import shutil

# 路径配置
ceph_config = "/mnt/shared-storage-user/gaozhangwei/workspace_glx/petreloss.conf"
meta_data_path = '/mnt/shared-storage-user/gaozhangwei/workspace_glx/data/export_meta_internvl3_5_cpt_tiny.json'
model_path = "/mnt/shared-storage-user/intern7shared/wangweiyun/OpenGVLab-rc1-hf/InternVL3_5-8B-INIT-HF" # 转换后的权重（hf官方格式）
work_dir = "/mnt/shared-storage-user/intern7shared/internvl_a4s/xtuner_saved_model/internvl3.5/internvl3.5-8B-cpt-tiny-bs96-epoch1-lr1e-5"
tokenizer_cache_dir = "/mnt/shared-storage-user/intern7shared/internvl_a4s/xtuner_tokenizer_cache/internvl3.5/slow_tokenize_cpt_ml_32k_tokenizer"

# 将当前配置文件拷贝到work_dir
if not os.path.exists(work_dir):
    os.makedirs(work_dir, exist_ok=True)
current_file = __file__
shutil.copy(current_file, work_dir)

# 训练超参数
sample_max_length = 32768
pack_max_length = 32768
num_workers = 8
min_num_frames = 4
max_num_frame = 24
global_batch_size = 96
total_epoch = 1
hf_interval = 1000
checkpoint_interval = 1000
checkpoint_maxkeep = 15
lr = 1e-5
lr_min = 1e-5
weight_decay = 0.05
warmup_ratio = 0.03
recompute_ratio = 1.0
drop_path_rate = 0.1
loss_reduction = "square"

# model config
model_cfg = InternVL3P5Dense8BConfig(vision_config=InternVLVisionConfig(drop_path_rate=drop_path_rate))

# dataset config
if ceph_config is not None:
    oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": ceph_config})
else:
    oss_loader_cfg = None

has_pretrain = False
ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    if _data.get('text_pretrain', False) or has_pretrain:
        has_pretrain = True
    
    # VLMJsonlDataset -> soft_pack 多个样本pack到一起，不会对样本内部进行拆分处理
    # JsonlDataset -> hard_pack 多个样本pack到一起，样本过长，会对样本内部进行拆分处理，目前只用于处理纯文本预训练数据
    # 可能存在问题的点：纯文本预训练数据不会和其他数据一起pack到一起
    class_name = 'JsonlDataset' if _data.get('text_pretrain', False) else 'VLMJsonlDataset'
    
    if _data.get('text_pretrain', False):
        tokenize_fn = PretrainTokenizeFunctionConfig(hash=_data.get('hash', None))
    else:
        tokenize_fn = InternS1VLTokenizeFnConfig(model_cfg=model_cfg,
                                                 max_length=sample_max_length,
                                                 max_dynamic_patch=_data.get('max_dynamic_patch',
                                                                             None),
                                                 min_dynamic_patch=_data.get('min_dynamic_patch',
                                                                             None),
                                                 min_num_frames=_data.get('min_num_frames', min_num_frames),
                                                 max_num_frames=_data.get('max_num_frames', max_num_frame),
                                                 data_augment=_data.get('data_augment', False),
                                                 system_message=_data.get('system_message', None),
                                                 hash=_data.get('hash', None),
                                                 oss_loader_cfg=oss_loader_cfg,
                                                 template_name="internvl-3.5",
                                                 debug=False,
                                                 oss_time_log_thr=10)
    
    _data_cfg = {"dataset": DatasetConfig(name=name,
                                          anno_path=_data['annotation'],
                                          media_root=_data.get('media_root', ''),
                                          sample_ratio=_data.get('sample_ratio', 1.0),
                                          class_name=class_name,
                                          enable_sequential_sampler=True, # 使用顺序采样，确保同样的sample ratio采样的样本是相同的
                                          cache_tag='cache_tags_v1',
                                          cache_dir=tokenizer_cache_dir),
                 "tokenize_fn": tokenize_fn
                 }
    dataset_config.append(_data_cfg)

if has_pretrain:
    pack_level = 'mllm_hybrid'
else:
    pack_level = 'soft'

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_level=pack_level,
    pack_to_max_length=False,
    collator="intern_s1_vl_sft_collator",
    num_workers=num_workers,
    pack_extra_buffer_size=20,
)
# optimizer and lr config
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(sp_size=1, recompute_ratio=recompute_ratio, torch_compile=True,
                      checkpoint_preserve_rng_state=False)

resume_cfg = ResumeConfig(auto_resume=True)

# trainer config
trainer = TrainerConfig(
    load_from=model_path,
    resume_cfg=resume_cfg,
    tokenizer_path=model_path,
    fsdp_cfg=fsdp_cfg,
    exp_tracker='tensorboard',
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024, loss_reduction=loss_reduction),
    global_batch_size=global_batch_size,
    total_epoch=total_epoch,
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    work_dir=work_dir,
)
