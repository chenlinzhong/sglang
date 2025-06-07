import torch
import logging
import os
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 测试配置
CONFIG_YAML = """
remote_url: eic://fdbd:dc0c:2:726::14-18512
local_addr: 127.0.0.1
local_port: 0
eic_instance_id: test_instance
eic_log_dir: /tmp/hpkv_logs
eic_log_level: 2
eic_trans_type: 3
enable_kvset_gpu_direct: False
enable_kvget_gpu_direct: True
"""

def setup_config():
    os.makedirs("/tmp/hpkv_logs", exist_ok=True)
    with open("/tmp/remote-eic.yaml", "w") as f:
        f.write(CONFIG_YAML)
    os.environ["REMOTE_EIC_YAML"] = "/tmp/remote-eic.yaml"

def test_hpkv_client_mha():
    # 设置 MHA KV 缓存
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    head_num = 32
    head_dim = 128
    layer_num = 12
    page_size = 16
    kv_cache_shape = (2, layer_num, page_size, head_num, head_dim)
    kv_cache_dtype = torch.float16

    # 初始化 device_pool
    device_pool = MHATokenToKVPool(
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        store_dtype=kv_cache_dtype,
        device=device,
    )

    # 初始化 HPKVClient
    client = HPKVClient(None, kv_cache_dtype, kv_cache_shape, device)

    # 测试数据
    keys = [f"test_key_{i}" for i in range(10)]
    values = torch.rand((10,) + kv_cache_shape, dtype=kv_cache_dtype, device=device)

    # 测试 set
    success = client.set(keys, values)
    assert success, "HPKVClient set failed"

    # 测试 exists
    exists = client.exists(keys[0])
    assert exists, f"HPKVClient exists failed for key {keys[0]}"

    # 测试 exists_batch
    exists_batch = client.exists_batch(keys)
    assert all(exists_batch), "HPKVClient exists_batch failed"

    # 测试 get
    retrieved = client.get(keys)
    assert len(retrieved) == len(keys), "HPKVClient get returned incorrect number of tensors"
    for i, (val, ret) in enumerate(zip(values, retrieved)):
        assert torch.allclose(val, ret), f"HPKVClient get data mismatch for key {keys[i]}"

    # 测试 batch_get
    retrieved_batch, success_mask = client.batch_get(keys)
    assert len(success_mask) == len(keys) and all(success_mask), "HPKVClient batch_get failed"
    assert retrieved_batch.shape == values.shape, "HPKVClient batch_get shape mismatch"
    assert torch.allclose(retrieved_batch, values), "HPKVClient batch_get data mismatch"

    logger.info("MHA test passed!")

def test_hpkv_client_mla():
    # 设置 MLA KV 缓存
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    kv_lora_rank = 64
    qk_rope_head_dim = 32
    layer_num = 12
    page_size = 16
    kv_cache_shape = (layer_num, page_size, 1, kv_lora_rank + qk_rope_head_dim)
    kv_cache_dtype = torch.float16

    # 初始化 device_pool
    device_pool = MLATokenToKVPool(
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        layer_num=layer_num,
        store_dtype=kv_cache_dtype,
        device=device,
    )

    # 初始化 HPKVClient
    client = HPKVClient(None, kv_cache_dtype, kv_cache_shape, device)

    # 测试数据
    keys = [f"test_key_mla_{i}" for i in range(5)]
    values = torch.rand((5,) + kv_cache_shape, dtype=kv_cache_dtype, device=device)

    # 测试 set
    success = client.set(keys, values)
    assert success, "HPKVClient set failed for MLA"

    # 测试 exists_batch
    exists_batch = client.exists_batch(keys)
    assert all(exists_batch), "HPKVClient exists_batch failed for MLA"

    # 测试 batch_get
    retrieved_batch, success_mask = client.batch_get(keys)
    assert len(success_mask) == len(keys) and all(success_mask), "HPKVClient batch_get failed for MLA"
    assert retrieved_batch.shape == values.shape, "HPKVClient batch_get shape mismatch for MLA"
    assert torch.allclose(retrieved_batch, values), "HPKVClient batch_get data mismatch for MLA"

    logger.info("MLA test passed!")

def main():
    setup_config()
    test_hpkv_client_mha()
    test_hpkv_client_mla()
    logger.info("All HPKVClient tests passed!")

if __name__ == "__main__":
    main()