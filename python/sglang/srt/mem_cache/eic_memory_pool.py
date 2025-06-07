import logging
import os
import threading
import time
from typing import List, Optional, Tuple

import hpkv
import torch
import yaml

from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MemoryStateInt,
    MHATokenToKVPool,
    MLATokenToKVPool,
    debug_timing,
    synchronized,
)

logger = logging.getLogger(__name__)
TensorPoolSize = 1024

REMOTE_EIC_YAML_ENV_VAR = "REMOTE_EIC_YAML"

# GPU Direct RDMA for kv set
G_EnableKVSetGPUDirect = False

# GPU Direct RDMA for kv get
G_EnableKVGetGPUDirect = True


class FlexibleKVCacheMemoryPool:
    def __init__(self, client, device: str, kv_cache_shape, kv_cache_dtype):
        self._init = False
        self.client = client  # Using HPKVTensorClient instead of eic.Client
        self.device = device if G_EnableKVGetGPUDirect else "cpu"

        """ (num_layer, 2, chunk_size, num_kv_head, head_size) """
        self.kv_cache_shape = kv_cache_shape
        self.kv_cache_dtype = kv_cache_dtype

        self.max_kv_cache_num = TensorPoolSize * 2

        self.mempool = torch.zeros(
            (self.max_kv_cache_num,) + kv_cache_shape,
            dtype=kv_cache_dtype,
            device=device,
        )
        self.kv_cache_idx = 0

        self.kv_cache_numel = 1
        for i in self.kv_cache_shape:
            self.kv_cache_numel *= i

        # Register memory with HPKVClient
        self.reg_buf_bytes = self.mempool.numel() * self.mempool.element_size()
        self.reg = self.client.reg_memory(self.mempool.data_ptr(), self.reg_buf_bytes)
        self.sgl = hpkv.SGL(self.mempool.data_ptr(), self.reg_buf_bytes, self.reg)

        logger.info(
            f"register memory pool shape {self.kv_cache_shape}, dtype {self.kv_cache_dtype}, "
            f"kv_cache_num {self.max_kv_cache_num}, device {device}, "
            f"total_size {self.max_kv_cache_num * (self.mempool[0].numel() * self.mempool[0].element_size())}"
        )

    def try_allocate_kv_cache(self, shape, dtype, count=1):
        if self.kv_cache_dtype != dtype or self.kv_cache_shape != shape:
            logger.error(
                f"allocate from mempool failed, self.kv_cache_shape {self.kv_cache_shape}, "
                f"dtype {self.kv_cache_dtype}, require shape {shape}, dtype {dtype}"
            )
            return None

        if count > self.max_kv_cache_num:
            logger.error(
                f"allocate from mempool failed, self.kv_cache_shape {self.kv_cache_shape}, "
                f"dtype {self.kv_cache_dtype}, require count {count}, max_kv_cache_num {self.max_kv_cache_num}"
            )
            return None

        if self.kv_cache_idx + count > self.max_kv_cache_num:
            self.kv_cache_idx = 0

        ret = self.mempool[self.kv_cache_idx : self.kv_cache_idx + count]
        self.kv_cache_idx = (self.kv_cache_idx + count) % self.max_kv_cache_num
        return ret

    def __del__(self):
        self.client.dereg_memory(self.reg)


class HPKVClient:
    """
    The remote url should start with "eic://" and only have one host-port pair
    """

    def __init__(self, endpoint: str, kv_cache_dtype, kv_cache_shape, device="cpu"):
        if os.environ.get(REMOTE_EIC_YAML_ENV_VAR) is not None:
            logger.info(f"hpkv init with env var {REMOTE_EIC_YAML_ENV_VAR}")
            config_file = os.environ.get(REMOTE_EIC_YAML_ENV_VAR)
        else:
            config_file = "/sgl-workspace/config/remote-eic.yaml"
            logger.info(f"hpkv init with default config, config_file {config_file}")

        if not os.path.exists(config_file):
            logger.error(f"config file {config_file} not exists")
            exit(1)

        with open(config_file, "r") as fin:
            config = yaml.safe_load(fin)

        remote_url = config.get("remote_url", None)
        if remote_url is None:
            raise AssertionError("remote_url is None")

        endpoint = remote_url[len("eic://") :]
        raddr, rport = endpoint.split("-")
        rport = int(rport)
        laddr = config.get("local_addr", None)
        lport = config.get("local_port", 0)

        logger.info(f"hpkv remote_url: {remote_url}, endpoint: {endpoint}")

        eic_instance_id = config.get("eic_instance_id", None)
        logger.info(f"hpkv instance_id: {eic_instance_id}")

        eic_log_dir = config.get("eic_log_dir", None)
        logger.info(f"hpkv log_dir: {eic_log_dir}")

        eic_log_level = config.get("eic_log_level", 2)
        logger.info(f"hpkv log_level: {eic_log_level}")

        if not os.path.exists(eic_log_dir) and not os.path.isdir(eic_log_dir):
            os.makedirs(eic_log_dir, exist_ok=True)

        self.client = hpkv.HPKVTensorClient(raddr, rport, laddr, lport, 1)
        if not self.client:
            logger.error("fail to init hpkv client")
            exit(1)

        self.device = device
        self.kv_cache_shape = kv_cache_shape
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_mem_pool = FlexibleKVCacheMemoryPool(
            self.client,
            self.device if G_EnableKVGetGPUDirect else "cpu",
            self.kv_cache_shape,
            self.kv_cache_dtype,
        )
        self.kv_cache_write_mem_pool = FlexibleKVCacheMemoryPool(
            self.client,
            self.device if G_EnableKVSetGPUDirect else "cpu",
            self.kv_cache_shape,
            self.kv_cache_dtype,
        )

    def __del__(self):
        self.client.close()

    def exists(self, key: str) -> bool:
        logger.debug(f"hpkv exists {key}")
        ret = self.client.test(key)
        if ret:
            logger.debug(f"hpkv exists {key} success")
        else:
            logger.debug(f"hpkv exists {key} failed")
        return ret

    def exists_batch(self, keys: List[str]) -> List[bool]:
        logger.debug(f"hpkv exists {len(keys)}")
        res = []
        for key in keys:
            ret = self.client.test(key)
            res.append(ret)
            if not ret:
                logger.debug(f"hpkv exists {key} failed")
        return res

    def get(self, keys: List[str]) -> Optional[torch.Tensor]:
        logger.debug(f"hpkv get {keys}")
        get_data_start_time = time.perf_counter()
        objs = []

        for key in keys:
            dtype = self.kv_cache_dtype
            shape = self.kv_cache_shape
            logger.debug(f"get tensor shape {shape}, dtype {dtype}")

            item = self.kv_cache_mem_pool.try_allocate_kv_cache(shape, dtype)
            if item is None:
                obj = torch.empty(shape, dtype=dtype, device="cpu")
                logger.error("can not allocate tensor from pool")
            else:
                obj = item
            ret = self.client.get(key, obj)
            if ret != 0:
                logger.error(f"hpkv get {key} failed, ret {ret}")
                return None
            logger.debug(f"hpkv get data {key} success")
            objs.append(obj)

        get_data_end_time = time.perf_counter()
        get_data_execution_time = (get_data_end_time - get_data_start_time) * 1e6
        logger.debug(f"hpkv get {keys} data cost %.2f ms", get_data_execution_time * 1e3)

        return objs

    def batch_get(self, keys: List[str]) -> Tuple[Optional[torch.Tensor], Optional[List[bool]]]:
        logger.debug(f"hpkv get {len(keys)}")
        get_data_start_time = time.perf_counter()
        count = len(keys)
        success_mask = [True for _ in range(count)]
        objs = self.kv_cache_mem_pool.try_allocate_kv_cache(
            self.kv_cache_shape, self.kv_cache_dtype, count
        )
        if objs is None:
            objs = torch.empty(
                (count,) + self.kv_cache_shape, dtype=self.kv_cache_dtype, device="cpu"
            )
            logger.error("can not allocate tensor from pool")

        for i, key in enumerate(keys):
            ret = self.client.get(key, objs[i])
            if ret != 0:
                logger.error(f"hpkv get data {key} failed, ret {ret}")
                success_mask[i] = False
            else:
                logger.debug(f"hpkv get data {key} success")

        get_data_end_time = time.perf_counter()
        get_data_execution_time = (get_data_end_time - get_data_start_time) * 1e6
        logger.debug(f"hpkv get {count} keys data cost %.2f us", get_data_execution_time)
        return objs, success_mask

    def set(self, keys: List[str], obj_inputs: torch.Tensor) -> bool:
        logger.debug(f"hpkv set {len(keys)} keys")
        count = len(keys)

        items = self.kv_cache_write_mem_pool.try_allocate_kv_cache(
            self.kv_cache_shape, self.kv_cache_dtype, count
        )
        if items is None:
            objs = torch.empty(
                (count,) + self.kv_cache_shape, dtype=self.kv_cache_dtype, device="cpu"
            )
            logger.error("can not allocate tensor from pool")
        else:
            objs = items

        success = True
        for i, key in enumerate(keys):
            temp = objs[i].reshape(obj_inputs[i].shape).contiguous()
            temp.copy_(obj_inputs[i])
            ret = self.client.set(key, temp)
            if ret != 0:
                logger.error(f"hpkv set {key} failed, ret {ret}")
                success = False
            else:
                logger.debug(f"hpkv set {key} success")

        return success


class EICBaseTokenToKVPoolHost:
    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float = 4.0,
        host_size: int = 10,
        device: str = "cpu",
        page_size: int = 1,
        rank: int = 0,
        extra_info: Optional[dict] = None,
    ):
        self.device_pool = device_pool
        self.host_to_device_ratio = host_to_device_ratio
        self.device = device
        self.dtype = device_pool.store_dtype
        self.page_size = page_size
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        self.size = self.size - (self.size % self.page_size)

        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int32)
        self.can_use_mem_size = self.size
        self.lock = threading.RLock()
        self.debug = logger.isEnabledFor(logging.DEBUG)
        self.rank = rank
        self.host_ip = self._get_host_ip()
        self.split_dim = 2
        self.extra_info = extra_info
        self.deploy_key = self._get_deploy_info()

    def _encode_key_exclusive(self, indices):
        return [
            f"{self.host_ip}_{self.rank}_{index}"
            for index in indices.to("cpu").tolist()
        ]

    def _get_host_ip(self):
        import socket
        return socket.gethostbyname(socket.gethostname())

    def _get_deploy_info(self):
        model_path = self.extra_info.get("model_path", "fake_model_path")
        world_size = self.extra_info.get("world_size", 1)
        rank = self.extra_info.get("tp_rank", 0)
        page_size = self.page_size
        framework = self.extra_info.get("framework", "sglang")
        deploy_key = f"{model_path}_{world_size}_{rank}_{page_size}@{framework}"
        return deploy_key

    def _encode_key_shared(self, content_hashs):
        return [f"{content_hash}@{self.deploy_key}" for content_hash in content_hashs]

    def get_flat_data(self, indices) -> Tuple[Optional[torch.Tensor], List[bool]]:
        logger.debug(f"get_flat_data indices {indices}")
        keys = self._encode_key_exclusive(indices)
        bs = TensorPoolSize
        ret = []
        masks = []

        for i in range(0, len(keys), bs):
            key = keys[i : i + bs]
            objs, success_mask = self.eic_client.batch_get(key)
            if objs is None:
                logger.error(f"get_flat_data keys {key} failed, hpkv_client return none")
                return None, []
            copy_objs = objs.clone()
            ret.extend([copy_objs[i] for i in range(copy_objs.shape[0])])
            masks.extend(success_mask)

        if len(ret) == 0:
            logger.error(
                f"get_flat_data keys size {len(keys)} failed, hpkv_client return none, ret {ret}"
            )
            return None, []

        flat_data = torch.cat(ret, dim=self.split_dim)
        return flat_data, masks

    def assign_flat_data(self, indices, flat_data):
        logger.debug(f"assign_flat_data indices {indices}")
        start_time = time.perf_counter()

        keys = self._encode_key_exclusive(indices)
        flat_data = flat_data.contiguous()
        if not G_EnableKVSetGPUDirect:
            values = torch.split(flat_data.cpu(), 1, dim=self.split_dim)
        else:
            values = torch.split(flat_data, 1, dim=self.split_dim)

        bs = TensorPoolSize
        split_time = time.perf_counter()
        for i in range(0, len(keys), bs):
            key = keys[i : i + bs]
            value = values[i : i + bs]
            ret = self.eic_client.set(key, value)
            if not ret:
                logger.error(
                    f"assign_flat_data keys {key} failed, hpkv_client return none"
                )
                return False
        cost_time = time.perf_counter() - split_time
        if cost_time > 1:
            logger.warning(
                f"finish assign flat data, total keys {len(keys)}, split time {split_time - start_time}, transfer time {cost_time}"
            )
        return True

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def exist_page(self, content_hashs):
        keys = self._encode_key_shared(content_hashs)
        ret = self.eic_client.exists_batch(keys)
        res = []
        for i, exist in enumerate(ret):
            if exist:
                res.append(content_hashs[i])
            else:
                break
        return res

    def get_page_data(self, content_hashs):
        logger.debug(f"get_page_data content_hashs {content_hashs}")
        keys = self._encode_key_shared(content_hashs)
        bs = TensorPoolSize
        ret = []
        masks = []

        for i in range(0, len(keys), bs):
            key = keys[i : i + bs]
            objs, success_mask = self.eic_client.batch_get(key)
            if objs is None:
                logger.error(f"get_page_data keys {key} failed, hpkv_client return none")
                return None, []
            copy_objs = objs.clone()
            ret.extend([copy_objs[i] for i in range(copy_objs.shape[0])])
            masks.extend(success_mask)

        if len(ret) == 0:
            logger.error(
                f"get_page_data keys size {len(keys)} failed, hpkv_client return none, ret {ret}"
            )
            return None, []

        flat_data = torch.cat(ret, dim=self.split_dim)
        return flat_data, masks

    def assign_page_data(self, content_hashs, flat_data):
        logger.debug(f"assign_page_data hashs {content_hashs}")
        keys = self._encode_key_shared(content_hashs)
        flat_data = flat_data.contiguous()
        values = torch.split(flat_data, self.page_size, dim=self.split_dim)
        bs = TensorPoolSize

        for i in range(0, len(keys), bs):
            key = keys[i : i + bs]
            value = values[i : i + bs]
            ret = self.eic_client.set(key, value)
            if not ret:
                logger.error(
                    f"assign_page_data keys {key} failed, hpkv_client return none"
                )
                return False
        return True

    @debug_timing
    def transfer(self, indices, flat_data):
        return self.assign_flat_data(indices, flat_data)

    @synchronized()
    def clear(self):
        self.mem_state.fill_(0)
        self.can_use_mem_size = self.size
        self.free_slots = torch.arange(self.size, dtype=torch.int32)

    @synchronized()
    def get_state(self, indices: torch.Tensor) -> MemoryStateInt:
        assert len(indices) > 0, "The indices should not be empty"
        states = self.mem_state[indices]
        assert (
            states == states[0]
        ).all(), "The memory slots should have the same state {}".format(states)
        return MemoryStateInt(states[0].item())

    @synchronized()
    def alloc(self, need_size: int) -> torch.Tensor:
        if need_size > self.can_use_mem_size:
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        self.mem_state[select_index] = MemoryStateInt.RESERVED
        self.can_use_mem_size -= need_size
        return select_index

    @synchronized()
    def is_reserved(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.RESERVED

    @synchronized()
    def is_protected(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.PROTECTED

    @synchronized()
    def is_synced(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.SYNCED

    @synchronized()
    def is_backup(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.BACKUP

    @synchronized()
    def update_backup(self, indices: torch.Tensor):
        assert self.is_synced(indices) or (
            self.page_size > 1 and self.is_reserved(indices)
        ), (
            f"The host memory slots should be in SYNCED state before turning into BACKUP. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.BACKUP

    @synchronized()
    def update_synced(self, indices: torch.Tensor):
        self.mem_state[indices] = MemoryStateInt.SYNCED

    @synchronized()
    def protect_write(self, indices: torch.Tensor):
        assert self.is_reserved(indices), (
            f"The host memory slots should be RESERVED before write operations. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized()
    def protect_load(self, indices: torch.Tensor):
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized()
    def complete_io(self, indices: torch.Tensor):
        assert self.is_protected(indices), (
            f"The host memory slots should be PROTECTED during I/O operations. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.SYNCED

    def available_size(self):
        return len(self.free_slots)

    @synchronized()
    def free(self, indices: torch.Tensor) -> int:
        self.mem_state[indices] = MemoryStateInt.IDLE
        self.free_slots = torch.concat([self.free_slots, indices])
        self.can_use_mem_size += len(indices)
        return len(indices)


class EICMHATokenToKVPoolHost(EICBaseTokenToKVPoolHost):
    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        device: str = "cpu",
        page_size: int = 1,
        rank: int = 0,
        extra_info: Optional[dict] = None,
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            device,
            page_size,
            rank,
            extra_info,
        )
        self.head_num = device_pool.head_num
        self.head_dim = device_pool.head_dim
        self.layer_num = device_pool.layer_num
        self.size_per_token = (
            self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2
        )
        self.kvcache_shape = (
            2,
            self.layer_num,
            page_size,
            self.head_num,
            self.head_dim,
        )
        self.eic_client = HPKVClient(
            None, self.dtype, self.kvcache_shape, device_pool.device
        )


class EICMLATokenToKVPoolHost(EICBaseTokenToKVPoolHost):
    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        device: str = "cpu",
        page_size: int = 1,
        rank: int = 0,
        extra_info: Optional[dict] = None,
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            device,
            page_size,
            rank,
            extra_info,
        )
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num
        self.size_per_token = (
            (self.kv_lora_rank + self.qk_rope_head_dim) * 1 * self.dtype.itemsize
        )
        self.kvcache_shape = (
            self.layer_num,
            page_size,
            1,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )
        self.eic_client = HPKVClient(
            None, self.dtype, self.kvcache_shape, device_pool.device
        )
        self.split_dim = 1

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num
        return (
            (self.kv_lora_rank + self.qk_rope_head_dim)
            * 1
            * self.dtype.itemsize
            * self.layer_num
        )