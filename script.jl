using DLRM, CachedArrays, Flux

manager = CachedArrays.CacheManager(
    "/mnt/pm1/public/";
    localsize = 150_000_000_000,
    remotesize = 20_000_000_000,
    gc_before_evict = true,
)
CachedArrays.materialize_os_pages!(manager.dram_heap)

model = DLRM.kaggle_dlrm(DLRM.tocached(manager))
data = DLRM.load(DLRM.DAC(), "/mnt/data1/dac/train.bin")
loader = DLRM.DACLoader(data, 2048; allocator = DLRM.tocached(manager))
opt = Flux.Descent(0.1)
loss = DLRM._Train.wrap_loss(DLRM._Train.bce_loss; strategy = DLRM.PreallocationStrategy(128))

# The main thing we want to profile
DLRM._Train.train!(loss, model, loader, opt)
