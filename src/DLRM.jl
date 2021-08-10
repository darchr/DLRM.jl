module DLRM

export DLRMModel, dlrm

# stdlib
using LinearAlgebra: LinearAlgebra
using Mmap: Mmap
using SparseArrays: SparseArrays
using Random: Random
using Serialization: Serialization
import Statistics: mean

# "Internal" dependencies
using CachedArrays
using EmbeddingTables
using OneDNN: OneDNN

# External Dependencies
using ChainRulesCore: ChainRulesCore
using ConstructionBase: ConstructionBase
using DataStructures: DataStructures
using Flux: Flux
using HDF5: HDF5
using NaturalSort: NaturalSort
using Polyester: Polyester
using ProgressMeter: ProgressMeter
import UnPack: @unpack
import Zygote

include("utils/utils.jl")
using ._Utils

# include("embedding/embedding.jl")
# using ._EmbeddingTables

include("model/model.jl")
using ._Model

include("train/train.jl")
using ._Train

include("data/criteo.jl")
include("validation.jl")
include("cachedarrays.jl")
include("playground.jl")

macro setup()
    return quote
        using DLRM, Zygote, Flux, HDF5, CachedArrays, OneDNN
        manager = CachedArrays.CacheManager(
            CachedArrays.AlignedAllocator(),
            CachedArrays.MmapAllocator("/mnt/pm1/public/");
            localsize = 50_000_000_000,
            remotesize = 100_000_000_000,
            minallocation = 21,
            #telemetry = CachedArrays.Telemetry(),
        )
        CachedArrays.materialize_os_pages!(manager.local_heap)

        # weight_init = function (x...)
        #     data = DLRM.tocached(manager)(x...)
        #     DLRM._Model.multithread_init(DLRM._Model.GlorotNormal(), data)
        #     return data
        # end

        # tables = DLRM._Model.create_embeddings(
        #     # Embedding Constructor
        #     DLRM.SimpleEmbedding{DLRM.Static{128}},
        #     # Sparse feature size
        #     128,
        #     # Embedding Sizes
        #     fill(1_000_000, 26),
        #     # Initializer
        #     weight_init,
        # )
        model = DLRM.kaggle_dlrm(DLRM.tocached(Float32, manager))
        data = DLRM.load(DLRM.DAC(), "/mnt/data1/dac/train.bin")
        loader = DLRM.DACLoader(
            data, 2^15; allocator = DLRM.tocached(Float32, manager, CachedArrays.ReadWrite())
        );
        loss = DLRM._Train.wrap_loss(
            #DLRM._Train.bce_loss; strategy = DLRM.SimpleParallelStrategy()
            DLRM._Train.bce_loss; strategy = DLRM.PreallocationStrategy(128)
        )
        opt = Flux.Descent(0.1)
    end |> esc
end
end # module
