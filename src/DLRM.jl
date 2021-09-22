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

include("model/model.jl")
using ._Model

include("train/train.jl")
using ._Train

include("data/criteo.jl")
include("validation.jl")
include("cachedarrays.jl")
include("playground.jl")

macro setup(eltyp, embedding_eltyp = eltyp)
    return esc(quote
        using DLRM, Zygote, Flux, HDF5, CachedArrays, OneDNN, EmbeddingTables
        manager = CachedArrays.CacheManager(
            CachedArrays.AlignedAllocator(),
            CachedArrays.AlignedAllocator();
            localsize = 80_000_000_000,
            remotesize = 80_000_000_000,
            minallocation = 21,
        )

        CachedArrays.materialize_os_pages!(manager.local_heap)
        CachedArrays.materialize_os_pages!(manager.remote_heap)

        feature_size = 128
        model = DLRM.kaggle_dlrm(
            DLRM.tocached(manager, CachedArrays.ReadWrite());
            weight_eltype = $eltyp,
            embedding_eltype = $embedding_eltyp,
            feature_size = feature_size,
            embedding_constructor = function(x)
                CachedArrays.evict!(x)
                return SimpleEmbedding{Static{feature_size}}(x)
            end,
        )

        # model = DLRM.load_hdf5(
        #     #"model_small.hdf5",
        #     #"./model_large.hdf5",
        #     DLRM.tocached(manager, CachedArrays.ReadWrite());
        #     weight_modifier = x -> convert.($eltyp, x),
        #     embedding_modifier = x -> convert.($embedding_eltyp, x),
        # )

        data = DLRM.load(DLRM.DAC(), "/home/mark/data/train.bin")
        #data = DLRM.load(DLRM.DAC(), "./train_data.bin")
        loader = DLRM.DACLoader(
            data,
            #2^13;
            2^15;
            allocator = DLRM.tocached(manager, CachedArrays.ReadWrite()),
        )

        # test_data = DLRM.load(DLRM.DAC(), "./test_data.bin")
        # test_loader = DLRM.DACLoader(
        #     test_data,
        #     2^16;
        #     allocator = DLRM.tocached(manager, CachedArrays.ReadWrite()),
        # )

        #strategy = DLRM.PreallocationStrategy{Float32}(128)
        strategy = DLRM.PreallocationStrategy{Float32}(feature_size)
        # record = Float32[]
        # _test_cb = () -> DLRM._Train.test(model, test_loader; record, strategy)
        # test_cb = DLRM._Train.Every(_test_cb, 128)
        # test_cb = DLRM._Train.Every(_test_cb, 512)

        cb = DLRM._Train.Recorder()
        loss = DLRM._Train.wrap_loss(
            DLRM._Train.bce_loss;
            strategy = DLRM.PreallocationStrategy{Float32}(feature_size),
            cb = cb,
        )

        opt = Flux.Descent(1.0)
    end)
end

end # module
