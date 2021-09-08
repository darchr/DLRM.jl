# General Criteo Dataset Layout

#####
##### DAC
#####

# Preprocessing Pipeline
#
# - Step 1: The raw dataset is stored as plain text with a label, 13 continuous features,
# and 26 categorical features.
# The first step is to convert this to a binary representation that can be directly memory
# mapped into an application.
#
# This is the job of the "binarize" function.
# Two additional things about the "binarize" function.
#
# * It applies the log transform to the continous variables, converting them from
# 32 bit integers to to 32 bit floating point numbers.
#
# * Continuous features that are missing are converted correctly to zero.
#
# - Step 2: We then need to run a pass through the dataset to construct a conversion from
# the indices used in the dataset to a number between `1` and `N` where `N` is the number
# of unique indices for a particular feature.
#
# Since the dataset may be split over multiple files, we construct an ordered set for
# each day, recording the unique indices seen in order.
#
# The "categorical_values" function performs this operation, constructing a set for each
# category.
#
# The "reindex" function then constructs a dictionary mapping original values to
# consecutive integers. This is a serial procedure as we must begin at the first dataset
# shard and then add each successive shard.
#
# - Step 3: With the translation dictionaries setup, we can then reindex the categorical
# values for the dataset

# Utility helper functions
"""
    emptyparse(T, str, fallback = zero(T); kw...)

Try to parse a value of type `T` from `str`.
If `str` is empty, return `fallback` instead.
If `str` is not-empty, all keywords are passed to `Base.parse`.
"""
emptyparse(::Type{T}, str; kw...) where {T} = isempty(str) ? zero(T) : parse(T, str; kw...)

"""
    logtransform([T,] x)

Perform a logtransform on `x`, converting it to a float of type `T`.
`T` defaults to `Float32`.
"""
logtransform(::Type{T}, x) where {T} = log(max(convert(T, x), zero(T)) + one(T))
logtransform(x) = logtransform(Float32, x)

function gunzip_open(f::F, src; kw...) where {F}
    cleanup = false
    if endswith(src, ".gz") && !endswith(src, ".tar.gz")
        path, _ = splitext(src)
        if !ispath(path)
            run(pipeline(`gunzip -c $src`, path))
        end
        cleanup = true
    else
        path = src
    end

    local result
    try
        result = open(f, path; kw...)
    finally
        cleanup && rm(path)
    end

    return result
end

#####
##### Preprocessing
#####

struct DAC end
Base.broadcastable(x::DAC) = (x,)

num_labels(::DAC) = 1
num_continuous_features(::DAC) = 13
num_categorical_features(::DAC) = 26

struct DACRecord
    label::Int32
    continuous::NTuple{13,Float32}
    categorical::NTuple{26,UInt32}
end

function Base.write(io::IO, x::DACRecord)
    a = write(io, x.label)
    b = foreach(i -> write(io, i), x.continuous)
    c = foreach(i -> write(io, i), x.categorical)
    # Sum up the total number of bytes read.
    return sizeof(x)
end

function Base.read(io::IO, ::Type{DACRecord})
    label = read(io, Int32)
    continuous = ntuple(i -> read(io, Float32), Val(num_continuous_features(DAC())))
    categorical = ntuple(i -> read(io, UInt32), Val(num_categorical_features(DAC())))
    return DACRecord(label, continuous, categorical)
end

# Because DACRecord is a struct with a bit-compatible layout,
# we can just Mmap the preprocessed file!
function load(::DAC, path; writable = false)
    return open(path; read = true, write = writable) do io
        Mmap.mmap(io, Vector{DACRecord})
    end
end

function create(::DAC, path::AbstractString, len)
    return Mmap.mmap(path, Vector{DACRecord}, len; grow = true)
end
create(::DAC, ::Nothing, len) = Mmap.mmap(Mmap.Anonymous(), Vector{DACRecord}, len, 0)

function reindex(maps::Vector{<:AbstractDict}, record::DACRecord)
    label = record.label
    continuous = record.continuous
    i = 1
    categorical = map(record.categorical) do c
        x = maps[i][c]
        i += 1
        return x
    end
    return DACRecord(label, continuous, categorical)
end

function binarize(src::AbstractString, dst = nothing)
    A = gunzip_open(src; read = true, lock = false) do io
        binarize(io, dst)
    end
    return A
end

function binarize(io::IO, _dst::Union{Nothing,AbstractString})
    print("Counting number of data points: ")
    seekstart(io)
    nlines = countlines(io)
    seekstart(io)
    println(nlines)

    progress = ProgressMeter.Progress(nlines, 1, "Binarizing Dataset ")

    # Memory map destination
    dst = create(DAC(), _dst, nlines)
    i = 1
    while !eof(io)
        dst[i] = parseline(io)
        ProgressMeter.next!(progress)
        i += 1
    end
    return dst
end

function parseline(io::IO)
    # Helper closures
    function load_continuous(i)
        return logtransform(emptyparse(Int32, readuntil(io, '\t'); base = 10))
    end
    function load_categorical(i)
        delim = (i == 26) ? '\n' : '\t'
        return emptyparse(UInt32, readuntil(io, delim); base = 16)
    end

    label = parse(Int32, readuntil(io, '\t'); base = 10)
    continuous = ntuple(load_continuous, Val(num_continuous_features(DAC())))
    categorical = ntuple(load_categorical, Val(num_categorical_features(DAC())))
    return DACRecord(label, continuous, categorical)
end

#####
##### Preprocessing Functions
#####

makemaps(len = num_categorical_features(DAC())) = [Dict{UInt32,UInt32}() for _ = 1:len]
function makesets(len = num_categorical_features(DAC()))
    return [DataStructures.OrderedSet{UInt32}() for _ = 1:len]
end

"""
    categorical_values(path::AbstractString, [sets]) -> Vector{OrderedSet}

Return a set for each category in the binary dataset stored at `path`.
"""
function categorical_values(path::AbstractString, sets = makesets(); save = true)
    sets = categorical_values(load(DAC(), path; writable = true), sets)
    if save
        save_path = join((first(splitext(path)), "values.jls"), '_')
        Serialization.serialize(save_path, sets)
    end
    return sets
end

function categorical_values(
    data::Vector{DACRecord}, sets::Vector{<:AbstractSet} = makesets()
)
    pmeter = ProgressMeter.Progress(length(data), 1, "Indexing Categorical Features ")

    # Parse out records, remap the categorical features.
    for i in eachindex(data)
        @inbounds record = data[i]
        for (_set, _val) in zip(sets, record.categorical)
            push!(_set, _val)
        end

        # Update progress meter.
        ProgressMeter.next!(pmeter)
    end
    return sets
end

function reindex(sets::AbstractVector{<:AbstractSet{T}}) where {T}
    dicts = makemaps(length(sets))
    for i in eachindex(dicts, sets)
        sizehint!(dicts[i], length(sets[i]))
    end
    return reindex!(dicts, sets)
end

function reindex(setsvector::AbstractVector{<:AbstractVector{<:AbstractSet}})
    nsets = length(setsvector)
    println("Indexing set 1 of $nsets")
    @time dicts = reindex(first(setsvector))
    for (i, sets) in enumerate(Iterators.drop(setsvector, 1))
        println("Indexing set $(i+1) of $nsets")
        @time reindex!(dicts, sets)
    end
    return dicts
end

function reindex!(
    dicts::AbstractVector{<:AbstractDict}, sets::AbstractVector{<:AbstractSet}
)
    Threads.@threads for i in eachindex(dicts, sets)
        reindex!(dicts[i], sets[i])
    end
    return dicts
end

function reindex!(dict::Dict{T,UInt32}, set::AbstractSet{T}) where {T}
    for v in set
        get!(dict, v, length(dict) + 1)
    end
    return dict
end

function reindex!(data::Vector{DACRecord}, by::AbstractVector{<:AbstractDict})
    ProgressMeter.@showprogress 1 for i in eachindex(data)
        @inbounds record = data[i]
        new_categorical = ntuple(
            j -> by[j][record.categorical[j]], Val(num_categorical_features(DAC()))
        )
        @inbounds data[i] = DACRecord(record.label, record.continuous, new_categorical)
    end
end

#####
##### Full pipeline
#####

_binpath(path::AbstractString) = join((first(splitext(path)), "bin"), '.')
function process(path::AbstractString, binpath = _binpath(path))
    binpath !== nothing && ispath(binpath) && rm(binpath)

    data = binarize(path, binpath)
    dicts = reindex(categorical_values(data))
    reindex!(data, dicts)
    return data
end

#####
##### Marshaling
#####

function load!(
    labels, dense, sparse, vx::AbstractVector{DACRecord}; nthreads = Threads.nthreads()
)
    Polyester.@batch per=core for i in eachindex(vx)
    #for i in eachindex(labels, vx)
        # @inbounds record = vx[i]
        # @inbounds labels[i] = record.label
        record = vx[i]
        labels[i] = record.label

        # Explicit loops faster the broadcasting.
        # Due to better constant propagaion maybe?
        continuous = record.continuous
        for j in Base.OneTo(num_continuous_features(DAC()))
            @inbounds(dense[j, i] = continuous[j])
        end

        # Locality for loading categorical features is not the best.
        # However, threading should be coarse grained enough to hopefully avoid false
        # sharing.
        categorical = record.categorical
        for j in Base.OneTo(num_categorical_features(DAC()))
            @inbounds(sparse[i, j] = categorical[j])
        end
    end
    return nothing
end

struct DACLoader{L,D,S}
    labels::L
    dense::D
    sparse::S
    dataset::Vector{DACRecord}
    batchsize::Int
end

function DACLoader(dataset, batchsize::Integer; allocator = default_allocator) where {T}
    labels = allocator(Float32, batchsize)
    dense = allocator(Float32, num_continuous_features(DAC()), batchsize)
    # Put batch along the continuous dimension for better lookup locality.
    sparse = allocator(UInt32, batchsize, num_categorical_features(DAC()))
    return DACLoader(labels, dense, sparse, dataset, batchsize)
end

function Base.iterate(loader::DACLoader, i = 1)
    @unpack labels, dense, sparse, dataset, batchsize = loader

    i * batchsize > length(dataset) && return nothing
    start = batchsize * (i - 1) + 1
    stop = batchsize * i

    # See increasing performance until about 8 threads.
    # TODO: Come up with a heuristic for choosing the number of threads based on batchsize.
    nthreads = min(Threads.nthreads(), 8)
    load!(labels, dense, sparse, view(dataset, start:stop); nthreads = nthreads)
    return (; labels, dense, sparse), i + 1
end

Base.length(loader::DACLoader) = div(length(loader.dataset), loader.batchsize)
Base.parent(loader::DACLoader) = loader.dataset
batchsize(loader::DACLoader) = loader.batchsize

#####
##### Model Builder
#####

const KAGGLE_EMBEDDING_SIZES = [
    1460,
    583,
    10131227,
    2202608,
    305,
    24,
    12517,
    633,
    3,
    93145,
    5683,
    8351593,
    3194,
    27,
    14992,
    5461306,
    10,
    5652,
    2173,
    4,
    7046547,
    18,
    15,
    286181,
    105,
    142572,
]

const TERABYTE_EMBEDDING_SIZES = [
    227605432,
    39060,
    17295,
    7424,
    20265,
    3,
    7122,
    1543,
    63,
    130229467,
    3067956,
    405282,
    10,
    2209,
    11938,
    155,
    4,
    976,
    14,
    292775614,
    40790948,
    187188510,
    590152,
    12973,
    108,
    36,
]

function kaggle_dlrm(
    allocator = default_allocator;
    weight_eltype = Float32,
    embedding_eltype = Float32,
)
    # Honestly, there's no reason to specialize here.
    @nospecialize

    v = default_allocator(Float32, 1, 1)
    dot = _Model.DotInteraction(v)
    sparse_feature_size = 128
    # sparse_feature_size = 16
    return dlrm(
        [13, 512, 256, sparse_feature_size],
        [1024, 1024, 512, 256, 1],
        # [13, 512, 256, 64, sparse_feature_size],
        # [512, 256, 1],
        sparse_feature_size,
        KAGGLE_EMBEDDING_SIZES;
        interaction = dot,
        weight_eltype = weight_eltype,
        embedding_constructor = x -> SimpleEmbedding{Static{sparse_feature_size}}(x),
        embedding_eltype = embedding_eltype,
        allocator = allocator,
    )
end

#####
##### PyTorch HDF5 loader
#####

# Allocate arrays so the base pointer is aligned to the start of a cache line.
function aligned_allocator(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    ptr_ref = Ref(Ptr{Nothing}())
    bytes = sizeof(T) * prod(dims)
    alignment = 64

    ret = ccall(
        :posix_memalign,
        Cint,
        (Ptr{Ptr{Cvoid}}, Csize_t, Csize_t),
        ptr_ref,
        alignment,
        bytes,
    )

    A = unsafe_wrap(Array, Ptr{T}(ptr_ref[]), dims; own = false)
    finalizer(A) do _A
        Libc.free(pointer(_A))
    end
    return A
end

# Load models from an HDF5 File.
# TODO: Provide options for CachedArrays etc.
function load_hdf5(path::AbstractString, allocator = aligned_allocator; kw...)
    return HDF5.h5open(path) do file
        load_hdf5(file, allocator; kw...)
    end
end

function load_hdf5(
    file::HDF5.File,
    allocator = aligned_allocator;
    weight_modifier = identity,
    embedding_modifier = identity
)
    # Load Embeddings
    embeddings = load_embeddings(file, allocator; modifier = embedding_modifier)
    bottom_mlp = load_mlp(file, "bot_", allocator; modifier = weight_modifier)
    top_mlp = load_mlp(file, "top_", allocator; modifier = weight_modifier)
    dot = _Model.DotInteraction(Matrix{Float32}(undef, 1, 1))
    return DLRMModel(bottom_mlp, embeddings, dot, top_mlp)
end

function load_embeddings(file::HDF5.File, allocator = aligned_allocator; modifier = identity)
    names = sort(filter(startswith("emb"), keys(file)); lt = NaturalSort.natural)
    return map(names) do name
        _data = modifier(read(file, name))
        data = allocator(eltype(_data), size(_data)...)
        data .= _data
        return SimpleEmbedding{Static{size(data,1)}}(data)
    end
end

function load_mlp(
    file::HDF5.File,
    prefix_filter::AbstractString,
    allocator = aligned_allocator;
    modifier = identity,
)
    names = sort(filter(startswith(prefix_filter), keys(file)); lt = NaturalSort.natural)
    # Names for layers are structured like this:
    # "bot_l.0.bias"
    # "bot_l.0.weight"
    # "bot_l.2.bias"
    # "bot_l.2.weight"
    # ...
    #
    # The strategy here is to grab the prefix before the period, find the unique prefixes,
    # the go through them in order to build the layers.
    prefixes = unique(first.(splitext.(names)))
    layers = []
    for prefix in prefixes
        _weight = modifier(read(file["$prefix.weight"]))
        _bias = modifier(read(file["$prefix.bias"]))

        weight = allocator(eltype(_weight), size(_weight)...)
        weight .= _weight
        bias = allocator(eltype(_bias), size(_bias)...)
        bias .= _bias

        # Choose whether to use Relu or Sigmoid activation function.
        isrelu = (prefix != prefixes[end] || prefix_filter == "bot_")
        if isrelu
            if prefix != prefixes[end]
                layer = OneDNN.Dense(weight, bias, Flux.relu)
            else
                layer = OneDNN.Dense(weight, bias, Flux.relu, Float32)
            end
        else
            layer = OneDNN.Dense(weight, bias, identity, Float32)
        end
        push!(layers, layer)
        if !isrelu
            push!(layers, x -> OneDNN.eltwise(Flux.sigmoid, x))
        end
    end
    return Flux.Chain(layers...)
end

function load_inputs(file::HDF5.File)
    labels = vec(read(file["labels"]))
    dense = read(file["input_bot"])

    # The sparse inputs are stored as individual vectors.
    # Concatenate them together into a single 2D matrix.
    prefixes = sort(filter(startswith("input_emb"), keys(file)); lt = NaturalSort.natural)
    sparse = read.(getindex.(Ref(file), prefixes))
    for _a in sparse
        _a .+= 1
    end

    batchsize = length(labels)

    # Maybe reshape if there are multiple lookups.
    sparse = map(sparse) do _vec
        if length(_vec) > batchsize
            return reshape(_vec, :, batchsize)
        else
            return _vec
        end
    end

    return (; labels, dense, sparse)
end
