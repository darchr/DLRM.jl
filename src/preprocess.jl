#####
##### Utility Macros
#####

# Hack to defeating `ntuple` not constant propogating.
macro doN(N,expr)
    expr = esc(expr)
    vars = [Symbol("a$i") for i in 1:N]
    exprs = [:($(vars[i]) = $expr) for i in 1:N]
    return quote
        $(exprs...)
        ($(vars...),)
    end
end

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

function maybegunzip(src)
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

    return path, () -> cleanup ? rm(path) : nothing
end


#####
##### DAC
#####

struct DAC end

num_labels(::DAC) = 1
num_continuous_features(::DAC) = 13
num_categorical_features(::DAC) = 26

struct DACRecord
    label::Int32
    continuous::NTuple{13,Float32}
    categorical::NTuple{26,UInt32}
end

function Base.read(io::IO, ::Type{DACRecord})
    label = read(io, Int32)
    continuous = @doN 13 read(io, Float32)
    categorical = @doN 26 read(io, UInt32)
    return DACRecord(label, continuous, categorical)
end

function Base.write(io::IO, x::DACRecord)
    a = write(io, x.label)
    b = write.((io,), x.continuous)
    c = write.((io,), x.categorical)
    # Sum up the total number of bytes read.
    return sum(sum, (a, b, c))
end

# Because DACRecord is a struct with a bit-compatible layout,
# we can just Mmap the preprocessed file!
function load(::DAC, path; writable = false)
    io = open(path; read = true, write = writable)
    return Mmap.mmap(io, Vector{DACRecord})
end

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

function binarize(maps::Vector{<:AbstractDict}, src, dst)
    # Check to see if the extension is `.gz`
    # If it is, decompress it.
    src, finalizer = maybegunzip(src)

    nlines = countlines(src)
    progress = Progress(nlines, 1, "Binarizing Dataset")

    # Read a record from the original file, write it in binary form to the
    # destination.
    open(src; read = true) do srcio
        open(dst; write = true) do dstio
            while !eof(srcio)
                write(dstio, reindex(maps, parseline(srcio)))
                next!(progress)
            end
        end
    end

    # Do any cleanup.
    finalizer()

    return nothing
end

function parseline(io::IO)
    label = parse(Int32, readuntil(io, '\t'); base = 10)
    continuous = load_continuous(io)
    categorical = load_categorical(io)

    # Grab the last categorical value
    last = emptyparse(UInt32, readuntil(io, '\n'); base = 16)
    return DACRecord(label, logtransform.(continuous), (categorical..., last))
end

# Wrap these in independent functions to avoid exploding the IR of `put_in_memory`.
@noinline load_continuous(io::IO) = @doN 13 emptyparse(Int32, readuntil(io, '\t'); base = 10)
@noinline load_categorical(io::IO) = @doN 25 emptyparse(UInt32, readuntil(io, '\t'); base = 16)

#####
##### Preprocessing Functions
#####

makemaps() = [Dict{UInt32,UInt32}() for _ in 1:num_categorical_features(DAC())]
makesets() = [DataStructures.OrderedSet{UInt32}() for _ in 1:num_categorical_features(DAC())]

function categorical_values(path::AbstractString, x...)
    path, cleanup = maybegunzip(path)
    maps = open(path) do io
        categorical_values(io, x...)
    end
    cleanup()
    return maps
end

function categorical_values(io::IO, sets = makesets())
    categorical_values!(sets, io)
    return sets
end

function categorical_values!(sets::Vector{<:AbstractSet}, io::IO)
    # Coune the number of lines in the file
    nlines = countlines(io)
    seekstart(io)
    pmeter = Progress(nlines, 1, "Indexing Categorical Features")

    # Parse out records, remap the categorical features.
    while !eof(io)
        record = parseline(io)

        # The idea here is:
        #
        # If we've seen an index for a given category, there is no update to the dictionary.
        # If we haven't seen an entry, then we assign it the next available number.
        push!.(sets, record.categorical)

        # Update progress meter.
        next!(pmeter)
    end
end

function reindex_merge!(a::AbstractDict, b::AbstractSet)
    for x in b
        get!(a, x, length(a) + 1)
    end
    return a
end

_reindex(X::AbstractSet) = Dict(x => UInt32(i) for (i, x) in enumerate(X))
function reindex(path::AbstractString)
    seen = categorical_values(path)
    return _reindex.(seen)
end

function reindex(paths::Vector{<:AbstractString}; maps = makemaps())
    seen = pmap(categorical_values, paths)

    # Merge dictionaries, giving priority to the first entry.
    aggregate = reduce((x,y) -> reindex_merge!.(x, y), seen; init = maps)
    return aggregate
end

