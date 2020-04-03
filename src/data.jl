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
    a = write(f, x.label)
    b = write.((f,), record.continuous)
    c = write.((f,), record.categorical)
    # Sum up the total number of bytes read.
    return sum(sum, (a, b, c))
end

# Because DACRecord is a struct with a bit-compatible layout,
# we can just Mmap the preprocessed file!
function load(::DAC, path; writable = false)
    io = open(path; read = true, write = writable)
    return Mmap.mmap(io, Vector{DACRecord})
end

function binarize(src, dst; cleanup = true)
    # Check to see if the extension is `.gz`
    # If it is, decompress it.
    cleanuppath = nothing
    if endswith(src, ".gz") && !endswith(src, ".tar.gz")
        cleanuppath, _ = splitext(src)
        run(pipeline(`gunzip -c $src`, cleanuppath))
    end

    nlines = countlines(src)
    progress = Progress(nlines, 1, "Reading Dataset into Memory")

    # Read a record from the original file, write it in binary form to the
    # destination.
    open(src; read = true) do srcio
        open(dst; write = true) do dstio
            write(dstio, parseline(srcio))
            next!(progress)
        end
    end

    # If we have to clean up, do so
    if cleanup && !isnothing(cleanuppath)
        rm(cleanuppath)
    end

    return nothing
end

function parseline(io::IO)
    label = parse(Int32, readuntil(io, '\t'); base = 10)
    continuous = load_continuous(io)
    categorical = load_categorical(io)

    # Grab the last categorical value
    last = emptyparse(UInt32, readuntil(f, '\n'); base = 10)
    return DACRecord(label, logtransform.(continuous), (categorical..., last))
end

# Wrap these in independent functions to avoid exploding the IR of `put_in_memory`.
@noinline load_continuous(io::IO) = @doN 13 emptyparse(Int32, readuntil(io, '\t'); base = 10)
@noinline load_categorical(io::IO) = @doN 25 emptyparse(UInt32, readuntil(io, '\t'); base = 16)

#####
##### Preprocess Terrabyte Dataset
#####

function binarize_terrabyte(dir, days)
    f = day -> binarize(
        joinpath(dir, "day_$(day).gz"),
        joinpath(dir, "preprocessed", "day_$(day).gz");
        cleanup = true
    )
    pmap(f, days)
end

path_iter(x::String) = (x,)
path_iter(x) = x

function reindex!(paths)
    # Get all of the
    values = reindex(categorical_values(paths))
end

reindex(X::Set) = Dict(x => i for (i, X) in enumerate(X))

categorical_values(path::AbstractString) = categorical_values(load(DAC(), path))
function categorical_values(records::Vector{DACRecord})
    sets = [Set{UInt32}() for _ in 1:num_categorical_features(DAC())]
    categorical_values!(sets, records)
    return sets
end

function categorical_values!(sets, records::Vector{DACRecord})
    @showprogress 1 for record in records
        push!.(sets, record.categorical)
    end
end

function categorical_values(path::Vector{<:AbstractString})
    seen = pmap(categorical_values, paths)
    aggregate = reduce(x -> union.(x), seen)
    return aggregate
end

# # Validate that records are parsed correctly.
# function validate(dac::DAC, src, dst)
#     nlines = countlines(src)
#     progress = Progress(nlines, 1, "Validating Dataset Conversion")
#
#     # Keep track of the integers we've assigned to the hashes.
#     category_maps = [Dict{UInt32,UInt32}() for _ in 1:num_categorical_features(dac)]
#     records = load(dac, dst)
#     open(src; read = true) do src_io
#         for (i, ln) in enumerate(eachline(src_io))
#             # Read a line from source
#             src_line = split(ln, '\t')
#             record = records[i]
#
#             # Check Label
#             @assert parse(Int32, first(src_line); base = 10) == record.label
#
#             # Check continuous variables
#             continuous = view(src_line, 2:(1 + num_continuous_features(dac)))
#             fields = logtransform.(emptyparse.(Int32, continuous))
#             for (v,f) in zip(record.continuous, fields)
#                 @assert v == f
#             end
#
#             # Check categorical variables
#             categorical = @views(src_line[(end - num_categorical_features(dac))+1:end])
#             fields = emptyparse.(UInt32, categorical; base = 16)
#             for (m, f, v) in zip(category_maps, fields, record.categorical)
#                 expected = get!(m, f, v)
#                 @assert expected == v
#             end
#
#             # Update progress meter
#             next!(progress)
#         end
#     end
#     return nothing
# end

