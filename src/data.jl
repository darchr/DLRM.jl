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
logtransform(::Type{T}, x::U) where {T,U} = log(max(convert(T, U), zero(T)) + one(T))
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

# Because DACRecord is a struct with a bit-compatible layout,
# we can just Mmap the preprocessed file!
load(::DAC, path) = Mmap.mmap(path, Vector{DACRecord})

"""
    preprocess(::DAC, src, dst)

Preprocess the training data at `src`. Place the preprocessed file at `dst`.
"""
preprocess(dac::DAC, src, dst) = preprocess(dac, put_in_memory(dac, src), dst)

function preprocess(dac::DAC, records::Vector{DACRecord}, dst)
    categorical_maps = categorize(dac, records)
    # Now we dump everything straight into a binary file.
    open(dst; write = true) do f
        @showprogress 1 "Writing Records to Disk" for record in records
            write(f, record.label)
            # Transform and write the continuous variables.
            write.((f,), record.continuous)

            # Write the transformed hashes
            write.((f,), getindex.(categorical_maps, record.categorical))
        end
    end
    return nothing
end

# Load the dataset into memory (we have lots of DRAM, so we can do this.
function put_in_memory(dac::DAC, src)
    lines = DACRecord[]
    nlines = countlines(src)
    progress = Progress(nlines, 1, "Reading Dataset into Memory")
    open(src; read = true) do f
        while !eof(f)
            # Read the label
            label = parse(Int32, readuntil(f, '\t'); base = 10)
            continuous = @doN 13 emptyparse(Int32, readuntil(f, '\t'); base = 10)
            categorical = @doN 25 emptyparse(UInt32, readuntil(f, '\t'); base = 16)

            # Grab the last categorical value
            last = emptyparse(UInt32, readuntil(f, '\n'); base = 16)
            categorical = (categorical..., last)
            push!(lines, DACRecord(label, logtransform.(continuous), categorical))

            # Update the progress meter
            next!(progress)
        end
    end
    return lines
end

function categorize(dac::DAC, records::Vector{DACRecord})
    # Construct sets for each category.
    category_values = [Set{UInt32}() for _ in 1:num_categorical_features(dac)]
    @showprogress 1 "Converting Categorical Hashes" for record in records
        push!.(category_values, record.categorical)
    end

    # Construct maps converting hashes into sequential indices
    return map(category_values) do values
        return Dict(k => UInt32(i) for (i,k) in enumerate(sort(collect(values))))
    end
end

# Validate that records are parsed correctly.
function validate(dac::DAC, src, dst)
    nlines = countlines(src)
    progress = Progress(nlines, 1, "Validating Dataset Conversion")

    # Keep track of the integers we've assigned to the hashes.
    category_maps = [Dict{UInt32,UInt32}() for _ in 1:num_categorical_features(dac)]
    records = load(dac, dst)
    open(src; read = true) do src_io
        for (i, ln) in enumerate(eachline(src_io))
            # Read a line from source
            src_line = split(ln, '\t')
            record = records[i]

            # Check Label
            @assert parse(Int32, first(src_line); base = 10) == record.label

            # Check continuous variables
            continuous = view(src_line, 2:(1 + num_continuous_features(dac)))
            fields = logtransform.(emptyparse.(Int32, continuous))
            for (v,f) in zip(record.continuous, fields)
                @assert v == f
            end

            # Check categorical variables
            categorical = @views(src_line[(end - num_categorical_features(dac))+1:end])
            fields = emptyparse.(UInt32, categorical; base = 16)
            for (m, f, v) in zip(category_maps, fields, record.categorical)
                expected = get!(m, f, v)
                @assert expected == v
            end

            # Update progress meter
            next!(progress)
        end
    end
    return nothing
end

