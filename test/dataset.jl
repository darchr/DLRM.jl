# Perform some tests on the sample dataset.to make sure the preprocessing
# pipeline works correctly.
mutable struct Counter
    x::Int
end

function (C::Counter)()
    C.x += 1
    return C.x - 1
end

function Random.rand(::Type{DLRM.DACRecord})
    label = rand(UInt32(0):UInt32(1))
    continuous = Tuple(rand(UInt32, DLRM.num_continuous_features(DLRM.DAC())))
    categorical = Tuple(rand(UInt32, DLRM.num_categorical_features(DLRM.DAC())))
    return DLRM.DACRecord(label, continuous, categorical)
end

stripext(str::AbstractString) = first(splitext(str))

@testset "Testing Datset Processing" begin
    # Test the `doN` macro
    f = Counter(0)
    x = DLRM.@doN 3 f()
    @test x == (0, 1, 2)
    @test DLRM.emptyparse(UInt32, "") == zero(UInt32)
    @test DLRM.emptyparse(UInt32, "10") == 10
    @test DLRM.emptyparse(UInt32, "10"; base = 16) == 0x10

    @test isa(DLRM.logtransform(UInt32(10)), Float32)
    @test isa(DLRM.logtransform(Float64, UInt32(10)), Float64)

    # Test reading and writing of records.
    record = rand(DLRM.DACRecord)
    io = IOBuffer()
    write(io, record)
    seekstart(io)
    @test read(io, DLRM.DACRecord) == record

    # Preprocessing tests
    #
    # These come from the /dataset directory in this repo.
    # This is the first 250 lines of the smaller DAC dataset.
    dir = joinpath(@__DIR__, "dataset")

    # Remap the hashed ID's for each categorical feature into a set of linear indices.
    maps_monolithic = DLRM.reindex(joinpath(dir, "alldays.txt"))

    # Make sure that if we distribute this across spread-out files, we get the same result
    files = [joinpath(dir, "day_$(i).gz") for i in 0:4]
    maps_sharded = DLRM.reindex(files)

    # Also, make sure that residual uncompressed files are not left around.
    for file in files
        uncompressed, _ = splitext(file)
        @test !ispath(uncompressed)
    end

    # The resulting maps should be the same, regardless of whether the dataset was sharded
    # or one single piece.
    @test maps_monolithic == maps_sharded

    # Now, we test warm starting the reindexing process.
    seed = DLRM.reindex(first(files))
    maps_seeded = DLRM.reindex(files[2:end]; maps = seed)
    @test maps_seeded == maps_monolithic

    #####
    ##### Now, test out the reindexing process.
    #####

    DLRM.binarize(
        maps_monolithic,
        joinpath(dir, "alldays.txt"),
        joinpath(dir, "alldays.bin"),
    )

    db = DLRM.load(DLRM.DAC(), joinpath(dir, "alldays.bin"))

    # Parse out the original file and make sure the two match.
    original = open(joinpath(dir, "alldays.txt")) do f
        records = DLRM.DACRecord[]
        while !eof(f)
            push!(records, DLRM.reindex(maps_monolithic, DLRM.parseline(f)))
        end
        return records
    end

    @test db == original
    rm(joinpath(dir, "alldays.bin"))

    # Test this for the sharded dataset
    destinations = begin
        bases = stripext.(files)
        map(bases) do base
            return "$base.bin"
        end
    end

    DLRM.binarize.(Ref(maps_monolithic), files, destinations)

    start = 1
    for dest in destinations
        db_dest = DLRM.load(DLRM.DAC(), dest)
        @test db_dest == db[start:start + length(db_dest) - 1]
        start = start + length(db_dest)
    end
    rm.(destinations)
end
