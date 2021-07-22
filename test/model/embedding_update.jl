@testset "Testing Embedding Update" begin
    # Create embedding tables.
    numcols = 100
    featuresize = 128
    numlookups = 100
    batchsize = 128
    update_batchsize = 16

    tables = map(1:20) do _
        return DLRM.SimpleEmbedding{DLRM.Static{featuresize}}(
            randn(Float32, featuresize, numcols)
        )
    end

    # Keep a second set of tables for error checking.
    tables_reference = [deepcopy(table.data) for table in tables]

    indices = map(tables) do _
        rand(1:numcols, numlookups, batchsize)
    end
    indices_reference = deepcopy(indices)

    out, back = Zygote._pullback(
        DLRM.maplookup,
        DLRM.SimpleParallelStrategy(),
        tables,
        indices,
    )

    # Now, generate a reference result.
    f = (_tables, _indices) -> mapreduce(DLRM.lookup, vcat, _tables, _indices)
    out_reference, back_reference = Zygote._pullback(f, tables_reference, indices_reference)
    @test isapprox(reduce(vcat, out), out_reference)

    # Just feed back the results to get the correct updates.
    updates = back(out)
    @test updates[1] === nothing
    @test updates[2] === nothing
    @test updates[4] === nothing
    @test isa(updates[3], Vector{<:DLRM.SparseEmbeddingUpdate})

    feeder = map(updates[3]) do update
        return DLRM.UpdatePartitioner(update, update_batchsize)
    end

    # Use the updater to process and apply the updates.
    opt = Flux.Descent(10.0)
    updater = DLRM._Model.BatchUpdater()
    DLRM._Model.process!(updater, opt, tables, feeder, 4)

    #####
    ##### Now, try the same thing with the reference implementation.
    #####

    update_reference = back_reference(out_reference)
    @test update_reference[1] === nothing
    @test update_reference[2] !== nothing
    @test isa(update_reference[3], Vector{Nothing})
    Flux.update!(opt, tables_reference, update_reference[2])

    for i in eachindex(tables, tables_reference)
        @test isapprox(tables[i], tables_reference[i])
    end
end
