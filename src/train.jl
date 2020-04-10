function extract(A, ::Val{S}) where {S}
    # Step 1 - create a mapped array
    mapped = MappedArrays.mappedarray(x -> getproperty(x, S), A)

    # Step 2 - Convert the element type
    reinterpreted = reinterpret(eltype(first(mapped)), mapped)

    # Setp 3 - Reshape to 2D
    return reshape(reinterpreted, length(first(mapped)), :)
end

mutable struct TestRun{F,D}
    f::F
    dataset::D
    count::Int
    every::Int
end

function (T::TestRun)()
    T.count += 1
    !iszero(mod(T.count, T.every)) && return

    # Run the test set.
    total = 0
    correct = 0
    println("Testing")
    @time for (dense, sparse, labels) in T.dataset
        result = round.(Int, T.f(dense, sparse))
        labels = clamp.(labels, 0, 1)

        # Update the total
        total += length(result)

        #println(result .- labels)
        # Update the number correct.
        correct += count(x -> x[1] == x[2], zip(result, labels))
    end
    println("Iteration: $(T.count)")
    println("Accuracy: $(correct / total)")
    println()

    if div(T.count, T.every) == 3
        throw(Flux.Optimise.StopException())
    end
end

# Routines for training DLRM.
function top(;
        debug = false,
        # Percent of the dataset to reserve for testing.
        testfraction = 0.1,
        train_batchsize = 128,
        #test_batchsize = 16384,
        test_batchsize = 128,
        test_frequency = 10000,
        learning_rate = 0.1,
    )

    #####
    ##### Import dataset
    #####
    dataset = load(DAC(), joinpath(dirname(@__DIR__), "train.bin"))

    # Split into training and testing regions.
    num_test_samples = ceil(Int, testfraction * length(dataset))

    trainset = @views dataset[1:end-num_test_samples-1]
    testset = @views dataset[end-num_test_samples:end]

    train_loader = Flux.Data.DataLoader(
        extract(trainset, Val{:continuous}()),
        extract(trainset, Val{:categorical}()),
        extract(trainset, Val{:label}());
        batchsize = train_batchsize,
    )

    test_loader = Flux.Data.DataLoader(
        extract(testset, Val{:continuous}()),
        extract(testset, Val{:categorical}()),
        extract(testset, Val{:label}());
        batchsize = test_batchsize,
    )

    model = kaggle_dlrm()

    loss = (dense, sparse, labels) -> begin
        # Clamp for stability
        forward = model(dense, sparse)
        ls = sum(Flux.binarycrossentropy.(forward, vec(labels))) / length(forward)
        isnan(ls) && throw(error("NaN Loss"))
        return ls
    end

    # The inner training loop
    callbacks = [
        TestRun(model, test_loader, 0, test_frequency)
    ]

    # Warmup
    opt = Flux.Descent(0.01)
    count = 1
    params = Flux.params(model)
    for (dense, sparse, labels) in train_loader
        grads = gradient(params) do
            loss(dense, sparse, labels)
        end
        Flux.Optimise.update!(opt, params, grads)
        count += 1
        count == 1000 && break
    end
    opt = Flux.Descent(learning_rate)

    if !debug
        Flux.train!(loss, Flux.params(model), train_loader, opt; cb = callbacks)
    end

    return model, loss, train_loader, opt
end

const KAGGLE_EMBEDDING_SIZES = [
    1460,   583,    10131227,   2202608,
    305,    24,     12517,      633,
    3,      93145,  5683,       8351593,
    3194,   27,     14992,      5461306,
    10,     5652,   2173,       4,
    7046547,18,     15,         286181,
    105, 142572
]

function kaggle_dlrm()
    return dlrm(
        [13, 512, 256, 64, 16],
        [512, 256, 1],
        16,
        KAGGLE_EMBEDDING_SIZES,
    )
end
