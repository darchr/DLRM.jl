# The update procedure for embedding tables is tricky!
#
# For one, we don't really want to assign each thread a table because the load balancing
# ends up being pretty bad.
#
# Furthermore, if all threads try to write their update to embedding tables in persistent
# memory at the same time, we end up with pretty terrible bandwidth.
#
# The goal of this bit of code is to get around these problems using the following
# architecture:
#
#                                Writeback Queue
#      ***                        **************
#      ***                           A    |
#      *** Embedding tables          |    |
#      *** and full updates          |    |
#      *** partitioned into          |    |
#      *** smaller chunks            |    +-----> Write back Threads
#      *** using the partitioner     |
#      ***                           |
#      ***                           |
#       |                            |
#       +---> Precompute Threads ----+
#
# The precompute threads perform operations like aggregating the embedding table updates
# while memory is in DRAM.
#
# Once a sufficient number have been aggregated, the bundled update gets sent to the
# writeback queue where a number of worker threads are responsible for trickling the
# data back to the original embedding tables.
#
# NOW, we have the question of if writebacks need to lock their respective embedding
# tables.
#
# The answer (hopefully) is "no" because floating point addition is mostly associative.
# Thus, we can let the CPU's cache coherence protocol work things out in case of
# collisions.
mutable struct BatchUpdater
    # Keep these fields pretty unspecialized since they aren't exactly performance
    # critical at this level.
    writeback_queue::DataStructures.CircularBuffer{Any}
    writeback_queue_lock::ReentrantLock

    # Control elements for dividing up work.
    current_index::Int
    done::Bool
    table_update_queue_lock::ReentrantLock
end

function BatchUpdater(queue_length = 500)
    return BatchUpdater(
        # Writeback Queue
        DataStructures.CircularBuffer{Any}(queue_length),
        ReentrantLock(),
        # Control for process queue,
        1,
        false,
        ReentrantLock(),
    )
end

function process!(updater::BatchUpdater, opt, tables, updates, writeback_threads)
    # Reset state of the updater.
    updater.current_index = 1
    updater.done = false

    ref = ManualMemory.Reference(updater)
    Polyester.@batch per=core for i in Base.OneTo(Threads.nthreads())
        _updater = ManualMemory.dereference(ref)

        # Dispatch threads based on what task they are to perform.
        if i <= writeback_threads
            writeback_task(_updater, opt, tables, updates)
        else
            process_task(_updater, opt, tables, updates)
        end
    end
end

#####
##### Processing
#####

function grabwork(updater::BatchUpdater, tables, updates)
    # Fast path, if work is completed, simply return.
    @unpack done = updater
    done && return nothing

    @unpack current_index, table_update_queue_lock = updater
    update = Base.@lock table_update_queue_lock begin
        # Check to see if there's any work available in the current table.
        current_iter = updates[current_index]
        sparse_update = iterate(current_iter)

        # Update exhaused.
        if sparse_update === nothing
            # If we've exhausted the last table, then we are done.
            if current_index == length(tables)
                updater.current_index = current_index + 1
                updater.done = true
                return nothing
            end

            # We haven't exhausted everything.
            # Increment the current index and try again.
            sparse_update = iterate(updates[current_index + 1])
            sparse_update === nothing && error("Expected this to work!!")

            updater.current_index = current_index + 1
        end
        sparse_update
    end

    table = tables[updater.current_index]
    return (; table, update)
end

function _process_task(updater::BatchUpdater, opt, tables, updates)
    work = grabwork(updater, tables, updates)
    work === nothing && return false
    @unpack table, update = work

    processed_update = Flux.Optimise.apply!(opt, table, update)

    # Add the results to the writeback queue.
    @unpack writeback_queue, writeback_queue_lock = updater
    Base.@lock writeback_queue_lock begin
        push!(writeback_queue, (table, processed_update...))
    end
    return true
end

function process_task(updater::BatchUpdater, args...)
    # Keep processing tasks until there is no work left to do.
    success = true
    while success
        success = _process_task(updater, args...)
    end
end


#####
##### Writeback
#####

function _writeback_task(
    table::AbstractEmbeddingTable,
    update::SparseEmbeddingUpdate,
    args...;
    free_update::Bool = false
)
    val = Flux.Optimise.update!(table, update, args...)
    free_update && CachedArrays.unsafe_free(update)
    return val
end

function writeback_task(updater::BatchUpdater, args...)
    # Grab the lock - see if there's work to be done.
    # If there isn't, then grab a table and update to work on so we don't hog the lock.
    @unpack writeback_queue, writeback_queue_lock = updater
    while true
        work = nothing
        Base.@lock writeback_queue_lock begin
            if !isempty(writeback_queue)
                work = popfirst!(writeback_queue)
            end
        end

        # If the queue was empty, then variable "work" will never be updated and
        # will still be "nothing".
        #
        # If this is the case, we then need to go process some more data which will
        # then populate the queue.
        if work === nothing
            # Check if we're done and exit if so.
            updater.done && return nothing

            success = _process_task(updater, args...)
            # If "success" is false, then the last bits of data are being processed.
            # Wait a little bit to give other threads a chance to finish.
            if success == false
                sleep(0.001)
            end
        else
            success = _writeback_task(work...)
        end
    end
end

