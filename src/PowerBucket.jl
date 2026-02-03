using Base.Math: log2, floor
using Printf

struct PowerBucket
  weights::Vector{Float64}
  large_buckets::Vector{Set{Int}}
  small_buckets::Vector{Set{Int}}
  index_to_level::Vector{Int}
end

function PowerBucket(initial_weights::Vector{Float64})
  n = length(initial_weights)
  if n == 0
    return PowerBucket(Float64[], Set{Int}[], Set{Int}[], Int[])
  end
  index_to_level = zeros(Int, n)
  max_large = 0
  max_small = 0
  for i in 1:n
    w = initial_weights[i]
    @assert w > 0 "Weights must be positive"
    level = floor(Int, log2(w)) + 1
    index_to_level[i] = level
    if level >= 1
      max_large = max(max_large, level)
    else
      small_level = 1 - level
      max_small = max(max_small, small_level)
    end
  end
  large_buckets = [Set{Int}() for _ in 1:max_large]
  small_buckets = [Set{Int}() for _ in 1:max_small]
  for i in 1:n
    level = index_to_level[i]
    if level >= 1
      push!(large_buckets[level], i)
    else
      small_level = 1 - level
      push!(small_buckets[small_level], i)
    end
  end
  PowerBucket(copy(initial_weights), large_buckets, small_buckets, index_to_level)
end

function update!(pb::PowerBucket, index::Int, new_weight::Float64)
  @assert 1 <= index <= length(pb.weights) "Invalid index"
  @assert new_weight > 0 "Weights must be positive"
  old_level = pb.index_to_level[index]
  if old_level >= 1
    delete!(pb.large_buckets[old_level], index)
  else
    old_small_level = 1 - old_level
    delete!(pb.small_buckets[old_small_level], index)
  end
  pb.weights[index] = new_weight
  new_level = floor(Int, log2(new_weight)) + 1
  pb.index_to_level[index] = new_level
  if new_level >= 1
    if new_level > length(pb.large_buckets)
      for _ in length(pb.large_buckets)+1:new_level
        push!(pb.large_buckets, Set{Int}())
      end
    end
    push!(pb.large_buckets[new_level], index)
  else
    new_small_level = 1 - new_level
    if new_small_level > length(pb.small_buckets)
      for _ in length(pb.small_buckets)+1:new_small_level
        push!(pb.small_buckets, Set{Int}())
      end
    end
    push!(pb.small_buckets[new_small_level], index)
  end
end

function get_elements_in_level(pb::PowerBucket, level::Int)::Vector{Int}
  if level >= 1
    if level > length(pb.large_buckets)
      return Int[]
    end
    return sort(collect(pb.large_buckets[level]))
  else
    small_level = 1 - level
    if small_level > length(pb.small_buckets) || small_level < 1
      return Int[]
    end
    return sort(collect(pb.small_buckets[small_level]))
  end
end

function get_level(pb::PowerBucket, i::Int)
  return pb.index_to_level[i]
end

function bucket_info(pb::PowerBucket, violations::Vector{Int} = Int[], samples::Vector{Int} = Int[])::String
    buf = IOBuffer()
    total_elements = length(pb.weights)
    if total_elements == 0
        return "No elements in the PowerBucket."
    end
    levels = Int[]
    for small_i in 1:length(pb.small_buckets)
        push!(levels, 1 - small_i)
    end
    for large_i in 1:length(pb.large_buckets)
        push!(levels, large_i)
    end
    sort!(levels)
    viol_set = Set(violations)
    sample_set = Set(samples)
    total_sample = length(samples)
    total_violations = length(viol_set)
    @printf(buf, "%-6s | %-9s | %-10s | %-10s | %-11s| %-10s | %-11s\n", "Level", "Elements", "Percentage", "Violations", "Viol. Perc.", "Samples", "Sample. Perc.")
    println(buf, "-------|-----------|------------|------------|------------|------------|------------")
    for level in levels
        if level >= 1
            elements_set = pb.large_buckets[level]
            elements_in_level = length(elements_set)
        else
            small_level = 1 - level
            elements_set = pb.small_buckets[small_level]
            elements_in_level = length(elements_set)
        end
        percentage = (elements_in_level / total_elements) * 100
        violations_in_level = length(intersect(viol_set, elements_set))
        viol_perc = total_violations > 0 ? (violations_in_level / total_violations) * 100 : 0.0
        samples_in_level = length(intersect(sample_set, elements_set))
        sample_perc = total_sample > 0 ? (samples_in_level / total_sample) * 100 : 0.0
        @printf(buf, "%-6d | %-9d | %-9.4f%% | %-10d | %-9.4f%% | %-10d | %-9.4f%% \n", level-1, elements_in_level, percentage, violations_in_level, viol_perc, samples_in_level, sample_perc)
    end
    return String(take!(buf))
end
