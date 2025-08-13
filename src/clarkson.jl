module CLARKSON
  using DataStructures
  using JuMP, Gurobi

  struct BucketSystem{T}
      buckets::Vector{Set{T}}
  end

  # Constructor to initialize with n levels
  function BucketSystem{T}(num_levels::Int) where T
      return BucketSystem([Set{T}() for _ in 1:num_levels])
  end

  # Function to add an element to a specific level
  function add_to_bucket!(bs::BucketSystem{T}, level::Int, elem::T) where T
      1 <= level <= length(bs.buckets) || error("Invalid level")
      push!(bs.buckets[level], elem)
  end

  # Function to promote elements from one level to the next (or any upper level)
  function promote!(bs::BucketSystem{T}, from_level::Int, to_level::Int, elements::Vector{T}) where T
      1 <= from_level <= length(bs.buckets) || error("Invalid levels")
      if (from_level == to_level)
        return
      end

      for elem in elements
          if elem in bs.buckets[from_level]
              delete!(bs.buckets[from_level], elem)
              if (to_level > length(bs.buckets))
                push!(bs.buckets, Set{T}())
              end
              push!(bs.buckets[to_level], elem)
          end
      end
  end

  mutable struct ModelConstraints
      #affConstraints::Vector{ConstraintRef{Model, MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, <:Union{MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}}}}}
      #affConstraints::Vector{ConstraintRef{Model, MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, Any}}}
      #affConstraints::Vector{Any}
      #numAffConstraints::Int64
      #varBounds::Dict{VariableRef, Tuple{Float64, Float64}}
      #numVarBounds::Int64
      #weights::Vector{Int64}
      #buckets::BucketSystem{Int64}
      #totalWeight::Int64
      
      constraints::Vector{ConstraintRef}
      numConstraints::Int64
      weights::Vector{Int64}
      buckets::BucketSystem{Int64}
      totalWeight::Int64
  end

  function ModelConstraints(model::Model)
      BS = BucketSystem{Int64}(64)
      constraints = all_constraints(model; include_variable_in_set_constraints = true)
      i = 1
      for _ in constraints
          add_to_bucket!(BS, 1, i)
          i += 1
      end
      m = length(constraints)
      return ModelConstraints(constraints, length(constraints), zeros(Int, m), BS, m)

      # Collect affine constraints
      #aff_le = all_constraints(model, AffExpr, MOI.LessThan{Float64})
      #aff_ge = all_constraints(model, AffExpr, MOI.GreaterThan{Float64})
      #aff_eq = all_constraints(model, AffExpr, MOI.EqualTo{Float64})
      #aff_constraints = [aff_le; aff_ge; aff_eq]

      #BS = BucketSystem{Int64}(64)

      ## Collect variable bounds
      #vars = all_variables(model)
      #bounds = Dict{VariableRef, Tuple{Float64, Float64}}()
      #i = 1
      #for v in vars
      #    lb = has_lower_bound(v) ? lower_bound(v) : -Inf
      #    ub = has_upper_bound(v) ? upper_bound(v) : Inf
      #    bounds[v] = (lb, ub)
      #    add_to_bucket!(BS, 1, i)
      #    i +=1
      #end

      #for c in aff_constraints
      #    add_to_bucket!(BS, 1, i)
      #    i += 1
      #end

      #m = length(vars) + length(aff_constraints)
      #return ModelConstraints(aff_constraints, length(aff_constraints), bounds,
      #                        length(bounds), zeros(Int, m), BS, m)
  end

  #
  # getConstraint(Constraints, i)
  #
  # Input:
  #   Constraints, affine and variable constraints.
  #   i, index.
  #
  # Pre: 1 <= i <= Constraints.numVarBounds + Constraints.numAffConstraints
  # 
  # Output: Return the i-th constraint.
  #
  function getConstraint(Constraints::ModelConstraints, i::Int)
    return Constraints.constraints[i]
    #if i <= Constraints.numVarBounds
    #  return Constraints.varBounds[i]
    #else
    #  return Constraints.affConstraints[i-Constraints.numVarBounds]
    #end
  end

  function updateWeight(Constraints::ModelConstraints, i::Int)
    Constraints.totalWeight += 2^Constraints.weights[i]
    promote!(Constraints.buckets, Constraints.weights[i]+1, Constraints.weights[i]+2, [i])
    Constraints.weights[i] += 1
  end

  # addConstraints(model, constraints, variableMap, R)
  #
  # Input:
  #   model, a optimization model
  #   constraints, set of constraints
  #   variableMap, maps VariableRef from constraints to the corresponding
  #                VariableRef in model.
  #   R, a sampled set such that R[i] is a valid index of constraints.
  #
  # Post: 
  #   model is modified such that all constraints represented by R is added in.
  #
  # Output: None
  #
  function addConstraints(model::Model, constraints::ModelConstraints,
      variableMap::Dict{VariableRef, VariableRef}, R::Vector{Int})
    for i in R
      con = getConstraint(constraints, i)
      con_obj = constraint_object(con)
      func = con_obj.func
      set = con_obj.set
      new_func = nothing
      if func isa VariableRef
        new_func = variableMap[func]
      else
        new_func = sum(coef * variableMap[var] for (var, coef) in func.terms) + func.constant
      end
      if isa(set, MOI.LessThan{Float64})
        @constraint(model, new_func <= set.upper)
      elseif isa(set, MOI.GreaterThan{Float64})
        @constraint(model, new_func >= set.lower)
      elseif isa(set, MOI.EqualTo{Float64})
        @constraint(model, new_func == set.value)
      end
    end
  end

  # createBaseModel(model)
  #
  # Input: model
  #
  # Output: newModel, varMap
  #   newModel, is a newModel containing all variables from model.
  #   variableMap, maps VariableRef from constraints to the corresponding
  #                VariableRef in model.
  function createBaseModel(model)
    baseModel = Model()
    vars = all_variables(model)
    # Map VariableRef from Model to VariableRef to baseModel
    varMap = Dict{VariableRef, VariableRef}()
    for v in vars
      #lb = has_lower_bound(v) ? lower_bound(v) : -Inf
      #ub = has_upper_bound(v) ? upper_bound(v) : Inf
      lb = -Inf
      ub = Inf
      varMap[v] = @variable(baseModel, base_name = name(v), lower_bound=lb, upper_bound=ub)
    end


    obj_sense = objective_sense(model)
    obj_func = objective_function(model)
    if obj_func isa VariableRef
      new_obj = varMap[obj_func]
    elseif obj_func isa AffExpr
      new_obj = AffExpr(obj_func.constant)
      for (var, coef) in obj_func.terms
        new_obj += coef * varMap[var]
      end
    #elseif obj_func isa QuadExpr
    #  new_obj = QuadExpr(obj_func.aff)
    #  for (pair, coef) in obj_func.terms
    #    new_obj += coef * varMap[pair.a] * varMap[pair.b]
    #  end
    else
      error("Unsupported objective type: $(typeof(obj_func))")
    end
    set_objective(baseModel, obj_sense, new_obj)
    return baseModel, varMap
  end


  function violatedConstraints(constraints::ModelConstraints, variableMap::Dict{VariableRef, VariableRef}, point::Dict{VariableRef, Float64})
    violated = []
    is_feasible = true

    # Check variable bounds (implicit constraints)
    i = 1
    #for (var, (lower, upper)) in constraints.varBounds
    #  val = point[var]
    #  if val < lower || val > upper 
    #    push!(violated, i)
    #  end 
    #  i += 1
    #end

    for con_ref in constraints.constraints
      con_obj = constraint_object(con_ref)
      func = con_obj.func
      set = con_obj.set

      if func isa VariableRef
        eval_val = point[variableMap[func]]
      else
        eval_val = func.constant
        for (var, coef) in func.terms
          eval_val += coef * point[variableMap[var]]
        end
      end

      violation = false
      if isa(set, MOI.LessThan{Float64})
        if eval_val > set.upper
          violation = true
          is_feasible = false
        end
      elseif isa(set, MOI.GreaterThan{Float64})
        if eval_val < set.lower
          violation = true
          is_feasible = false
        end
      elseif isa(set, MOI.EqualTo{Float64})
        if eval_val != set.value
          violation = true
          is_feasible = false
        end
      end

      if violation
          push!(violated, i)
      end
      i += 1
    end

    return is_feasible, violated
  end


  function sample(model::ModelConstraints)
    sample = rand(1:model.totalWeight)
    accumulator = 0
    i = 0
    for S in model.buckets.buckets
      accumulator += length(S) * 2^i
      if sample <= accumulator
        #return S[rand(1:length(S))]
        return rand(collect(S))
      end
      i += 1
    end
    error("Invalid sample")
  end

  function sample(model::ModelConstraints, r::Int)
    samples = Vector{Int}();
    for i in 1:r
      push!(samples, sample(model))
    end
    return samples
  end

  # clarkson(model)
  #
  # Input: model containing the LP.
  #
  # Output: Return an optimal value and primal solution to the LP.
  #
  function clarkson(model::Model)
    # Initial setup stage:
    constraintTypes = list_of_constraint_types(model)
    constraints = ModelConstraints(model)
    # First insert all the variable lower and upper bounds then insert affine
    # constraints.
    #for F, S in constraintTypes
    #  if F isa VariableRef
    #  append!(constraints, all_constraints(model, F, S))
    #end
    #for F, S in constraintTypes
    #  if !(F isa VariableRef)
    #  append!(constraints, all_constraints(model, F, S))
    #end

    #m = length(constraints.numVarBounds + constraints.numAffConstraints)
    m = length(all_constraints(model; include_variable_in_set_constraints = true))
    n = length(all_variables(model))

    # Get number of constraints and number of variables in MPS file.
    #r = n
    r = 6*n^2
    # NOTE: We use base 2 to represent weights. So initializing all weights to
    # 0 represents each element has weight 2^0 = 1.

    while true
      # Sampling procedure:
      R = sample(constraints, r)
      newModel, variableMap = createBaseModel(model)
      addConstraints(newModel, constraints, variableMap, R)
      set_optimizer(newModel, Gurobi.Optimizer)
      # Solve base case.
      optimize!(newModel)
      status = termination_status(newModel)
      optimalPrimal = nothing
      if status == MOI.OPTIMAL
        # Extract the optimal solution.
        optimalPrimal = Dict(zip(all_variables(newModel), value(all_variables(newModel))))
      else
        println("Base case did not have an optimal solution. Continue to next iteration...")
        continue
      end
      # Check violated constraints
      is_feasible, V = violatedConstraints(constraints, variableMap, optimalPrimal)
      if isempty(V)
        return objective_value(newModel), optimalPrimal
        # Note our totalWeight can always be bounded by m^2, since |H| <=
        # (1+1/(3n))^{n ln(m)} m ~ m^2
      elseif length(V) < (2*n*constraints.totalWeight)/r
        for v in V
          updateWeight(constraints, v)
        end
      end
    end 
  end

  export clarkson
end
