module clarkson
  using DataStructures
  import MathOptInterface as MOI
  using JuMP, Gurobi
  using Random, WeightVectors
  using LinearAlgebra
  using Dualization
  using IterativeSolvers
  using SparseArrays
  using TimerOutputs
  include(joinpath(@__DIR__, "PowerBucket.jl"))

  EPS = 1e-8

  const to = TimerOutput()

  mutable struct ModelConstraints
      constraints::Vector{ScalarConstraint}
      numConstraints::Int64
      weights::WeightVectors.AbstractWeightVector
      totalWeight::Float64
      PB::PowerBucket
      rng::AbstractRNG
      data::LPMatrixData{Float64}
      numAffConstraints::Int64
      numAffEqualTo::Int64
      numAffLessThan::Int64
      numAffGreaterThan::Int64
      numVarConstraints::Int64
      numVarEqualTo::Int64
      numVarLessThan::Int64
      numVarGreaterThan::Int64
      model::Model
      LHS::Vector{AffExpr}
      RHS::Vector{Float64}
      Operators::Vector{Function}
      include_variable::Bool
      rowNorm::Vector{Float64}
      b::Vector{Float64}
  end

  function isVariableConstraint(m::ModelConstraints, i::Int64)
    return (i > m.numAffConstraints) && (i <= m.numConstraints)
  end

  function isAffConstraint(m::ModelConstraints, i::Int64)
    return (i <= m.numAffConstraints) && (i >= 1)
  end

  function ModelConstraints(model::Model, include_variable)
      # Use lp_matrix_data ordering for consistency between constraint indexing and matrix A
      data = lp_matrix_data(model)
      # Get constraint objects in the same order as rows in data.A
      constraints = constraint_object.(data.affine_constraints)
      i = 1
      # Use size of matrix A for constraint count to ensure consistency
      numAffConstraints = size(data.A, 1)
      m = numAffConstraints
      # TODO: Right now we only support =, <=, >=, but not of type # MOI.Interval{float64}.
      numAffEqualTo = length(all_constraints(model, AffExpr, MOI.EqualTo{Float64}))
      numAffLessThan = length(all_constraints(model, AffExpr, MOI.LessThan{Float64}))
      numAffGreaterThan = length(all_constraints(model, AffExpr, MOI.GreaterThan{Float64}))

      numVarEqualTo = length(all_constraints(model, VariableRef, MOI.EqualTo{Float64}))

      numVarLessThan = length(all_constraints(model, VariableRef, MOI.LessThan{Float64}))

      numVarGreaterThan = length(all_constraints(model, VariableRef, MOI.GreaterThan{Float64}))

      numVarConstraints = numVarEqualTo + numVarLessThan + numVarGreaterThan

      function getValue(set)
        if isa(set, MOI.LessThan{Float64})
          return set.upper
        elseif isa(set, MOI.GreaterThan{Float64})
          return set.lower
        elseif isa(set, MOI.EqualTo{Float64})
          return set.value
        else
          # Throw an exception later.
          exit(1)
        end
      end

      function getOperator(set)
        if isa(set, MOI.LessThan{Float64})
          return <=
        elseif isa(set, MOI.GreaterThan{Float64})
          return >=
        elseif isa(set, MOI.EqualTo{Float64})
          return ==
        else
          # Throw an exception later.
          exit(1)
        end
      end

      A = data.A
      b = [ (isinf(data.b_lower[i]) ? data.b_upper[i] : data.b_lower[i]) for i in 1:size(A, 1)]
      LHS = []
      RHS = []
      Operators = []
      rowNorm = vec(sqrt.(sum(abs2, A, dims=2)))
      #LHS = A * data.variables
      #RHS = [ getValue(c.set) for c in constraint_object.(data.affine_constraints)]
      #Operators = [ getOperator(c.set) for c in constraint_object.(data.affine_constraints)]
      #LHS = vcat(LHS, [ c.func for c in constraint_object.(data.variable_constraints)])
      #RHS = vcat(RHS, [ getValue(c.set) for c in constraint_object.(data.variable_constraints)])
      #Operators = vcat(Operators, [ getOperator(c.set) for c in constraint_object.(data.variable_constraints)])

      return ModelConstraints(
        constraints,
        length(constraints),
        FixedSizeWeightVector(ones(Float64, m)),
        m,
        PowerBucket(ones(Float64, m)),
        Xoshiro(42),
        data, 
        numAffConstraints,
        numAffEqualTo,
        numAffLessThan,
        numAffGreaterThan,
        numVarConstraints,
        numVarEqualTo,
        numVarLessThan,
        numVarGreaterThan,
        model,
        LHS,
        RHS,
        Operators,
        include_variable,
        rowNorm,
        b
        )
  end

  #
  # getConstraint(Constraints, i)
  #
  # Input:
  #   Constraints, ModelConstraints.
  #   i, index.
  #
  # Pre: 1 <= i <= Constraints.numVarBounds + Constraints.numAffConstraints
  # 
  # Output: Return the i-th constraint.
  #
  function getConstraint(Constraints::ModelConstraints, i::Int)
    return Constraints.constraints[i]
  end

  function updateWeight(Constraints::ModelConstraints, i::Int, mul::T = 2) where (T<:Number)
    if isinf(Constraints.totalWeight + (mul - 1) * Constraints.weights[i]) || isinf(Constraints.weights[i] * mul)
      println("WARNING: Weight exceeds float64 and so not updated.")
      return 0
    end
    Constraints.totalWeight += (mul - 1) * Constraints.weights[i]
    Constraints.weights[i] *= mul 
    update!(Constraints.PB, i, Constraints.weights[i])
    return 1
  end

  # addConstraints(constraints, R)
  #
  # Input:
  #   constraints, set of constraints
  #   R, a sampled sorted vector such that R[i] is a valid index of constraints.
  #
  # Post: 
  #   model is modified such that all constraints represented by R is added in.
  #
  # Output: None
  #
  function addConstraints(modelConstraints::ModelConstraints, R::Vector{Int})
    #affRIdx = searchsortedfirst(R, constraints.numAffConstraints, lt=<=)
    #if (constraints.numAffEqualTo != 0)
    #  @constraint(model, )
    #end
    for c in modelConstraints.constraints[R]
      add_constraint(modelConstraints.model, c)
    end
  end

  function addConstraints(modelConstraints::ModelConstraints, R)
    for c in R
      add_constraint(modelConstraints.model, c)
    end
  end

  function setOptimizer(model::Model)
    set_optimizer(model, () -> Gurobi.Optimizer(Gurobi.Env()))
    #set_attribute(model, "Threads", Threads.nthreads())
    set_attribute(model, "InfUnbdInfo", 1)
    #set_attribute(model, "Presolve", 0)
    #set_attribute(model, "DualReductions", 0)
    #set_attribute(model, "Method", 0)
    #set_silent(model)
  end

  function createNewSampledModel(modelConstraints::ModelConstraints, R::Vector{Int})
    clearConstraints(modelConstraints.model, modelConstraints.include_variable)
    addConstraints(modelConstraints, R)
    newModel, newModelMap = copy_model(modelConstraints.model)
    sampledConstraintRefs = all_constraints(newModel, include_variable_in_set_constraints = modelConstraints.include_variable)

    #for c in all_constraints(model, VariableRef, MOI.LessThan{Float64})
    #  delete(model, c)
    #end
    # Add number of edges cannot exceed n.
    #var = all_variables(newModel)
    #num_vert = 0
    #while(true)
    #  num_vert += 1
    #  if (num_vert * (num_vert-1)) == 2 * length(var)
    #    break
    #  end
    #end
    #@constraint(newModel, dot(ones(length(var)), var) <= num_vert + 1)

    return newModel, sampledConstraintRefs, newModelMap
  end

  function violatedConstraints(constraints::ModelConstraints, point::Vector{Float64})

    totalW = 0
    for i in 1:length(constraints.weights)
      totalW += constraints.weights[i]
    end
    if (abs(totalW - constraints.totalWeight) > 1e-6)
      println("ERROR: totalWeight is not calculated correctly.")
    end
    violated = Int[]
    is_feasible = true
    violated_weight = 0
    m = constraints.numAffConstraints
    n = length(point)

    startTime = time_ns()
    # Fast pass: compute Ax with Float64
    A = constraints.data.A
    LHSData = A * point

    # Identify constraints needing precise check (within 1e-4 of boundary)
    COARSE_TOL = 1e-4
    needs_precise = Int[]
    for i in 1:m
      ax_val = LHSData[i]
      b_lower = constraints.data.b_lower[i]
      b_upper = constraints.data.b_upper[i]
      # Check if close to lower bound
      if abs(ax_val - b_lower) < COARSE_TOL
        push!(needs_precise, i)
      # Check if close to upper bound
      elseif isfinite(b_upper) && abs(ax_val - b_upper) < COARSE_TOL
        push!(needs_precise, i)
      end
    end

    # Precise pass: recompute only ambiguous rows with BigFloat + Kahan
    if !isempty(needs_precise)
      point_big = BigFloat.(point)
      for i in needs_precise
        sum_val = BigFloat(0.0)
        comp = BigFloat(0.0)
        for j in 1:n
          if A[i, j] != 0
            y = BigFloat(A[i, j]) * point_big[j] - comp
            t = sum_val + y
            comp = (t - sum_val) - y
            sum_val = t
          end
        end
        LHSData[i] = Float64(sum_val)
      end
    end
    endTime = time_ns()
    println("time to calculate Ax (fast + ", length(needs_precise), " precise): ", (endTime - startTime)/1e9)

    # Use relative tolerance: tol = EPS * max(|Ax|, |b|, 1)
    startTime = time_ns()
    for i in 1:m
      ax_val = LHSData[i]
      b_val = constraints.data.b_lower[i]
      scale = max(abs(ax_val), abs(b_val), 1.0)
      tol = EPS * scale
      if ax_val < b_val - tol
        push!(violated, i)
        violated_weight += constraints.weights[i]
        is_feasible = false
      elseif isfinite(constraints.data.b_upper[i])
        b_upper = constraints.data.b_upper[i]
        scale_upper = max(abs(ax_val), abs(b_upper), 1.0)
        tol_upper = EPS * scale_upper
        if ax_val > b_upper + tol_upper
          push!(violated, i)
          violated_weight += constraints.weights[i]
          is_feasible = false
        end
      end
    end

    # Check variable bounds with relative tolerance
    for i in 1:n
      lb = constraints.data.x_lower[i]
      ub = constraints.data.x_upper[i]
      if isfinite(lb)
        scale_lb = max(abs(point[i]), abs(lb), 1.0)
        if point[i] < lb - EPS * scale_lb
          push!(violated, i + m)
          violated_weight += constraints.weights[i+m]
          is_feasible = false
          continue
        end
      end
      if isfinite(ub)
        scale_ub = max(abs(point[i]), abs(ub), 1.0)
        if point[i] > ub + EPS * scale_ub
          push!(violated, i + m)
          violated_weight += constraints.weights[i+m]
          is_feasible = false
        end
      end
    end
    endTime = time_ns()
    println("time to check violation: ", (endTime - startTime)/1e9)

    return is_feasible, violated, violated_weight
  end

  function sample(model::ModelConstraints, r::Int64, include_index::Vector{Int64} = Vector{Int64}(undef, 0))
    ret = sort(unique(rand(model.rng, model.weights, r)))
    println("Percent of unique sampled constraints: ", length(ret)/r)
    ret = unique(vcat(ret, include_index))
    return ret
  end

  function clearConstraints(model::Model, include_variable::Bool)
    println("clearing constriants")
    startTime = time_ns()
    constraints = all_constraints(model, include_variable_in_set_constraints=include_variable)
    for c in constraints
      delete(model, c)
    end
    endTime = time_ns()
    println("finish clearing: ", (endTime - startTime)/1e9)

    setOptimizer(model)
  end

  function myLsqr(A, b)
    m, n = size(A)
    k = size(b, 2)
    X = zeros(n, k)
    for i in 1:k
      bi = b[:, i]
      xi = lsmr(A, bi)
      X[:, i] = xi
    end
    return X
  end

  # Input: model containing a LP.
  #
  # Post: Transform the LP into the form of min c^T x, A x >= b
  #
  # Output: None
  #
  function transform_model!(model::Model)
    variable_lower = all_constraints(model, VariableRef, MOI.GreaterThan{Float64})
    variable_upper = all_constraints(model, VariableRef, MOI.LessThan{Float64})
    variable_equal = all_constraints(model, VariableRef, MOI.EqualTo{Float64})
    variable_int = all_constraints(model, VariableRef, MOI.Interval{Float64})
    #aff_lower = all_constraints(model, AffExpr, MOI.GreaterThan{Float64})
    aff_upper = all_constraints(model, AffExpr, MOI.LessThan{Float64})
    aff_equal = all_constraints(model, AffExpr, MOI.EqualTo{Float64})
    aff_int = all_constraints(model, AffExpr, MOI.Interval{Float64})

    for con_ref in variable_lower
      co = constraint_object(con_ref)
      @constraint(model, co.func >= co.set.lower)
      delete(model, con_ref)
    end

    for con_ref in vcat(variable_upper, aff_upper)
      co = constraint_object(con_ref)
      @constraint(model, -co.func >= -co.set.upper)
      delete(model, con_ref)
    end

    for con_ref in vcat(variable_equal, aff_equal)
      co = constraint_object(con_ref)
      @constraint(model, co.func >= co.set.value)
      @constraint(model, -co.func >= -co.set.value)
      delete(model, con_ref)
    end

    for con_ref in vcat(variable_int, aff_int)
      co = constraint_object(con_ref)
      @constraint(model, co.func >= co.set.lower)
      @constraint(model, -co.func >= -co.set.upper)
      delete(model, con_ref)
    end

    if (objective_sense(model) == MAX_SENSE)
      @objective(m, Min, -objective_function(m))
    end

    bounded_box_constraint = []
    # Put a bounded box.
    for var in all_variables(model)
      push!(bounded_box_constraint, @constraint(model, -var >= -1e6))
      push!(bounded_box_constraint, @constraint(model, var >= -1e6))
    end
    return [ i.value for i in index.(bounded_box_constraint) ]
  end

  # Clarkson(model)
  #
  # Input: model containing the LP.
  #
  # Output: Return an optimal value and primal solution to the LP.
  #
  function Clarkson(model::Model, alpha::Number=2, include_variable::Bool=false,
                    topPercent::Float64=0.1, beta::Number=2)
    # Initial setup stage:
    bounded_box_constraint_index = transform_model!(model)
    #bounded_box_constraint_index = []
    constraintTypes = list_of_constraint_types(model)
    modelConstraints = @time ModelConstraints(model, include_variable)
    n = length(all_variables(model))
    r = 6*n^2
    r = 2*n*trunc(Int64, log2(n)+1)
    objSense = objective_sense(model)
    warmVBasis::Union{Nothing, Vector{Int}} = nothing
    warmCBasis::Union{Nothing, Dict{Int, Int}} = nothing
    warmConstr::Union{Nothing, Vector{Int}} = nothing

    numOfViolatedIterates = []
    optimalityIterates = []
    timeToOptimize = []
    timeToCheckConstraints = []
    timeToSample = []
    timeToAddConstraints = []
    timeToUpdateWeights = []
    objValues = []
    numIt = 0
    while true
      numIt += 1
      # Sampling procedure:
      startTime = time_ns()
      #R = @timeit to "sample()" sample(modelConstraints, r, bounded_box_constraint_index)
      R = @timeit to "sample()" sample(modelConstraints, r)
      if warmVBasis !== nothing
          R = sort(unique(vcat(R, warmConstr)))
      end
      endTime = time_ns()
      push!(timeToSample, (endTime - startTime)/1e9)
      startTime = time_ns()
      newModel, sampledConstraintRefs, newModelMap = @timeit to "createNewSampledModel()" createNewSampledModel(modelConstraints, R)

      endTime = time_ns()
      push!(timeToAddConstraints, (endTime - startTime)/1e9)
      setOptimizer(newModel)
      #set_optimizer(newModel, () -> Gurobi.Optimizer(Gurobi.Env()))
      #set_attribute(newModel, "InfUnbdInfo", 1)
      # Solve base case.
      startTime = time_ns()
      if warmVBasis !== nothing
          println("Setting warm start basis.")
          grb = backend(newModel)
          # Set VBasis on all variables
          for (j, var) in enumerate(all_variables(newModel))
              MOI.set(grb, Gurobi.VariableAttribute("VBasis"), index(var), warmVBasis[j])
          end
          # Set CBasis on all constraints
          all_cons_ws = all_constraints(newModel; include_variable_in_set_constraints = false)
          for (j, con) in enumerate(all_cons_ws)
              if haskey(warmCBasis, R[j])
                  MOI.set(grb, Gurobi.ConstraintAttribute("CBasis"), index(con), warmCBasis[R[j]])
              else
                  # New constraint: slack is basic (not tight)
                  MOI.set(grb, Gurobi.ConstraintAttribute("CBasis"), index(con), 0)
              end
          end
          set_attribute(newModel, "LPWarmStart", 2)
          set_attribute(newModel, "Method", 1)
      end
      @timeit to "optimize!" optimize!(newModel)
      endTime = time_ns()
      push!(timeToOptimize, (endTime - startTime)/1e9)
      status = termination_status(newModel)
      optimalPrimal = nothing
      y = nothing
      dual_reduced_cost = nothing
      c_basis = nothing
      println(status)
      push!(optimalityIterates, status)
      if status == MOI.OPTIMAL
        # Extract the optimal solution in the same variable order as data.A columns
        # data.variables contains the variable ordering used in the matrix
        optimalPrimal = [value(newModelMap[v]) for v in modelConstraints.data.variables]
        #y = shadow_price.(all_constraints(newModel, include_variable_in_set_constraints=modelConstraints.include_variable))
        #y = shadow_price.(sampledConstraintRefs)
        dual_reduced_cost = modelConstraints.b - modelConstraints.data.A * optimalPrimal
        #dual_reduced_cost = constraints.data.b_upper - constraints.data.A * x
        grb_backend = backend(newModel)
        all_cons = all_constraints(newModel; include_variable_in_set_constraints = false)
        c_basis = [i for i in 1:length(all_cons) if MOI.get(grb_backend, Gurobi.ConstraintAttribute("CBasis"), index(all_cons[i])) != 0 ]
        println("len(c_basis)", length(c_basis))
        println("len(var)", size(modelConstraints.data.A)[2])
        if (length(c_basis) != size(modelConstraints.data.A)[2])
          println("-----They are not equal!!!!!-----")
        end
        push!(objValues, objective_value(newModel))

      elseif status == MOI.DUAL_INFEASIBLE && primal_status(newModel) == MOI.INFEASIBILITY_CERTIFICATE
        unbounded_ray = value(all_variables(newModel))
        set_objective(newModel, objSense, 0)
        optimize!(newModel)
        println("The sampled LP is unbounded.")
        optimalPrimal = 100 * unbounded_ray + value(all_variables(newModel))
        push!(objValues, NaN)
      elseif status == MOI.INFEASIBLE
        println("The original LP is infeasible.")
        return false
      else
        println("Base case did not have an optimal solution. Continue to next iteration...")
        continue
      end
      # Check violated constraints
      startTime = time_ns()
      is_feasible, V, violated_weight = @timeit to "violatedConstraints()" violatedConstraints(modelConstraints, optimalPrimal)
      endTime = time_ns()
      push!(timeToCheckConstraints, (endTime - startTime)/1e9)
      println("There are: ", length(V), " constraints violated.")
      println("The weight of violated constraints are: ", violated_weight)
      println("The current threshold is: ", (2*n*modelConstraints.totalWeight)/r)
      println(bucket_info(modelConstraints.PB, V, R))

      # DEBUG: Check for constraints that are both sampled AND violated
      sampled_and_violated = intersect(Set(R), Set(V))
      if !isempty(sampled_and_violated)
        println("=== BUG DEBUG: $(length(sampled_and_violated)) constraints are BOTH sampled AND violated! ===")
        for idx in first(collect(sampled_and_violated), 3)  # Show first 3
          c = modelConstraints.constraints[idx]
          ax_val = dot(modelConstraints.data.A[idx, :], optimalPrimal)
          b_val = modelConstraints.data.b_lower[idx]
          println("  Constraint $idx: $(c.func) >= $(c.set.lower)")
          println("    Ax = $ax_val, b = $b_val, violation = $(b_val - ax_val)")
        end
        println("=== END BUG DEBUG ===")
      end
      #println("shadow price:", y)
      push!(numOfViolatedIterates, length(V))
      if isempty(V)
        println("Violations: ", numOfViolatedIterates)
        println("Optimality: ", optimalityIterates)
        println("Time to optimize: ", timeToOptimize)
        println("Time to check violation: ", timeToCheckConstraints)
        println("Time to sample: ", timeToSample)
        println("Time to add constraints: ", timeToAddConstraints)
        println("Time to update weights: ", timeToUpdateWeights)
        println("Objective: ", objValues)
        println("Relative Objective: ", [abs(o - objective_value(newModel))/objective_value(newModel) for o in objValues])
        println("Total iterations:", length(numOfViolatedIterates))
        show(to)
        return objective_value(newModel), optimalPrimal
        # Note our totalWeight can always be bounded by m^2, since |H| <=
        # (1+1/(3n))^{n ln(m)} m ~ m^2
      elseif violated_weight < (2*n*modelConstraints.totalWeight)/r
        startTime = time_ns()
        c_basis = nothing
        if c_basis == nothing
          @timeit to "update weight on violated constraint" begin
          levels = sort([ (get_level(modelConstraints.PB, i), i) for i in V ], rev=true)
          for v in V
            updateWeight(modelConstraints, v, alpha)
          end
          end #@timeit to "update weight on violated constraint" begin
        end
        endTime = time_ns()
        push!(timeToUpdateWeights, (endTime - startTime)/1e9)
      else
        println("Not updated becuase too many constraints are violated.")
      end
      if length(V) <= r
        r = 6*n^2
      end
      if status == MOI.OPTIMAL && length(V) <= n
        # Extract basis for warm starting future iterations
        grb = backend(newModel)
        warmVBasis = [MOI.get(grb, Gurobi.VariableAttribute("VBasis"), index(v)) for v in all_variables(newModel)]
        all_cons_extract = all_constraints(newModel; include_variable_in_set_constraints = false)
        warmCBasis = Dict(R[j] => MOI.get(grb, Gurobi.ConstraintAttribute("CBasis"), index(all_cons_extract[j])) for j in 1:length(all_cons_extract))
        # Only remember the tight (non-basic) constraints — the ~n constraints that define the optimal vertex
        warmConstr = [R[j] for j in 1:length(all_cons_extract) if MOI.get(grb, Gurobi.ConstraintAttribute("CBasis"), index(all_cons_extract[j])) != 0]
      end
    end # While end

  end

  export Clarkson
end # module clarkson
