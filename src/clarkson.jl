module clarkson
  using DataStructures
  using JuMP, Gurobi
  using Random, WeightVectors
  using LinearAlgebra
  using Dualization
  using IterativeSolvers
  using SparseArrays
  using TimerOutputs

  EPS = 1e-6

  const to = TimerOutput()

  mutable struct ModelConstraints
      constraints::Vector{ScalarConstraint}
      numConstraints::Int64
      weights::WeightVectors.AbstractWeightVector
      totalWeight::Float64
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
      # NOTE: We assume that all_constraints will return constraints
      # with the following priority:
      #   1. Affine constraint =, >=, <=
      #   2. Variable constraint =, >=, <=
      constraints = constraint_object.(all_constraints(model; include_variable_in_set_constraints = include_variable))
      i = 1
      m = length(constraints)
      data = lp_matrix_data(model)
      # TODO: Right now we only support =, <=, >=, but not of type # MOI.Interval{float64}.
      numAffEqualTo = length(all_constraints(model, AffExpr, MOI.EqualTo{Float64}))
      numAffLessThan = length(all_constraints(model, AffExpr, MOI.LessThan{Float64}))
      numAffGreaterThan = length(all_constraints(model, AffExpr, MOI.GreaterThan{Float64}))

      numVarEqualTo = length(all_constraints(model, VariableRef, MOI.EqualTo{Float64}))

      numVarLessThan = length(all_constraints(model, VariableRef, MOI.LessThan{Float64}))

      numVarGreaterThan = length(all_constraints(model, VariableRef, MOI.GreaterThan{Float64}))

      numVarConstraints = numVarEqualTo + numVarLessThan + numVarGreaterThan
      numAffConstraints = numAffEqualTo + numAffLessThan + numAffGreaterThan

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
    newModel, _ = copy_model(modelConstraints.model)
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

    return newModel, sampledConstraintRefs
  end

  function violatedConstraints(constraints::ModelConstraints, point::Vector{Float64})

    totalW = 0
    for i in 1:length(constraints.weights)
      totalW += constraints.weights[i]
    end
    if (abs(totalW - constraints.totalWeight) > 1e-6)
      println("ERROR: totalWeight is not calculated correctly.")
    end
    violated = []
    is_feasible = true
    violated_weight = 0
    m = constraints.numAffConstraints
    n = length(point)

    startTime = time_ns()
    LHSData = constraints.data.A * point
    endTime = time_ns()
    println("time to calculate Ax: ", (endTime - startTime)/1e9)

    # =, >=, <=
    violationConstrVector = LHSData .>= (constraints.data.b_lower - EPS * ones(m))
    violationConstrVector = violationConstrVector .& (LHSData .<= (constraints.data.b_upper + EPS * ones(m)))
    violatedVarVector = point .>= constraints.data.x_lower - EPS*ones(n)
    violatedVarVector = violatedVarVector .& (point .<= constraints.data.x_upper + EPS*ones(n))
    startTime = time_ns()
    for i in 1:m
      if (violationConstrVector[i] == false)
       push!(violated, i)
       violated_weight += constraints.weights[i]
       is_feasible = false
      end
    end
    for i in 1:n
      if (violatedVarVector[i] == false)
        push!(violated, i + m)
        violated_weight += constraints.weights[i+m]
        is_feasible = false
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
    #aff_lower = all_constraints(model, AffExpr, MOI.GreaterThan{Float64})
    aff_upper = all_constraints(model, AffExpr, MOI.LessThan{Float64})
    aff_equal = all_constraints(model, AffExpr, MOI.EqualTo{Float64})

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
    constraintTypes = list_of_constraint_types(model)
    modelConstraints = @time ModelConstraints(model, include_variable)
    n = length(all_variables(model))
    r = 6*n^2
    r = 2*n*trunc(Int64, log2(n)+1)
    objSense = objective_sense(model)

    numOfViolatedIterates = []
    optimalityIterates = []
    timeToOptimize = []
    timeToCheckConstraints = []
    timeToSample = []
    timeToAddConstraints = []
    timeToUpdateWeights = []
    objValues = []
    while true
      # Sampling procedure:
      startTime = time_ns()
      R = @timeit to "sample()" sample(modelConstraints, r, bounded_box_constraint_index)
      endTime = time_ns()
      push!(timeToSample, (endTime - startTime)/1e9)
      startTime = time_ns()
      newModel, sampledConstraintRefs = @timeit to "createNewSampledModel()" createNewSampledModel(modelConstraints, R)
      endTime = time_ns()
      push!(timeToAddConstraints, (endTime - startTime)/1e9)
      setOptimizer(newModel)
      #set_optimizer(newModel, () -> Gurobi.Optimizer(Gurobi.Env()))
      #set_attribute(newModel, "InfUnbdInfo", 1)
      # Solve base case.
      startTime = time_ns()
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
        # Extract the optimal solution.
        #optimalPrimal = Dict(zip(all_variables(newModel), value(all_variables(newModel))))
        optimalPrimal = value(all_variables(newModel))
        #y = shadow_price.(all_constraints(newModel, include_variable_in_set_constraints=modelConstraints.include_variable))
        y = shadow_price.(sampledConstraintRefs)
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
        if c_basis == nothing
          @timeit to "update weight on violated constraint" begin
          for v in V
            updateWeight(modelConstraints, v, alpha)
          end
          end #@timeit to "update weight on violated constraint" begin
        end
        if c_basis != nothing
          @timeit to "steepest edge rule" begin
          #dual_model, primal_dual_map = dualize(newModel)
          #setOptimizer(dual_model)
          #for (primal_con, dual_var) in primal_dual_map.primal_con_dual_var
          #  set_start_value(dual_var, dual(primal_con))
          #end
          #for (primal_var, dual_con) in primal_dual_map.primal_var_dual_con
          #  set_dual_start_value(dual_con, value(primal_var))
          #end
          #optimize!(dual_model)
          candidate_indices = (objSense == MIN_SENSE) ? findall(dual_reduced_cost .> EPS) : findall(dual_reduced_cost .< EPS)
          if sort(V) != sort(candidate_indices)
            println("------NOT EQUAL!!!!----")
            exit(1)
          end
          #candidate_indices = (objSense == MAX_SENSE) ? findall(dual_reduced_cost .> EPS) : findall(dual_reduced_cost .< EPS)
          #statuses = get_attribute.(all_variables(newModel), MOI.VariableBasisStatus())
          #statuses = get_attribute.(sampledConstraintRefs, MOI.ConstraintBasisStatus())
          #basic_indices = findall(s -> s == MOI.BASIC, statuses)


          #basic_indices = findall(abs.(y) .> EPS)
          basic_indices = R[c_basis]
          ## Calculate d = A_B^(-T) A^T
          A_B = modelConstraints.data.A[basic_indices, :]
          #if (size(A_B, 1) < size(A_B, 2))
          #  A_B = vcat(A_B, zeros(size(A_B, 2) - size(A_B, 1), size(A_B, 2)))
          #end
          #println(size(A_B))
          #println(size(modelConstraints.data.A[candidate_indices, :]))
          #d = myLsqr(transpose(A_B), transpose(modelConstraints.data.A[candidate_indices, :]))
          
          #QR = qr(transpose(A_B))
          #Q = QR.Q
          #R = QR.R
          #d = inv(R) * transpose(Q) * transpose(modelConstraints.data.A[candidate_indices, :])
          LU = lu(sparse(transpose(A_B)))
          #d = LU \ Matrix(transpose(modelConstraints.data.A[candidate_indices, :]))
          d = LU \ Matrix(transpose(modelConstraints.data.A[candidate_indices, :]))

          
          #d = Matrix(transpose(A_B))\Matrix(transpose(modelConstraints.data.A[candidate_indices, :]))
          #rowNorm = vec(sqrt.(sum(abs2, modelConstraints.data.A[candidate_indices, :] * transpose(inv(A_B)), dims=2)))
          #@time println(A_B' \ modelConstraints.data.A[candidate_indices, :]')
          #@time println(modelConstraints.data.A[candidate_indices, :] * transpose(inv(Matrix(A_B))))
          #
          # b^T - b_B^T (A_B)^-T A^T
          colNorm = vec(sqrt.(sum(abs2, d, dims=1)))
          top = sort([Pair(abs(dual_reduced_cost[i]/colNorm[j]), i) for (j,i) in enumerate(candidate_indices)], rev=true)
          #top = sort([Pair(abs(dual_reduced_cost[i]/colNorm[j]), i) for (j,i) in enumerate(candidate_indices)], rev=false)
          top = first(top, max(trunc(Int64, length(top) * topPercent), 10))
          #top_five = first(sort([Pair(abs(dual_reduced_cost[i]), i) for i in candidate_indices], rev=true), 40)
          for (_, i::Int) in top
            updateWeight(modelConstraints, i, beta)
          end
          end # @timeit to "steepest edge rule"
        end
        #if y != nothing
        #  non_zero_indices = (objSense == MIN_SENSE) ? findall(y .< -EPS) : findall(y .> EPS)
        #  # = sort([Pair(abs(y[i]/modelConstraints.rowNorm[R[i]]), i) for i in non_zero_indices], rev=true)
        #  denom = sum([ abs(y[i]) for i in non_zero_indices ])
        #  #top_five = first(sort([Pair(abs(y[i]/modelConstraints.rowNorm[R[i]]), i) for i in non_zero_indices], rev=true), 5)
        #  #for (_, i::Int) in top_five
        #  for i::Int in non_zero_indices
        #    updateWeight(modelConstraints, R[i], 1.0 + 5.0 * abs(y[i]/denom))
        #    #if (isAffConstraint(modelConstraints, R[i])) 
        #    #  println(y[i]/modelConstraints.rowNorm[R[i]])
        #    #  updateWeight(modelConstraints, R[i], 1+abs(y[i]/modelConstraints.rowNorm[R[i]]))
        #    #end
        #  end
        #end
        endTime = time_ns()
        push!(timeToUpdateWeights, (endTime - startTime)/1e9)
      else
        println("Not updated becuase too many constraints are violated.")
      end
      #if status == MOI.OPTIMAL
      #  r = 6*n^2
      #end
      if length(V) <= 100
        r = 6*n^2
      end
    end # While end

  end

  export Clarkson
end # module clarkson
