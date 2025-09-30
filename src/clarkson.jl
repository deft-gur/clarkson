module clarkson
  using DataStructures
  using JuMP, Gurobi
  using Random, WeightVectors

  EPS = 1e-6

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
  end

  function ModelConstraints(OrigModel::Model)
      # NOTE: We assume that all_constraints will return constraints
      # with the following priority:
      #   1. Affine constraint =, >=, <=
      #   2. Variable constraint =, >=, <=
      include_variable = true
      #constraints = all_constraints(model; include_variable_in_set_constraints = true)
      #constraints = all_constraints(model; include_variable_in_set_constraints = false)
      #BaseModel = createBaseModel(model)
      model, ref_map = copy_model(OrigModel)
      constraints = constraint_object.(all_constraints(model; include_variable_in_set_constraints = include_variable))
      i = 1
      m = length(constraints)
      data = lp_matrix_data(model)
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
      LHS = []
      RHS = []
      Operators = []
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
        include_variable
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

  function updateWeight(Constraints::ModelConstraints, i::Int, mul::Float64 = 2.0)
    Constraints.totalWeight += Constraints.weights[i]
    #promote!(Constraints.buckets, Constraints.weights[i]+1, Constraints.weights[i]+2, [i])
    #Constraints.weights[i] += 1
    Constraints.weights[i] *= mul 
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
  function addConstraints(ModelConstraints::ModelConstraints,
      R::Vector{Int})
    #affRIdx = searchsortedfirst(R, constraints.numAffConstraints, lt=<=)
    #if (constraints.numAffEqualTo != 0)
    #  @constraint(model, )
    #end

    for c in ModelConstraints.constraints[R]
      add_constraint(ModelConstraints.model, c)
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
      lb = has_lower_bound(v) ? lower_bound(v) : -Inf
      ub = has_upper_bound(v) ? upper_bound(v) : Inf
      #lb = -Inf
      #ub = Inf
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

  function getLHSData(constraints::ModelConstraints, LHSData, point, i)
    constr_obj = constraint_object(constraints.constraints[i])
    func = constr_obj.func

    if (func isa VariableRef)
      return point[func.index.value]
      if (constr_obj.set isa MOI.EqualTo{Float64})
        return point[i - constraints.numAffConstraints]
      elseif (constr_obj.set isa MOI.GreaterThan{Float64})
        return point[i - constraints.numAffConstraints - constraints.numVarEqualTo]
      else
        return point[i - constraints.numAffConstraints - constraints.numVarEqualTo - constraints.numVarGreaterThan]
      end
    elseif (func isa AffExpr)
      return LHSData[i]
    else
      throw(ErrorException("Cannot determine if it is AffExpr or VariableRef."))
    end
  end

  function getRHSData(constraints::ModelConstraints, i)
    constr_obj = constraint_object(constraints.constraints[i])
      if (constr_obj.set isa MOI.EqualTo{Float64})
        return constr_obj.set.value
      elseif (constr_obj.set isa MOI.GreaterThan{Float64})
        return constr_obj.set.lower
      elseif (constr_obj.set isa MOI.LessThan{Float64})
        return constr_obj.set.upper
      end
  end

  function violatedConstraints(constraints::ModelConstraints, point::Vector{Float64})
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

  function sample(model::ModelConstraints, r::Int64)
    ret = rand(model.rng, model.weights, r)
    return sort(unique(ret))
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

    set_optimizer(model, () -> Gurobi.Optimizer(Gurobi.Env()))
    #set_attribute(model, "Threads", Threads.nthreads())
    set_attribute(model, "InfUnbdInfo", 1)
    #set_attribute(model, "Presolve", 0)
    #set_attribute(model, "DualReductions", 0)
    #set_attribute(model, "Method", 0)
    #write_to_file(model, "model-squared.mps")
    #set_silent(model)
  end

  # Clarkson(model)
  #
  # Input: model containing the LP.
  #
  # Output: Return an optimal value and primal solution to the LP.
  #
  function Clarkson(model::Model)
    # Initial setup stage:
    constraintTypes = list_of_constraint_types(model)
    modelConstraints = ModelConstraints(model)
    n = length(all_variables(model))
    r = 6*n^2
    r = 2*n*trunc(Int64, log2(n)+1)

    numOfViolatedIterates = []
    optimalityIterates = []
    timeToOptimize = []
    timeToCheckConstraints = []
    timeToSample = []
    timeToCreateBaseModel = []
    timeToAddConstraints = []
    objValues = []
    while true
      # Sampling procedure:
      startTime = time_ns()
      @time clearConstraints(modelConstraints.model, modelConstraints.include_variable)
      R = sample(modelConstraints, r)
      endTime = time_ns()
      push!(timeToSample, (endTime - startTime)/1e9)
      startTime = time_ns()
      addConstraints(modelConstraints, R)
      endTime = time_ns()
      push!(timeToAddConstraints, (endTime - startTime)/1e9)
      println("Time to add constraints: ", timeToAddConstraints)
      newModel, _ = copy_model(modelConstraints.model)
      set_optimizer(newModel, () -> Gurobi.Optimizer(Gurobi.Env()))
      set_attribute(newModel, "InfUnbdInfo", 1)
      # Solve base case.
      startTime = time_ns()
      optimize!(newModel)
      endTime = time_ns()
      push!(timeToOptimize, (endTime - startTime)/1e9)
      status = termination_status(newModel)
      optimalPrimal = nothing
      y = nothing
      println(status)
      push!(optimalityIterates, status)
      if status == MOI.OPTIMAL
        # Extract the optimal solution.
        #optimalPrimal = Dict(zip(all_variables(newModel), value(all_variables(newModel))))
        optimalPrimal = value(all_variables(newModel))
        #y = shadow_price.(all_constraints(newModel, include_variable_in_set_constraints=false))
        #dual_reduced_cost = constraints.data.b_lower - constraints.data.A * x
        #dual_reduced_cost = constraints.data.b_upper - constraints.data.A * x
        push!(objValues, objective_value(newModel))

      elseif status == MOI.DUAL_INFEASIBLE && primal_status(newModel) == MOI.INFEASIBILITY_CERTIFICATE
        unbounded_ray = value(all_variables(newModel))
        set_objective(newModel, objective_sense(newModel), 0)
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
      is_feasible, V, violated_weight = violatedConstraints(modelConstraints, optimalPrimal)
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
        println("Objective: ", objValues)
        println("Relative Objective: ", [abs(o - objective_value(newModel))/objective_value(newModel) for o in objValues])
        return objective_value(newModel), optimalPrimal
        # Note our totalWeight can always be bounded by m^2, since |H| <=
        # (1+1/(3n))^{n ln(m)} m ~ m^2
      elseif violated_weight < (2*n*modelConstraints.totalWeight)/r
        for v in V
          updateWeight(modelConstraints, v, 2.0)
        end
        if y != nothing
          println("y:", y)
          for i in 1:length(R)
            if abs(y[i]) > EPS
              updateWeight(modelConstraints, R[i], 9.0)
            end
          end
        end
      else
        data = lp_matrix_data(model)
        x = value(all_variables(newModel))
        println(data.A * x - data.b_lower, data.b_upper - data.A * x)
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
