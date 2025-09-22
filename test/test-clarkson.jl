using JuMP, Gurobi
using Profile

include("../src/clarkson.jl")
using .CLARKSON

# check_feasibility_and_objective(model, sol)
#
# Checks whether or not sol is a feasible solution of model and return its
# objective value.
#
# Input:
#   model, Model representing a linear program.
#   sol, is a dictionary from variable to values.
#
# TODO: Remove this dependency and assume if sol doesn't set a particular value
# then it is automatically 0.
# Pre: sol contains all the variable values of model.
# 
# Output: is_feasible, obj_val
#   is_feasible, Boolean
#   obj_val, Float64
#
function check_feasibility_and_objective(model::Model, sol::Dict{VariableRef, Float64})
    vars = all_variables(model)
    if Set(keys(sol)) != Set(vars)
        error("The solution dictionary must provide values for all variables in the model.")
    end

    original_lbs = Dict{VariableRef, Float64}()
    original_ubs = Dict{VariableRef, Float64}()
    no_lb_vars = VariableRef[]
    no_ub_vars = VariableRef[]
    for var in vars
        if has_lower_bound(var)
            original_lbs[var] = lower_bound(var)
        else
            push!(no_lb_vars, var)
        end
        if has_upper_bound(var)
            original_ubs[var] = upper_bound(var)
        else
            push!(no_ub_vars, var)
        end
    end

    for (var, val) in sol
        set_lower_bound(var, val)
        set_upper_bound(var, val)
    end

    optimize!(model)
    status = termination_status(model)

    is_feasible = false
    obj_val = nothing
    if status == MOI.OPTIMAL
        is_feasible = true
        obj_val = objective_value(model)
    elseif status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
        is_feasible = false
    else
        error("Unexpected termination status: $status. Check solver output for details.")
    end

    for (var, lb) in original_lbs
        set_lower_bound(var, lb)
    end
    for (var, ub) in original_ubs
        set_upper_bound(var, ub)
    end
    for var in no_lb_vars
        unset_lower_bound(var)
    end
    for var in no_ub_vars
        unset_upper_bound(var)
    end

    return is_feasible, obj_val
end

function test_clarkson(filename::String)
  tol = 1e-6
  model = read_from_file(filename)
  obj, sol = @time clarkson(model)

  set_optimizer(model, Gurobi.Optimizer)
  set_silent(model)
  optimize!(model)
  status = termination_status(model)
  if status != MOI.OPTIMAL || abs(objective_value(model) - obj) > tol
    return false
  end
  return true
  is_feasible, obj_val = check_feasibility_and_objective(model, sol)

  if !is_feasible || abs(obj_val - obj) > tol
    return false
  end

  return true
end

#@assert test_clarkson("input.mps")
#@assert test_clarkson("bench/netlarge1.mps")
#@assert test_clarkson("bench/16_n14.mps")
#@assert test_clarkson("bench/a2864.mps")
#@assert test_clarkson("bench/qap15.mps")
#@assert test_clarkson("bench/fit1d.mps")
@assert test_clarkson("bench/fit1d-dual.mps")
