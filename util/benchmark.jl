using JuMP, Gurobi
using Profile
using ArgParse, LinearAlgebra

include("../src/clarkson.jl")
using .CLARKSON

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "input_file"
            help = "a required input file."
            required = true
        #"--output_file", "-o"
        #    help = "an optional output file."
        #    arg_type = String
        #    default = "output.mps"
        "--gurobi", "-g"
            help = "Use native gurobi."
            arg_type = Bool
            default = false
    end

    return parse_args(s)
end


function benchmark_clarkson(filename::String)
  model = read_from_file(filename)
  obj, sol = @time clarkson(model)
  return true
end

function benchmark_gurobi(filename::String)
  model = read_from_file(filename)
  set_optimizer(model, () -> Gurobi.Optimizer(Gurobi.Env()))
  set_attribute(model, "InfUnbdInfo", 1)
  #set_attribute(model, "Presolve", 0)
  #set_attribute(model, "DualReductions", 0)
  set_attribute(model, "Method", 0)

  startTime = time_ns()
  @time optimize!(model)
  endTime = time_ns()
  println((endTime-startTime)/1e9)
  return true
end

function main()
  parsed_args = parse_commandline()

  input_file = parsed_args["input_file"]
  use_gurobi = parsed_args["gurobi"]

  if (use_gurobi)
    benchmark_gurobi(input_file)
  else
    benchmark_clarkson(input_file)
  end
end

main()

#@assert benchmark_clarkson("bench/afiro-red.mps")
#@assert benchmark_clarkson("bench/qap15.mps")
#@assert benchmark_clarkson("bench/my_model.mps")
#@assert benchmark_clarkson("bench/my_model-20.mps")
#@assert benchmark_gurobi("bench/my_model-20.mps")
#@assert benchmark_gurobi("model-squared.mps")
