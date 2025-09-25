using JuMP, Gurobi
using Profile
using ArgParse, LinearAlgebra
using CodecZlib

include("../src/clarkson.jl")
using .clarkson

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
          action = :store_true
  end

  return parse_args(s)
end


function benchmark_clarkson(filename::String)
  model = @time read_from_file(filename)
  obj, sol = @time Clarkson(model)
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

function bench_main()::Cint
  parsed_args = parse_commandline()

  input_file = parsed_args["input_file"]
  use_gurobi = parsed_args["gurobi"]
  println("use_gurobi:", use_gurobi)

  if (use_gurobi)
    benchmark_gurobi(input_file)
  else
    benchmark_clarkson(input_file)
  end

  return 0
end

bench_main()

#@assert benchmark_clarkson("bench/afiro-red.mps")
#@assert benchmark_clarkson("bench/qap15.mps")
#@assert benchmark_clarkson("bench/my_model.mps")
#@assert benchmark_clarkson("bench/my_model-20.mps")
#@assert benchmark_gurobi("bench/my_model-20.mps")
#@assert benchmark_gurobi("model-squared.mps")
