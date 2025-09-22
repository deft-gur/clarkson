using JuMP, Dualization

using ArgParse, LinearAlgebra

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "input_file"
            help = "a required input file"
            required = true
        "--output_file", "-o"
            help = "an optional output file"
            arg_type = String
            default = "output.mps"
        "--numbers", "-n"
            help = "number of redundant constraints to add"
            arg_type = Int64
            default = 1
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    input_file = parsed_args["input_file"]
    output_file = parsed_args["output_file"]
    num_to_add = parsed_args["numbers"]
    model = read_from_file(input_file)
    vars = all_variables(model)
    data = lp_matrix_data(model)
    A = data.A
    n, m = size(A)
    b_upper = data.b_upper
    b_lower = data.b_lower
    b_upper_masked = map(x -> (isinf(x) ? 0.0 : x) , b_upper)
    b_lower_masked = map(x -> (isinf(x) ? 0.0 : x) , b_lower)

    for i in 1:num_to_add
      # Flip a coin to decide add upper or lower bound.
      #y = rand(Float64, n)
      y = rand(0:4, n)
      y = [convert(Float64, j) for j in y]
      upper = Bool(rand(0:1))
      b = upper ? b_upper : b_lower
      b_masked = upper ? b_upper_masked : b_lower_masked
      x = [(isinf(b[j]) ? 0.0 : y[j]) for j in 1:n]
      LHS = transpose(x) * A
      RHS = dot(b_masked, x)

      if upper
        @constraint(model, dot(LHS, vars) <= RHS + rand(Float64) * 1e-4)
      else
        @constraint(model, dot(LHS, vars) >= RHS - rand(Float64) * 1e-4)
        #@constraint(model, dot(LHS, vars) >= RHS)
      end

    end

    write_to_file(model, output_file)
end

main()
