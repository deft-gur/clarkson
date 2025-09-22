using JuMP, Dualization

if length(ARGS) != 2
    println("Usage: julia dualize.jl input.mps output.mps")
    exit(1)
end

input_file = ARGS[1]
output_file = ARGS[2]

primal = read_from_file(input_file)

println(primal)

#dual = dualize(primal; dual_names = DualNames("dual_var_", "dual_con_"))
dual = dualize(primal)

println(dual)

write_to_file(dual, output_file)


println("Dual LP exported to $output_file")
