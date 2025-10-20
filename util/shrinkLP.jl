using JuMP, ArgParse, LinearAlgebra, Random

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
        #"--numbers", "-n"
        #    help = "number of redundant constraints to add"
        #    arg_type = Int64
        #    default = 1
    end

    return parse_args(s)
end

function reconstruct_model_from_lp_data(A, b_lower, b_upper, x_lower, x_upper, c, c_offset)
    new_model = Model()

    n_vars = length(x_lower)
    @variable(new_model, x_lower[i] <= x[i=1:n_vars] <= x_upper[i])
    @constraint(new_model, b_lower .<= A * x .<= b_upper)
    @objective(new_model, Min, dot(c, x) + c_offset)

    return new_model
end

function modify_lp_data(A, b_lower, b_upper, x_lower, x_upper, c)
    """
    Modifies the LP data by randomly selecting some variables (columns of A),
    computing a linear combination of them,
    replacing the selected columns with this new column, and setting the new
    variable's bounds to the min lower and max upper of the selected ones.
    
    Returns the modified A, b_lower, b_upper, x_lower, x_upper.
    Note: b_lower and b_upper remain unchanged.
    """
    m, n = size(A)
    if n < 2
        return A, b_lower, b_upper, x_lower, x_upper
    end
    
    k = rand(2:min(10, n))
    selected = sort(randperm(n)[1:k])
    
    coeffs = ones(k) 
    new_col = A[:, selected] * coeffs
    non_selected = setdiff(1:n, selected)
    new_A = hcat(A[:, non_selected], new_col)
    new_c = vcat(c[non_selected], dot(c[selected], coeffs))
    new_x_lower = vcat(x_lower[non_selected], minimum(x_lower[selected]))
    new_x_upper = vcat(x_upper[non_selected], maximum(x_upper[selected]))
    return new_A, b_lower, b_upper, new_x_lower, new_x_upper, new_c
end


function main()
    parsed_args = parse_commandline()

    input_file = parsed_args["input_file"]
    output_file = parsed_args["output_file"]
    model = read_from_file(input_file)
    vars = all_variables(model)
    lp_data = lp_matrix_data(model)
    A = lp_data.A
    b_lower = lp_data.b_lower
    b_upper = lp_data.b_upper
    x_lower = lp_data.x_lower
    x_upper = lp_data.x_upper
    c = lp_data.c
    c_offset = lp_data.c_offset

    m = size(A)[1]
    print("m, n", m, " ", size(A)[2])
    while (m < 10*size(A)[2]^2)
      A, b_lower, b_upper, x_lower, x_upper, c = modify_lp_data(A, b_lower, b_upper, x_lower, x_upper, c)
    end
    new_model = reconstruct_model_from_lp_data(A, b_lower, b_upper, x_lower, x_upper, c, c_offset)
    write_to_file(new_model, output_file)
end #main()

main()
