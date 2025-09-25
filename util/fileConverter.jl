using MathOptInterface
using ArgParse
using JuMP
using CodecZlib

const MOI = MathOptInterface
const FF = MOI.FileFormats

function parse_commandline()
  s = ArgParseSettings()

  @add_arg_table! s begin
      "input_file"
          help = "a required input file."
          required = true
      "--output_file", "-o"
          help = "an optional output file."
          arg_type = String
          default = "output.mof.json.gz"
  end

  return parse_args(s)
end

function convert(filename::String, outputname::String)
  src = FF.Model(format = FF.FORMAT_MPS)
  MOI.read_from_file(src, filename)

  dst = FF.Model(filename=outputname)
  MOI.copy_to(dst, src)
  MOI.write_to_file(dst, outputname)
end


function converter_main()::Cint
  parsed_args = parse_commandline()

  input_file = parsed_args["input_file"]
  outputname = parsed_args["output_file"]

  convert(input_file, outputname)

  return 0
end

converter_main()
