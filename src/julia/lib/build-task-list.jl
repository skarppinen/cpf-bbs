include("../../../config.jl");
import ArgParse

"""
Function checks if and ArgParse.Field object is identified by
the letter in the "short option" of a command line argument.
"""
function identified_by_short_name(field::ArgParse.ArgParseField)
    dest_name = field.dest_name;
    length(dest_name) > 1 && dest_name in field.short_opt_name;
end


"""
Function generates a text line representing a call to a command line script.

Arguments:
`script_name`: The name of the script file to be included in the beginning of the call.
`args`: Key value pairs that give the arguments and values used in the call.
These should match with what is found in `s`. (checked)
`s`: An ArgParseSettings object describing what arguments the script file is anticipating.
"""
function generate_call(script_name::AString, args::Dict{String}, s::ArgParse.ArgParseSettings)
    if !endswith(script_name, ".jl")
        script_name *= ".jl";
    end

    # Get fields object of ArgParseSettings i.e objects representing arguments that can be set
    # in the script.
    fields = s.args_table.fields;

    # Check that all keys in `args` actually correspond to some field in ArgParseSettings.
    # (prevent mistyped argname).
    dest_names = getfield.(fields, :dest_name);
    for key in keys(args)
        if !(key in dest_names)
            msg = string("key `", key,
                         "` in input arguments does not match any input argument of script.");
            throw(ArgumentError(msg));
        end
    end

    # Build the command to `cmd_string`.
    cmd_string = string("julia", " ", script_name);
    for field in fields
        dest_name = field.dest_name;
        value = field.default; # = nothing if no default.
        if dest_name in keys(args)
            value = args[dest_name];
        end
        if isnothing(value)
            # Value is still nothing. If field is not required, move on, otherwise
            # error since field is required, has no default and was not given a value.
            !field.required && (continue;)
            msg = string("argument `", dest_name, "` not found in arguments and has no default.");
            throw(ArgumentError(msg));
        end

        # Figure out how argument should be represented in command string.
        flag = ArgParse.is_flag(field);
        short_arg = identified_by_short_name(field);
        prefix = short_arg ? "-" : "--";
        if flag
            # Argument is a flag. Leave empty if value is false.
            to_append = "";
            value && (to_append = string(" ", prefix, dest_name);)
        else
            # Short arguments use `-` instead of "--".
            # Long arguments have "=".
            midfix = short_arg ? " " : "=";

            # If value is a string representation of a Dict, need to
            # put quotes around. Here matching start with any number of whitespace
            # and then "Dict".
            if value isa AbstractString && startswith(value, r"^\s*Dict")
                value = string("\"", value, "\"");
            end

            to_append = string(" ", prefix, dest_name, midfix, value);
        end
        cmd_string *= to_append;
    end
    cmd_string *= "\n"; # End the line (command now complete.)
    cmd_string;
end
