### A Pluto.jl notebook ###
# v0.19.15

using Markdown
using InteractiveUtils

function get_expr_type(expr)
    if expr isa Symbol
        return Val(:variable)
    else
        return Val(expr.head)
    end
end

function ensure_math(s)
    "\\ensuremath{$s}"
end

function convert_loop(expr, loop; level=0)
    condition = expr.args[begin]
    body = expr.args[end]
    s_condition = convert_expr(condition; level=0, delim=", ")
    s_condition = replace(s_condition, raw"\State" => "")
    s_body = convert_expr(body, level=level + 1)
    """
    \\$loop{$(ensure_math(s_condition))} 
    $s_body
    \\End$loop{}
    """
end

function convert_expr(::Val{:return}, expr; kw...)
    s = "\\Return "
    args = if expr.args isa Vector
        mapreduce(convert_expr, *, expr.args)
    else
        convert_expr(expr.args)
    end
    return s * ensure_math(args)
end

function need_mathrm(name)
    !isnothing(match(r"[a-zA-Z]{2,}", name))
end

function convert_expr(::Val{:variable}, name; kw...)
    s = string(name)
    s = replace(s, r"′" => "'")
    #= s = replace(s, r"_([a-zA-Z]{2,})" => s"_{\\mathrm{\1}}") =#
    return need_mathrm(s) ? "\\mathrm{$s}" : "\\ensuremath{$s}"
end
function convert_expr(::Val{:ref}, expr; kw...)
    convert_expr(Val(:variable), expr; kw...)
end

function convert_expr(::Val{:operator}, name; kw...)
    return "\\operatorname{$name}"
end

function need_brace(expr)
    if !isa(expr, Expr)
        return false
    end
    return expr.head == :call
end

function convert_expr(::Val{:call}, expr; kw...)
    operator = expr.args[begin]
    s = if operator in [:+, :-, :*, :/, :÷]
        args = expr.args[2:end]
        # @assert length(args) == 2
        args = map(args) do arg
            if need_brace(arg)
                raw"(" * convert_expr(arg) * raw")"
            else
                convert_expr(arg)
            end
        end
        s_opt = string(operator)
        join(args, s_opt)
    else
        name = convert_expr(Val(:operator), operator)
        name = replace(name, ":" => "range")
        args = join(convert_expr.(expr.args[2:end]), ", ")
        "$(name)\\left($(args)\\right)"
    end
    return s
end

function convert_expr(::Val{:block}, expr; level=0, kw...)
    delim = get(kw, :delim, "\n")
    states = map(expr.args) do e
        T = get_expr_type(e)
        s = convert_expr(T, e; level=level + 1)
        s = if T in (Val(:block), Val(:for), Val(:while))
            s
        else
            raw"\State " * s
        end

        format_level(s, level=level + 1)
    end
    join(states, delim)
end

function convert_expr(::Val{:for}, expr; kw...)
    convert_loop(expr, "For"; kw...)
end

function convert_expr(::Val{:while}, expr; level=0, kw...)
    s = convert_loop(expr, "While"; kw...)
    format_level(s; level=level)
end

function convert_expr(::Val{:tuple}, expr; level=0, kw...)
    args = convert_expr.(expr.args)
    "[" * join(args, ", ") * "]"
end

function convert_expr(::Val{:(=)}, expr; level=0, kw...)
    lhs_e, rhs_e = expr.args
    lhs = convert_expr(lhs_e)
    rhs = convert_expr(rhs_e)
    s = ensure_math("$lhs \\gets $rhs")
    return s
end

function convert_expr(expr::Number; level=0, kw...)
    return string(expr)
end

function convert_expr(expr; level=0, kw...)
    Base.remove_linenums!(expr)
    s = convert_expr(get_expr_type(expr), expr; level=level, kw...)
end

function format_level(s; level=0, indent="  ")
    lines = split(s, "\n")
    lines = map(lines) do line
        "$(indent ^ level)$(line)"
    end
    join(lines, "\n")
end

#= res = convert_expr( =#
#=     quote =#
#=         m, n, p = size(img) =#
#=         grad = zeros(eltype(img), m, n, p, 2) =#
#=         for i in 2:m-1, j in 2:n-1, k in 2:p-1 =#
#=             grad[i, j, k, 1] = (img[i+1, j, p] - img[i-1, j, p]) =#
#=             grad[i, j, k, 2] = (img[i, j+1, p] - img[i, j-1, p]) =#
#=             grad[i, j, k, 3] = (img[i, j, p+1] - img[i, j, p-1]) =#
#=         end =#
#=         grad * 0.5 =#
#=     end) =#

#= clipboard(res) =#
#= println(res) =#
