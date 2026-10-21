using Trixi
using JSON3

const DATA_DIR = normpath(joinpath(@__DIR__, "..", "data"))

"""
    with_elixir(f, elixir; mapexpr = identity, overrides...)

`trixi_include` `elixir` with `overrides` into a fresh module `m`, with stdout silenced
during the include, and return `f(m)`. `mapexpr` is applied to each top-level expression
of the elixir.

A fresh module keeps one run's definitions from leaking into the next. They are newer
than any compiled caller, so `f` runs under `Base.invokelatest`; return plain data from
`f`, since a value carrying functions the elixir defined needs `invokelatest` again.
"""
function with_elixir(f, elixir; mapexpr = identity, overrides...)
    m = Module(:ElixirRun)
    # A bare `Module` has no `include`, which the elixirs call.
    Base.eval(m, :(include(p) = Base.include($m, p)))
    redirect_stdout(devnull) do
        trixi_include(mapexpr, m, elixir; overrides...)
    end
    return Base.invokelatest(f, m)
end

"""
    write_dataset(name, data)

Write `data` as JSON to `data/<name>.json`, creating any subdirectory `name` contains, and
return the path.
"""
function write_dataset(name, data)
    path = joinpath(DATA_DIR, name * ".json")
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, data)
        println(io)
    end
    println("\n-> $path")
    return path
end
