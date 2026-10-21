# Values come from the included elixir module and `semi`/`solver`, not from the
# trixi_include overrides, so parameters left at an elixir default are recorded too.
# A generator that drives the time integration itself must pass every such value it sets
# differently from the elixir (e.g. `tspan`, `ode_algorithm`, `cfl`) as a `setup_metadata`
# keyword.

using Trixi

# Each case parameter is recorded only if the elixir defines it. Probing `isdefined`
# relies on each run getting a fresh module (see `with_elixir`); a shared one would leak
# names across runs.
const CASE_PARAMS = (:alpha_ex, :shock_speed)

# Node count per axis, as DGSEM's `nnodes` counts it. A per-cell basis is read from the
# first element only, so this assumes every element has the same count.
function nnodes_per_element(solver, semi)
    _, _, _, cache = Trixi.mesh_equations_solver_cache(semi)
    solver.basis isa Trixi.PerCellBasis || return Trixi.nnodes(solver)
    # `nnodes(dg, elements, element)` is per-element only in 1D, so ask the basis directly.
    # For a tensor-product basis `ndofs_per_element` multiplies the axes; take one axis.
    basis = cache.elements.bases[1]
    basis isa Trixi.TensorProductCellBasis && return Trixi.nnodes(first(basis.bases))
    return Trixi.ndofs_per_element(basis)
end

# "closed": nodes on both cell boundaries; "open": a `GeneralSBPCellBasis`, whose nodes
# need not include them.
function node_family(solver, semi)
    _, _, _, cache = Trixi.mesh_equations_solver_cache(semi)
    solver.basis isa Trixi.PerCellBasis || return "closed"
    basis = cache.elements.bases[1]
    basis isa Trixi.TensorProductCellBasis && (basis = first(basis.bases))
    return basis isa Trixi.GeneralSBPCellBasis ? "open" : "closed"
end

# Read from the elixir, not inferred from the node count: for FSBP the two can differ.
function space_dimension(solver, m)
    solver.basis isa Trixi.PerCellBasis || return Trixi.polydeg(solver) + 1
    (isdefined(m, :fsbp_space) && isdefined(m, :fsbp_space_dim)) || return nothing
    return getfield(m, :fsbp_space_dim)(getfield(m, :fsbp_space))
end

"""
    setup_metadata(elixir, m, semi, solver; overrides...)

Named tuple describing the run. `m` is the module from `with_elixir`; call this inside
that block. Keywords are merged in last, replacing fields of the same name.
"""
function setup_metadata(elixir, m, semi, solver; overrides...)
    val(s) = getfield(m, s)
    # A name the elixir does not define is recorded as null, not guessed; a caller that
    # knows the value passes it as an override.
    maybe(s) = isdefined(m, s) ? getfield(m, s) : nothing
    typename(x) = isnothing(x) ? nothing : string(nameof(typeof(x)))
    vi = solver.volume_integral
    # polydeg errors on a PerCellBasis; fsbp_space is meaningless for DGSEM.
    fsbp = solver.basis isa Trixi.PerCellBasis
    base = (; elixir = basename(elixir),
            equations = string(nameof(typeof(semi.equations))),
            gamma = hasproperty(semi.equations, :gamma) ? semi.equations.gamma : nothing,
            initial_condition = string(semi.initial_condition),
            source_terms = isnothing(semi.source_terms) ? nothing :
                           string(semi.source_terms),
            domain = [maybe(:coordinates_min), maybe(:coordinates_max)],
            periodic = semi.mesh.tree.periodicity[1],
            tspan = collect(val(:tspan)),
            ode_algorithm = typename(maybe(:ode_algorithm)),
            cfl = maybe(:cfl),
            solver = fsbp ? "FSBP $(something(maybe(:fsbp_space), "?"))" :
                     "DGSEM P$(Trixi.polydeg(solver))",
            volume_integral = string(nameof(typeof(vi))),
            volume_flux = hasproperty(vi, :volume_flux) ? string(vi.volume_flux) : nothing,
            surface_flux = string(solver.surface_integral.surface_flux),
            nodes_per_element = nnodes_per_element(solver, semi),
            node_family = node_family(solver, semi),
            dim_F = space_dimension(solver, m),
            analysis_polydeg = maybe(:analysis_polydeg),
            adaptive = maybe(:adaptive),
            # Free-form slot for integrator settings the elixir does not name, e.g. the
            # relaxation solver's root/gamma tolerances.
            integrator_options = nothing,
            trixi_version = string(pkgversion(Trixi)),
            (s => val(s) for s in CASE_PARAMS if isdefined(m, s))...)
    return merge(base, NamedTuple(overrides))
end
