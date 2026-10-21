# Regenerates the datasets behind the paper into data/. Each job runs in its own Julia
# process: the scripts read their options from ARGS and set process-wide state such as
# the BigFloat precision.
#
#   julia --project=. run_all.jl                 the datasets in PAPER
#   julia --project=. run_all.jl --supporting    also those in SUPPORTING
#   julia --project=. run_all.jl --list          print the jobs and exit
#   julia --project=. run_all.jl <job> ...       only the named jobs

const ROOT = @__DIR__
const SCRIPTS = joinpath(ROOT, "scripts")

struct Job
    name::String
    script::String
    args::Vector{String}
    outputs::Vector{String}
    backs::String
end

# `backs` names the result in the paper that each dataset supports.
const PAPER = [
    Job("shock_snapshots", "run_shock_snapshots.jl", String[],
        ["shock_snapshots.json"], "Burgers: solution snapshots"),
    Job("shock", "run_convergence.jl", ["shock"],
        ["shock.json"], "Burgers: smooth-regime error table and convergence plot"),
    Job("burgers_invariants", "run_burgers_invariants.jl", String[],
        ["burgers_invariants.json"], "Burgers: mass and entropy over time"),
    Job("burgers_mms_profile", "run_mms_profile.jl", String[],
        ["burgers_mms_profile.json"], "Burgers MMS: exact solution profile"),
    Job("exp", "run_convergence.jl", ["exp_0_5", "exp_1", "exp_2"],
        ["exp_0_5.json", "exp_1.json", "exp_2.json"],
        "Burgers MMS: error table and convergence plots"),
    Job("exp_1_ec", "run_convergence.jl", ["exp_1_ec"],
        ["exp_1_ec.json"], "Burgers MMS: odd-even EOCs with the EC interface flux (text)"),
    Job("density_wave_invariants", "run_density_wave_invariants.jl", String[],
        ["density_wave_invariants.json"], "Density wave: mass and entropy over time"),
    Job("euler_density_wave", "run_convergence.jl", ["euler_density_wave"],
        ["euler_density_wave.json"], "Density wave: convergence plot"),
    Job("euler_density_wave_weak", "run_convergence.jl", ["euler_density_wave_weak"],
        ["euler_density_wave_weak.json"], "Density wave: weak-form T* exactness (text)"),
    Job("modified_sod", "run_modified_sod.jl", String[],
        [joinpath("sod", "modified_sod.json")], "Sod: solution plot"),
    Job("sod_invariants", "run_sod_invariants.jl", String[],
        ["sod_invariants.json"], "Sod: conservation and entropy residuals"),
    Job("khi", "run_khi_snapshot.jl", ["fd"],
        ["khi_snapshot.json"], "Kelvin-Helmholtz: density at t = 3"),
    Job("khi_failure_time", "run_khi_failure_time.jl", String[],
        ["khi_failure_time.json"], "Kelvin-Helmholtz: failure time (text)")]

# BigFloat runs are slow, so most use a short horizon, with a Float64 run at the same
# horizon as the baseline. An empty `bits` keeps BigFloat's default precision. The full
# horizon runs only the two drifting operators, one process each.
const SUPPORTING = [
    Job("density_wave_invariants_t5", "run_density_wave_invariants.jl", ["tend=5"],
        ["density_wave_invariants_t5.json"], "Density wave: Float64 baseline, t = 5");
    [Job("density_wave_invariants_big$(bits)_t5", "run_density_wave_invariants.jl",
         ["big", "tend=5", (isempty(bits) ? String[] : ["bits=$bits"])...],
         ["density_wave_invariants_big$(bits)_t5.json"],
         "Density wave: higher-precision rerun, t = 5")
     for bits in ("", "96", "128", "160")];
    [Job("density_wave_invariants_big96_$(tag)", "run_density_wave_invariants.jl",
         ["big", "bits=96", "runs=$op"], ["density_wave_invariants_big96_$(tag).json"],
         "Density wave: higher-precision rerun, t = 100")
     for (op, tag) in (("P4", "P4"), ("E*", "Estar"))]]

function run_job(job)
    script = joinpath(SCRIPTS, job.script)
    # One thread, whatever JULIA_NUM_THREADS says: threaded reductions make the summation
    # order, and so the last digits, depend on the thread count.
    cmd = `$(Base.julia_cmd()) --threads=1 --project=$ROOT $script $(job.args)`
    println("==> $(job.name)  ($(job.backs))")
    flush(stdout)
    t = @elapsed run(cmd)
    missing_outputs = filter(o -> !isfile(joinpath(ROOT, "data", o)), job.outputs)
    isempty(missing_outputs) ||
        error("$(job.name) finished without writing $(join(missing_outputs, ", "))")
    println("    done in $(round(t / 60; digits = 1)) min")
end

const OPTIONS = ("--supporting", "--list")

function main(args)
    flags = filter(a -> startswith(a, "--"), args)
    unknown_flags = setdiff(flags, OPTIONS)
    isempty(unknown_flags) ||
        error("unknown option(s) $(unknown_flags); options: $(OPTIONS)")
    jobs = "--supporting" in args ? [PAPER; SUPPORTING] : PAPER
    if "--list" in args
        all_jobs = [PAPER; SUPPORTING]
        w_name = maximum(j -> length(j.name), all_jobs) + 2
        w_out = maximum(j -> length(join(j.outputs, " ")), all_jobs) + 2
        for j in all_jobs
            println(rpad(j.name, w_name), rpad(join(j.outputs, " "), w_out), j.backs)
        end
        return
    end
    names = unique(filter(a -> !startswith(a, "--"), args))
    !isempty(names) && "--supporting" in args &&
        error("--supporting selects a job set; name jobs without it")
    if !isempty(names)
        known = Dict(j.name => j for j in [PAPER; SUPPORTING])
        unknown = setdiff(names, keys(known))
        isempty(unknown) || error("unknown job(s) $(unknown); see --list")
        jobs = [known[n] for n in names]
    end
    foreach(run_job, jobs)
end

main(ARGS)
