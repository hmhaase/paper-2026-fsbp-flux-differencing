# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function calc_volume_integral!(backend::Nothing, du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralFluxDifferencing,
                               dg::DGFSBP, cache)
    @threaded for element in eachelement(dg, cache)
        flux_differencing_kernel!(du, u, element, typeof(mesh),
                                  have_nonconservative_terms, equations,
                                  volume_integral.volume_flux, dg, cache)
    end

    return nothing
end

function calc_volume_integral!(backend::Nothing, du, u, mesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGFSBP, cache)
    @threaded for element in eachelement(dg, cache)
        weak_form_kernel!(du, u, element, typeof(mesh),
                          have_nonconservative_terms, equations,
                          dg, cache)
    end

    return nothing
end
end # @muladd
