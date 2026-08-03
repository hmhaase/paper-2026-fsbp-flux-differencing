# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function prolong2interfaces!(cache, u, mesh::TreeMesh{1}, equations, dg::DGFSBP)
    @unpack interfaces = cache
    bases = cache.elements.bases

    @threaded for interface in eachinterface(dg, cache)
        left_element = interfaces.neighbor_ids[1, interface]
        right_element = interfaces.neighbor_ids[2, interface]
        boundary_value!(view(interfaces.u, 1, :, interface), u[left_element],
                        bases[left_element], Positive())
        boundary_value!(view(interfaces.u, 2, :, interface), u[right_element],
                        bases[right_element], Negative())
    end

    return nothing
end

function prolong2boundaries!(backend::Nothing, cache, u, mesh::TreeMesh{1}, equations,
                             dg::DGFSBP)
    @unpack boundaries = cache
    bases = cache.elements.bases

    @threaded for boundary in eachboundary(dg, cache)
        element = boundaries.neighbor_ids[boundary]
        if boundaries.neighbor_sides[boundary] == 1
            # element is in −x direction of boundary: extract from its +x (right) face
            boundary_value!(view(boundaries.u, 1, :, boundary), u[element],
                            bases[element], Positive())
        else
            # element is in +x direction of boundary: extract from its −x (left) face
            boundary_value!(view(boundaries.u, 2, :, boundary), u[element],
                            bases[element], Negative())
        end
    end

    return nothing
end

@inline function weak_form_kernel!(du, u,
                                   element, ::Type{<:TreeMesh{1}},
                                   have_nonconservative_terms::False, equations,
                                   dg::DGFSBP, cache, alpha = true)
    basis = cache.elements.bases[element]
    derivative_dhat = basis.derivative_dhat
    u_element = u[element]
    du_element = du[element]

    for i in eachnode(basis)
        u_node = get_node_vars(u_element, equations, dg, i)

        flux1 = flux(u_node, 1, equations)
        for ii in eachnode(basis)
            multiply_add_to_node_vars!(du_element, alpha * derivative_dhat[ii, i],
                                       flux1,
                                       equations, dg, ii)
        end
    end

    return nothing
end

@inline function flux_differencing_kernel!(du, u,
                                           element,
                                           ::Type{<:TreeMesh{1}},
                                           have_nonconservative_terms::False, equations,
                                           volume_flux, dg::DGFSBP, cache, alpha = true)
    basis = cache.elements.bases[element]
    derivative_split = basis.derivative_split
    u_element = u[element]
    du_element = du[element]

    # Calculate volume integral in one element
    for i in eachnode(basis)
        u_node = get_node_vars(u_element, equations, dg, i)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.

        # x direction
        for ii in (i + 1):nnodes(basis)
            u_node_ii = get_node_vars(u_element, equations, dg, ii)
            flux1 = volume_flux(u_node, u_node_ii, 1, equations)
            multiply_add_to_node_vars!(du_element, alpha * derivative_split[i, ii],
                                       flux1,
                                       equations, dg, i)
            multiply_add_to_node_vars!(du_element, alpha * derivative_split[ii, i],
                                       flux1,
                                       equations, dg, ii)
        end
    end
end

function calc_surface_integral!(backend::Nothing, du, u, mesh::TreeMesh{1}, equations,
                                surface_integral::SurfaceIntegralWeakForm, dg::DGFSBP,
                                cache)
    # Note that all fluxes have been computed with outward-pointing normal vectors.
    @unpack surface_flux_values = cache.elements

    @threaded for element in eachelement(dg, cache)
        apply_surface_lift_line!(du[element],
                                 view(surface_flux_values, :, 1, element),
                                 view(surface_flux_values, :, 2, element),
                                 cache.elements.bases[element], equations)
    end

    return nothing
end

function apply_jacobian!(backend::Nothing, du::WrappedArray, mesh::TreeMesh{1},
                         equations, dg::DGFSBP, cache)
    # FSBP absorbs the Jacobian into the weights; only the global minus sign remains.
    du.a .*= -one(eltype(du))
    return nothing
end
end # @muladd
