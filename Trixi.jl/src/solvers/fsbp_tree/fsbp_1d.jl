# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# `reset_du!` override: the generic version calls `du[.., element]` which
# allocates an unsafe_wrap'd view per element. Directly zeroing the flat
# backing vector avoids all allocations.
function reset_du!(du::WrappedArray, dg::DGFSBP, cache)
    fill!(du.a, zero(eltype(du)))
    return nothing
end

@inline function extract_boundary_value!(dest, u, element,
                                         basis::LobattoLikeCellBasis, ::Negative)
    dest .= @view u[element][:, 1]
end
@inline function extract_boundary_value!(dest, u, element,
                                         basis::LobattoLikeCellBasis, ::Positive)
    dest .= @view u[element][:, end]
end

@inline function extract_boundary_value!(dest, u, element,
                                         basis::GeneralSBPCellBasis, ::Negative)
    mul!(dest, u[element], @view basis.boundary_interpolation[1, :])
end
@inline function extract_boundary_value!(dest, u, element,
                                         basis::GeneralSBPCellBasis, ::Positive)
    mul!(dest, u[element], @view basis.boundary_interpolation[2, :])
end

function prolong2interfaces!(cache, u, mesh::TreeMesh{1}, equations, dg::DGFSBP)
    @unpack interfaces = cache
    bases = cache.elements.bases

    @threaded for interface in eachinterface(dg, cache)
        left_element = interfaces.neighbor_ids[1, interface]
        right_element = interfaces.neighbor_ids[2, interface]
        extract_boundary_value!(view(interfaces.u, 1, :, interface), u, left_element,
                                bases[left_element], Positive())
        extract_boundary_value!(view(interfaces.u, 2, :, interface), u, right_element,
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
            extract_boundary_value!(view(boundaries.u, 1, :, boundary), u, element,
                                    bases[element], Positive())
        else
            # element is in +x direction of boundary: extract from its −x (left) face
            extract_boundary_value!(view(boundaries.u, 2, :, boundary), u, element,
                                    bases[element], Negative())
        end
    end

    return nothing
end

@inline function weak_form_kernel!(du, u,
                                   element, mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                   have_nonconservative_terms::False, equations,
                                   dg::DGFSBP, cache, alpha = true)
    derivative_dhat = cache.elements.bases[element].derivative_dhat

    for i in eachnode(dg, cache.elements, element)
        u_node = get_node_vars(u, equations, dg, i, element)

        flux1 = flux(u_node, 1, equations)
        for ii in eachnode(dg, cache.elements, element)
            multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], flux1,
                                       equations, dg, ii, element)
        end
    end

    return nothing
end

@inline function flux_differencing_kernel!(du, u,
                                           element,
                                           mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                           have_nonconservative_terms::False, equations,
                                           volume_flux, dg::DGFSBP, cache, alpha = true)
    derivative_split = cache.elements.bases[element].derivative_split

    # Calculate volume integral in one element
    for i in eachnode(dg, cache.elements, element)
        u_node = get_node_vars(u, equations, dg, i, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.

        # x direction
        for ii in (i + 1):nnodes(dg, cache.elements, element)
            u_node_ii = get_node_vars(u, equations, dg, ii, element)
            flux1 = volume_flux(u_node, u_node_ii, 1, equations)
            multiply_add_to_node_vars!(du, alpha * derivative_split[i, ii], flux1,
                                       equations, dg, i, element)
            multiply_add_to_node_vars!(du, alpha * derivative_split[ii, i], flux1,
                                       equations, dg, ii, element)
        end
    end
end

@inline function apply_surface_lift!(du, surface_flux_values, element,
                                     basis::LobattoLikeCellBasis, equations)
    # boundary_lift is nonzero only at the two endpoint nodes for Lobatto-like bases
    boundary_lift = basis.boundary_lift
    n = nnodes(basis)
    factor_neg = boundary_lift[1, 1]
    factor_pos = boundary_lift[n, 2]
    for v in eachvariable(equations)
        du[v, 1, element] = du[v, 1, element] -
                            surface_flux_values[v, 1, element] * factor_neg
        du[v, n, element] = du[v, n, element] +
                            surface_flux_values[v, 2, element] * factor_pos
    end
end

@inline function apply_surface_lift!(du, surface_flux_values, element,
                                     basis::GeneralSBPCellBasis, equations)
    boundary_lift = basis.boundary_lift
    for i in eachnode(basis)
        factor_neg = boundary_lift[i, 1]
        factor_pos = boundary_lift[i, 2]
        for v in eachvariable(equations)
            du[v, i, element] = (du[v, i, element] -
                                 surface_flux_values[v, 1, element] * factor_neg +
                                 surface_flux_values[v, 2, element] * factor_pos)
        end
    end
end

function calc_surface_integral!(backend::Nothing, du, u,
                                mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                                equations, surface_integral, dg::DGFSBP, cache)
    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # We also use explicit assignments instead of `+=` to let `@muladd` turn these
    # into FMAs (see comment at the top of the file).
    @unpack surface_flux_values = cache.elements

    @threaded for element in eachelement(dg, cache)
        apply_surface_lift!(du, surface_flux_values, element,
                            cache.elements.bases[element], equations)
    end

    return nothing
end

function apply_jacobian!(backend::Nothing, du, mesh::TreeMesh{1},
                         equations, dg::DGFSBP, cache)
    ET = eltype(du)
    @threaded for element in eachelement(dg, cache)
        for i in eachnode(dg, cache.elements, element)
            for v in eachvariable(equations)
                du[v, i, element] *= -one(ET)
            end
        end
    end

    return nothing
end
end # @muladd
