# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function weak_form_kernel!(du, u, element, ::Type{<:TreeMesh{2}},
                                   have_nonconservative_terms::False, equations,
                                   dg::DGFSBP, cache, alpha = true)
    bx, by = cache.elements.bases[element].bases
    uel = u[element]
    duel = du[element]

    for j in eachnode(by), i in eachnode(bx)
        u_node = get_node_vars(uel, equations, dg, i, j)

        flux1 = flux(u_node, 1, equations)
        for ii in eachnode(bx)
            multiply_add_to_node_vars!(duel, alpha * bx.derivative_dhat[ii, i], flux1,
                                       equations, dg, ii, j)
        end

        flux2 = flux(u_node, 2, equations)
        for jj in eachnode(by)
            multiply_add_to_node_vars!(duel, alpha * by.derivative_dhat[jj, j], flux2,
                                       equations, dg, i, jj)
        end
    end

    return nothing
end

@inline function flux_differencing_kernel!(du, u, element, ::Type{<:TreeMesh{2}},
                                           have_nonconservative_terms::False, equations,
                                           volume_flux, dg::DGFSBP, cache, alpha = true)
    bx, by = cache.elements.bases[element].bases
    uel = u[element]
    duel = du[element]

    # The diagonal of `derivative_split` is zero (SBP), so it is skipped; volume-flux
    # symmetry lets us halve the two-point flux evaluations.
    for j in eachnode(by), i in eachnode(bx)
        u_node = get_node_vars(uel, equations, dg, i, j)

        # x direction
        for ii in (i + 1):nnodes(bx)
            u_node_ii = get_node_vars(uel, equations, dg, ii, j)
            flux1 = volume_flux(u_node, u_node_ii, 1, equations)
            multiply_add_to_node_vars!(duel, alpha * bx.derivative_split[i, ii], flux1,
                                       equations, dg, i, j)
            multiply_add_to_node_vars!(duel, alpha * bx.derivative_split[ii, i], flux1,
                                       equations, dg, ii, j)
        end

        # y direction
        for jj in (j + 1):nnodes(by)
            u_node_jj = get_node_vars(uel, equations, dg, i, jj)
            flux2 = volume_flux(u_node, u_node_jj, 2, equations)
            multiply_add_to_node_vars!(duel, alpha * by.derivative_split[j, jj], flux2,
                                       equations, dg, i, j)
            multiply_add_to_node_vars!(duel, alpha * by.derivative_split[jj, j], flux2,
                                       equations, dg, i, jj)
        end
    end

    return nothing
end

function prolong2interfaces!(backend::Nothing, cache, u, mesh::TreeMesh{2}, equations,
                             dg::DGFSBP)
    @unpack interfaces = cache
    @unpack orientations, neighbor_ids = interfaces
    bases = cache.elements.bases

    @threaded for interface in eachinterface(dg, cache)
        left = neighbor_ids[1, interface]
        right = neighbor_ids[2, interface]
        u_left = u[left]
        u_right = u[right]
        bx_l, by_l = bases[left].bases
        bx_r, by_r = bases[right].bases

        # Faces are conforming (see `check_conforming_faces`): the left element's
        # tangential basis indexes both sides.
        if orientations[interface] == 1
            # interface in x-direction: left +x face, right -x face; tangential index = y
            for j in eachnode(by_l)
                boundary_value!(view(interfaces.u, 1, :, j, interface),
                                view(u_left, :, :, j), bx_l, Positive())
                boundary_value!(view(interfaces.u, 2, :, j, interface),
                                view(u_right, :, :, j), bx_r, Negative())
            end
        else
            # interface in y-direction: left +y face, right -y face; tangential index = x
            for i in eachnode(bx_l)
                boundary_value!(view(interfaces.u, 1, :, i, interface),
                                view(u_left, :, i, :), by_l, Positive())
                boundary_value!(view(interfaces.u, 2, :, i, interface),
                                view(u_right, :, i, :), by_r, Negative())
            end
        end
    end

    return nothing
end

function calc_interface_flux!(backend::Nothing, surface_flux_values, mesh::TreeMesh{2},
                              have_nonconservative_terms::False, equations,
                              surface_integral, dg::DGFSBP, cache)
    @unpack surface_flux = surface_integral
    @unpack u, neighbor_ids, orientations = cache.interfaces

    @threaded for interface in eachinterface(dg, cache)
        left_id = neighbor_ids[1, interface]
        right_id = neighbor_ids[2, interface]

        # orientation 1: left -> 2 (+x), right -> 1 (-x); orientation 2: left -> 4, right -> 3
        left_direction = 2 * orientations[interface]
        right_direction = 2 * orientations[interface] - 1

        # tangential node count
        bx_l, by_l = cache.elements.bases[left_id].bases
        n_face = orientations[interface] == 1 ? nnodes(by_l) : nnodes(bx_l)
        for i in 1:n_face
            u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
            flux = surface_flux(u_ll, u_rr, orientations[interface], equations)
            for v in eachvariable(equations)
                surface_flux_values[v, i, left_direction, left_id] = flux[v]
                surface_flux_values[v, i, right_direction, right_id] = flux[v]
            end
        end
    end

    return nothing
end

function calc_interface_flux!(backend::Nothing, surface_flux_values, mesh::TreeMesh{2},
                              have_nonconservative_terms::True, equations,
                              surface_integral, dg::DGFSBP, cache)
    #TODOFSBP: nonconservative terms in 2D
    return error("nonconservative terms are not implemented for DGFSBP on TreeMesh{2}")
end

function prolong2boundaries!(backend::Nothing, cache, u, mesh::TreeMesh{2}, equations,
                             dg::DGFSBP)
    @unpack boundaries = cache
    @unpack orientations, neighbor_sides = boundaries
    bases = cache.elements.bases

    @threaded for boundary in eachboundary(dg, cache)
        element = boundaries.neighbor_ids[boundary]
        u_el = u[element]
        bx, by = bases[element].bases

        if orientations[boundary] == 1
            if neighbor_sides[boundary] == 1
                # element in -x direction of boundary: extract its +x face
                for l in eachnode(by)
                    boundary_value!(view(boundaries.u, 1, :, l, boundary),
                                    view(u_el, :, :, l), bx, Positive())
                end
            else
                # element in +x direction of boundary: extract its -x face
                for l in eachnode(by)
                    boundary_value!(view(boundaries.u, 2, :, l, boundary),
                                    view(u_el, :, :, l), bx, Negative())
                end
            end
        else
            if neighbor_sides[boundary] == 1
                # element in -y direction of boundary: extract its +y face
                for l in eachnode(bx)
                    boundary_value!(view(boundaries.u, 1, :, l, boundary),
                                    view(u_el, :, l, :), by, Positive())
                end
            else
                # element in +y direction of boundary: extract its -y face
                for l in eachnode(bx)
                    boundary_value!(view(boundaries.u, 2, :, l, boundary),
                                    view(u_el, :, l, :), by, Negative())
                end
            end
        end
    end

    return nothing
end

function calc_boundary_flux!(backend::Nothing, cache, t,
                             boundary_conditions::NamedTuple, mesh::TreeMesh{2},
                             equations, surface_integral, dg::DGFSBP)
    @unpack surface_flux_values = cache.elements
    @unpack n_boundaries_per_direction = cache.boundaries

    lasts = accumulate(+, n_boundaries_per_direction)
    firsts = lasts - n_boundaries_per_direction .+ 1

    # Unrolled with literal indices so heterogeneous boundary conditions stay type-stable.
    calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[1],
                                     have_nonconservative_terms(equations), equations,
                                     surface_integral, dg, cache, 1, firsts[1],
                                     lasts[1])
    calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[2],
                                     have_nonconservative_terms(equations), equations,
                                     surface_integral, dg, cache, 2, firsts[2],
                                     lasts[2])
    calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[3],
                                     have_nonconservative_terms(equations), equations,
                                     surface_integral, dg, cache, 3, firsts[3],
                                     lasts[3])
    calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[4],
                                     have_nonconservative_terms(equations), equations,
                                     surface_integral, dg, cache, 4, firsts[4],
                                     lasts[4])

    return nothing
end

function calc_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any, 4},
                                          t,
                                          boundary_condition,
                                          have_nonconservative_terms::False, equations,
                                          surface_integral, dg::DGFSBP, cache,
                                          direction, first_boundary, last_boundary)
    @unpack surface_flux = surface_integral
    @unpack u, neighbor_ids, neighbor_sides, node_coordinates,
    orientations = cache.boundaries

    @threaded for boundary in first_boundary:last_boundary
        neighbor = neighbor_ids[boundary]
        bx, by = cache.elements.bases[neighbor].bases
        n_face = orientations[boundary] == 1 ? nnodes(by) : nnodes(bx)
        for i in 1:n_face
            u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, boundary)
            # `neighbor_sides == 1` -> element on the left, its state is the inner one
            u_inner = neighbor_sides[boundary] == 1 ? u_ll : u_rr
            x = get_node_coords(node_coordinates, equations, dg, i, boundary)
            flux = boundary_condition(u_inner, orientations[boundary], direction, x, t,
                                      surface_flux, equations)
            for v in eachvariable(equations)
                surface_flux_values[v, i, direction, neighbor] = flux[v]
            end
        end
    end

    return nothing
end

function calc_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any, 4},
                                          t,
                                          boundary_condition,
                                          have_nonconservative_terms::True, equations,
                                          surface_integral, dg::DGFSBP, cache,
                                          direction, first_boundary, last_boundary)
    #TODOFSBP: nonconservative terms in 2D
    return error("nonconservative terms are not implemented for DGFSBP on TreeMesh{2}")
end

function calc_surface_integral!(backend::Nothing, du, u, mesh::TreeMesh{2}, equations,
                                surface_integral::SurfaceIntegralWeakForm, dg::DGFSBP,
                                cache)
    @unpack surface_flux_values = cache.elements

    @threaded for element in eachelement(dg, cache)
        bx, by = cache.elements.bases[element].bases
        duel = du[element]

        # x-faces (directions 1 = -x, 2 = +x): one lift per y-line
        for l in eachnode(by)
            apply_surface_lift_line!(view(duel, :, :, l),
                                     view(surface_flux_values, :, l, 1, element),
                                     view(surface_flux_values, :, l, 2, element),
                                     bx, equations)
        end

        # y-faces (directions 3 = -y, 4 = +y): one lift per x-line
        for l in eachnode(bx)
            apply_surface_lift_line!(view(duel, :, l, :),
                                     view(surface_flux_values, :, l, 3, element),
                                     view(surface_flux_values, :, l, 4, element),
                                     by, equations)
        end
    end

    return nothing
end

function apply_jacobian!(backend::Nothing, du::WrappedArray, mesh::TreeMesh{2},
                         equations,
                         dg::DGFSBP, cache)
    # FSBP absorbs the Jacobian into the weights; only the global minus sign remains.
    du.a .*= -one(eltype(du))
    return nothing
end

function calc_sources!(backend::Nothing, du, u, t, source_terms::Nothing,
                       equations::AbstractEquations{2}, dg::DGFSBP, cache)
    return nothing
end

function calc_sources!(backend::Nothing, du, u, t, source_terms,
                       equations::AbstractEquations{2}, dg::DGFSBP, cache)
    @threaded for element in eachelement(dg, cache)
        uel = u[element]
        duel = du[element]
        bx, by = cache.elements.bases[element].bases
        for j in eachnode(by), i in eachnode(bx)
            u_local = get_node_vars(uel, equations, dg, i, j)
            x_local = get_node_coords(cache.elements, i, j, element)
            du_local = source_terms(u_local, x_local, t, equations)
            add_to_node_vars!(duel, du_local, equations, dg, i, j)
        end
    end

    return nothing
end

function compute_coefficients!(backend::Nothing, u, func, t, mesh::AbstractMesh{2},
                               equations, dg::DGFSBP, cache)
    @threaded for element in eachelement(dg, cache)
        u_element = u[element]
        bx, by = cache.elements.bases[element].bases
        for j in eachnode(by), i in eachnode(bx)
            x_node = get_node_coords(cache.elements, i, j, element)
            u_node = func(x_node, t, equations)
            set_node_vars!(u_element, u_node, equations, dg, i, j)
        end
    end

    return nothing
end
end # @muladd
