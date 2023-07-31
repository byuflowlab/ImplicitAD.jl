
"""
    implicit_eigval(A, B, eigsolve)

Make eigenvalue problems AD compatible with ForwardDiff and ReverseDiff

# Arguments
- `A::matrix`, `B::matrix`: generlized eigenvalue problem. A v = λ B v  (B is identity for standard eigenvalue problem)
- `eigsolve::function`: λ, V, U = eigsolve(A, B). Function to solve the eigenvalue problem.
    λ is a vector containing eigenvalues.
    V is a matrix whose columns are the corresponding eigenvectors (i.e., λ[i] corresponds to V[:, i]).
    U is a matrix whose columns contain the left eigenvectors (u^H A = λ u^H B)
    The left eigenvectors must be in the same order as the right ones (i.e., U' * B * V must be diagonal).
    U can be provided with any normalization as we normalize internally s.t. U' * B * V = I
    If A and B are symmetric/Hermitian then U = V.

# Returns
- `λ::vector`: eigenvalues and their derivatives.  (Currently only eigenvalue derivatives are provided.  not eigenvectors)
"""
implicit_eigval(A, B, eigsolve) = eigsolve(A, B)[1]  # If no AD, just solve normally. Returns only eigenvalue.

# forward cases
implicit_eigval(A::AbstractArray{<:ForwardDiff.Dual{T}}, B::AbstractArray{<:ForwardDiff.Dual{T}}, eigsolve) where {T} = eigval_fwd(A, B, eigsolve)
implicit_eigval(A, B::AbstractArray{<:ForwardDiff.Dual{T}}, eigsolve) where {T} = eigval_fwd(A, B, eigsolve)
implicit_eigval(A::AbstractArray{<:ForwardDiff.Dual{T}}, B, eigsolve) where {T} = eigval_fwd(A, B, eigsolve)

# reverse cases
implicit_eigval(A::Union{ReverseDiff.TrackedArray, AbstractArray{<:ReverseDiff.TrackedReal}}, B::Union{ReverseDiff.TrackedArray, AbstractArray{<:ReverseDiff.TrackedReal}}, eigsolve) = eigval_rev(A, B, eigsolve)
implicit_eigval(A, B::Union{ReverseDiff.TrackedArray, AbstractArray{<:ReverseDiff.TrackedReal}}, eigsolve) = eigval_rev(A, B, eigsolve)
implicit_eigval(A::Union{ReverseDiff.TrackedArray, AbstractArray{<:ReverseDiff.TrackedReal}}, B, eigsolve) = eigval_rev(A, B, eigsolve)

# extract values from A and B before passing to common function
eigval_fwd(A, B, eigsolve) = eigval_deriv(A, B, ForwardDiff.value.(A), ForwardDiff.value.(B), eigsolve)
eigval_rev(A, B, eigsolve) = eigval_deriv(A, B, ReverseDiff.value(A), ReverseDiff.value(B), eigsolve)

# eigenvalue derivatives used in both forward and reverse
function eigval_deriv(A, B, Av, Bv, eigsolve)
    λ, V, U = eigsolve(Av, Bv)
    U = U ./ (diag(U' * Bv * V))'  # normalize U s.t. U' * B * V = I

    # compute derivatives
    etype = promote_type(eltype(A), eltype(B))
    λd = similar(λ, complex(etype))
    for i = 1:length(λ)
        λd[i] = λ[i] + view(U, :, i)'*(A - λ[i]*B)*view(V, :, i)  # right hand side is zero for primal.  left side is zero for dual (only and A and B contain derivatives)
    end

    return λd
end
