# This file defines TensorAlgebra interface for a FusionTensor

using BlockArrays: Block
using GradedArrays: space_isequal
using LinearAlgebra: mul!
using TensorAlgebra: TensorAlgebra, AbstractBlockPermutation, FusionStyle, blockedperm,
    genperm, matricize, trivialbiperm, unmatricize

function TensorAlgebra.output_axes(
        ::typeof(contract),
        biperm_dest::AbstractBlockPermutation{2},
        a1::FusionTensor,
        biperm1::AbstractBlockPermutation{2},
        a2::FusionTensor,
        biperm2::AbstractBlockPermutation{2},
    )
    axes_codomain, axes_contracted = blocks(axes(a1)[biperm1])
    axes_contracted2, axes_domain = blocks(axes(a2)[biperm2])
    @assert all(space_isequal.(dual.(axes_contracted), axes_contracted2))
    flat_axes = genperm((axes_codomain..., axes_domain...), Tuple(biperm_dest))
    return FusionTensorAxes(
        tuplemortar(
            (
                flat_axes[begin:length_codomain(biperm_dest)],
                flat_axes[(length_codomain(biperm_dest) + 1):end],
            )
        ),
    )
end

struct FusionTensorFusionStyle <: FusionStyle end

TensorAlgebra.FusionStyle(::Type{<:FusionTensor}) = FusionTensorFusionStyle()

unval(::Val{x}) where {x} = x

function TensorAlgebra.matricize(
        ::FusionTensorFusionStyle, ft::AbstractArray, length_codomain::Val
    )
    first(blocklengths(axes(ft))) == unval(length_codomain) ||
        throw(ArgumentError("Invalid trivial biperm"))
    return FusionTensor(data_matrix(ft), (codomain_axis(ft),), (domain_axis(ft),))
end

function TensorAlgebra.unmatricize(
        ::FusionTensorFusionStyle,
        m::AbstractMatrix,
        codomain_axes::Tuple{Vararg{AbstractUnitRange}},
        domain_axes::Tuple{Vararg{AbstractUnitRange}},
    )
    return FusionTensor(data_matrix(m), codomain_axes, domain_axes)
end

function TensorAlgebra.bipermutedims(
        ft::FusionTensor,
        codomain_perm::Tuple{Vararg{Int}},
        domain_perm::Tuple{Vararg{Int}},
    )
    return permutedims(ft, permmortar((codomain_perm, domain_perm)))
end

function TensorAlgebra.bipermutedims!(
        a_dest::FusionTensor,
        a_src::FusionTensor,
        codomain_perm::Tuple{Vararg{Int}},
        domain_perm::Tuple{Vararg{Int}},
    )
    return permutedims!(a_dest, a_src, permmortar((codomain_perm, domain_perm)))
end

# TODO: Define custom broadcast rules for FusionTensors so that we can delete
# this method.
function TensorAlgebra.unmatricizeadd!(
        style::FusionTensorFusionStyle,
        a_dest::AbstractArray,
        a_dest_mat::AbstractMatrix,
        codomain_perm::Tuple{Vararg{Int}},
        domain_perm::Tuple{Vararg{Int}},
        α::Number, β::Number,
    )
    a12 = unmatricize(a_dest_mat, axes(a_dest), codomain_perm, domain_perm)
    data_matrix(a_dest) .= α .* data_matrix(a12) .+ β .* data_matrix(a_dest)
    return a_dest
end

const MATRIX_FUNCTIONS = [
    :exp,
    :cis,
    :log,
    :sqrt,
    :cbrt,
    :cos,
    :sin,
    :tan,
    :csc,
    :sec,
    :cot,
    :cosh,
    :sinh,
    :tanh,
    :csch,
    :sech,
    :coth,
    :acos,
    :asin,
    :atan,
    :acsc,
    :asec,
    :acot,
    :acosh,
    :asinh,
    :atanh,
    :acsch,
    :asech,
    :acoth,
]

for f in MATRIX_FUNCTIONS
    @eval begin
        function TensorAlgebra.$f(
                style::FusionTensorFusionStyle, a::AbstractArray,
                length_codomain::Val; kwargs...,
            )
            a_mat = matricize(style, a, length_codomain)
            biperm = trivialbiperm(length_codomain, Val(ndims(a)))
            permuted_axes = axes(a)[biperm]
            checkspaces_dual(codomain(permuted_axes), domain(permuted_axes))
            fa_mat = set_data_matrix(a_mat, Base.$f(data_matrix(a_mat); kwargs...))
            return unmatricize(style, fa_mat, permuted_axes)
        end
    end
end
