# This file implements the DotProductNode and corresponding sumProductRule! methods

using ForneyLab # v0.4.0

export DotProductNode
import ForneyLab.sumProductRule!

"""
Description:

    out = β⋅in

    in:  d-dimensional vector
    β:   d-dimensional vector
    out: scalar

           β
           |
      in   V   out
    ----->[⋅]----->

    f(in, β, out) =  δ(out - β⋅in)

Interfaces:

    1 i[:in], 2 i[:β], 3 i[:out]

Construction:

    DotProductNode(id=:my_node)
"""
type DotProductNode <: Node
    id::Symbol
    interfaces::Array{Interface,1}
    i::Dict{Symbol,Interface}

    function DotProductNode(; id=ForneyLab.generateNodeId(DotProductNode))
        self = new(id, Array(Interface, 3), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)

        for (id, name) in enumerate([:in, :β, :out])
            self.i[name] = self.interfaces[id] = Interface(self)
        end

        return self
    end
end

isDeterministic(::DotProductNode) = true


"""
DotProductNode:

             β
             | ↓ N{d}
             |
             V
    x ----->[⋅]-----> y
        →        →
       δ{d}      N
"""
function sumProductRule!{dims}( node::DotProductNode,
                                outbound_interface_index::Type{Val{3}},
                                y::Gaussian,
                                msg_1::Message{MvDelta{Float64,dims}},
                                msg_2::Message{MvGaussian{dims}},
                                ::Any)

    x = msg_1.payload.m # We'll call in x
    β = ensureParameters!(msg_2.payload, (:m, :V))

    y.m = (x' * β.m)[1,1]
    y.V = (x' * β.V * x)[1,1]
    y.xi = NaN; y.W = NaN

    return y
end


"""
DotProductNode:

             β
             | ↑ N{d}
             |
             V
    x ----->[⋅]-----> y
        →        ←
       δ{d}      N
"""
function sumProductRule!{dims}( node::DotProductNode,
                                outbound_interface_index::Type{Val{2}},
                                β::MvGaussian{dims},
                                msg_1::Message{MvDelta{Float64,dims}},
                                ::Any,
                                msg_3::Message{Gaussian})

    x = msg_1.payload.m # We'll call in x
    d = length(x)

    y = ensureParameters!(msg_3.payload, (:xi, :W))
    β.xi = x * y.xi
    β.W = x * y.W * x'
    invalidate!(β.m)
    invalidate!(β.V)

    return β
end