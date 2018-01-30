# This file implements the DotProductNode and corresponding sum-product rules

using ForneyLab # v0.7.0

import ForneyLab: DeltaFactor, Univariate, Multivariate, generateId, associate!, addNode!, ensureParameters!
export DotProduct

"""
Description:

    out = β⋅in

    in1: d-dimensional vector
    in2: d-dimensional vector
    out: scalar

           in2
           |
      in1  V   out
    ----->[⋅]----->

    f(in1, in2, out) =  δ(out - in1'*in2)

Interfaces:

    1 i[:in1], 2 i[:in2], 3 i[:out]

Construction:

    DotProduct(out, in1, in2, id=:my_node)
"""
type DotProduct <: DeltaFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function DotProduct(out::Variable, in1::Variable, in2::Variable; id=generateId(DotProduct))
        self = new(id, Array{Interface}(3), Dict{Int,Interface}())
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:in1] = self.interfaces[2] = associate!(Interface(self), in1)
        self.i[:in2] = self.interfaces[3] = associate!(Interface(self), in2)

        return self
    end
end

slug(::Type{DotProduct}) = "dot"


# Update rule definitions

#             β
#             | ↓ N{d}
#             |
#             V
#    x ----->[⋅]-----> y
#        →        →
#       δ{d}      N

@sumProductRule(:node_type     => DotProduct,
                :outbound_type => Message{Gaussian},
                :inbound_types => (Void, Message{PointMass}, Message{Gaussian}),
                :name          => SPDotProductOutVPG)

function ruleSPDotProductOutVPG(msg_out::Void,
                            msg_in1::Message{PointMass, Multivariate},
                            msg_in2::Message{Gaussian, Multivariate})
    x = msg_in1.dist.params[:m]
    β = ensureParameters!(msg_in2.dist, (:m, :V))
    out_m = x' * β.m
    out_v = x' * β.V * x

    return Message(Multivariate, Gaussian, m=out_m, v=out_v)

end

#             β
#             | ↓ δ{d}
#             |
#             V
#    x ----->[⋅]-----> y
#        →        →
#       N{d}      N

@sumProductRule(:node_type     => DotProduct,
                :outbound_type => Message{Gaussian},
                :inbound_types => (Void, Message{Gaussian}, Message{PointMass}),
                :name          => SPDotProductOutVGP)

function ruleSPDotProductOutVGP(msg_out::Void,
                            msg_in1::Message{Gaussian, Multivariate},
                            msg_in2::Message{PointMass, Multivariate})
    SPDotProductOutVGP(msg_out, msg_in2, msg_in1)
end

#             β
#             | ↑ N{d}
#             |
#             V
#    x ----->[⋅]-----> y
#        →        ←
#       δ{d}      N

@sumProductRule(:node_type     => DotProduct,
                :outbound_type => Message{Gaussian},
                :inbound_types => (Message{Gaussian}, Message{PointMass}, Void),
                :name          => SPDotProductIn2GPV)

function ruleSPDotProductIn2GPV(msg_out::Message{Gaussian, Univariate},
                            msg_in1::Message{PointMass, Multivariate},
                            msg_in2::Void)
    x = msg_in1.dist.params[:m] # We'll call in1 x
    d = length(x)

    y = ensureParameters!(msg_out.dist, (:xi, :w))
    xi = x * y.params[:xi]
    w = x * y.params[:w] * x'

    return Message(Multivariate, Gaussian, xi=xi, w=w)
end

#             β
#             | ↓ δ{d}
#             |
#             V
#    x ----->[⋅]-----> y
#        ←        ←
#       N{d}      N

@sumProductRule(:node_type     => DotProduct,
                :outbound_type => Message{Gaussian},
                :inbound_types => (Message{Gaussian}, Void, Message{PointMass}),
                :name          => SPDotProductIn1GVP)

function ruleSPDotProductIn1GVP(msg_out::Message{Gaussian, Univariate},
                            msg_in1::Void,
                            msg_in2::Message{PointMass, Multivariate})
    SPDotProductIn2GPV(msg_out, msg_in2, msg_in1)
end
