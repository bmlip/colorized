########################################################################################
# Linear transition model factor node
########################################################################################
#
#                      | u
#                      |
#                      v
#           +----------------------+
#  -------->|p(z_new|z_old,u;A,b,Σ)|-------->
#    z_old  +----------------------+  z_new
#
#  The factor node captures the following linear model:
#
#  z_new = A * z_old + b * u + ε(Σ)
#
#  A:     known DxD matrix (parameter)
#  z_old: D-dimensional random vector (interface, accepts only multivariate Gaussians)
#  b:     known D-dimensional vector (parameter)
#  u:     random scalar (interface, accepts only univariate Gaussians)
#  ε(Σ):  white Gaussian noise term ε ~ N(0,Σ) (parameter Σ)
#
########################################################################################

using ForneyLab # requires ForneyLab v0.4

type LinearTransitionModelNode <: Node
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    internal_factorgraph::FactorGraph # internal graph
    z_old::TerminalNode               # node in internal graph
    z_new::TerminalNode               # node in internal graph
    u::TerminalNode                   # node in internal graph
    algo_state_forward::SumProduct    # sum-product algorithm for message towards z_new
    algo_state_backward::SumProduct   # sum-product algorithm for message towards z_old

    function LinearTransitionModelNode(; id = ForneyLab.generateNodeId(LinearTransitionModelNode),
                                         A::Matrix{Float64} = [NaN]',
                                         b::Vector{Float64} = [NaN],
                                         Σ::Matrix{Float64} = [NaN]')
        # Sanity checks
        D = size(A,1)
        (!any(isnan(A)) && (size(A,2)==D)) || error("Invalid parameter A")
        (!any(isnan(b)) && (length(b)==D)) || error("Invalid parameter b")
        (!any(isnan(Σ)) && (size(Σ)==size(A)) && isposdef(Σ)) || error("Invalid parameter Σ")

        # Create node and interfaces
        self = new(id, Vector{Interface}(3), Dict{Int,Interface}())
        self.i[:z_old] = self.interfaces[1] = Interface(self)
        self.i[:u]     = self.interfaces[2] = Interface(self)
        self.i[:z_new] = self.interfaces[3] = Interface(self)
        addNode!(currentGraph(), self)

        # Build internal factor graph
        outer_graph = currentGraph()
        self.internal_factorgraph = FactorGraph()
        self.z_old = TerminalNode(vague(MvGaussian{D}), id=:z_old)
        self.z_new = TerminalNode(vague(MvGaussian{D}), id=:z_new)
        self.u = TerminalNode(vague(MvGaussian{D}), id=:u)
        g_A = GainNode(gain=A, id=:A)
        g_B = GainNode(gain=diagm(b), id=:B)
        add_input = AdditionNode()
        process_noise = GaussianNode(V=Σ, id=:noise)
        Edge(self.z_old, g_A.i[:in])
        Edge(g_A.i[:out], add_input.i[:in1])
        Edge(self.u, g_B.i[:in])
        Edge(g_B.i[:out], add_input.i[:in2])
        Edge(add_input.i[:out], process_noise.i[:mean])
        Edge(process_noise.i[:out], self.z_new)

        # Build internal algorithms
        self.algo_state_forward = SumProduct(process_noise.i[:out])
        prepare!(self.algo_state_forward) # initialize messages in internal graph
        self.algo_state_backward = SumProduct(g_A.i[:in])
        prepare!(self.algo_state_backward) # initialize messages in internal graph
        # Exit internal graph
        setCurrentGraph(outer_graph)

        return self
    end
end

ForneyLab.isDeterministic(::LinearTransitionModelNode) = false
import ForneyLab.sumProductRule!

# Message towards z_old
function sumProductRule!{D}(node::LinearTransitionModelNode,
                            outbound_interface_index::Type{Val{1}},
                            outbound_dist::MvGaussian{D},
                            msg_1::Any,
                            msg_2::Message{Gaussian},
                            msg_3::Message{MvGaussian{D}})
    outer_graph = currentGraph()
    setCurrentGraph(node.internal_factorgraph)
    node.u.value.m = mean(msg_2.payload)*ones(D)
    node.u.value.V = var(msg_2.payload)*eye(D)
    node.z_new.value = msg_3.payload
    result = execute(node.algo_state_backward)
    for field in fieldnames(outbound_dist)
        setfield!(outbound_dist, field, getfield(result.payload, field))
    end
    setCurrentGraph(outer_graph)

    return outbound_dist
end

# Message towards z_new
function sumProductRule!{D}(node::LinearTransitionModelNode,
                            outbound_interface_index::Type{Val{3}},
                            outbound_dist::MvGaussian{D},
                            msg_1::Message{MvGaussian{D}},
                            msg_2::Message{Gaussian},
                            msg_3::Any)
    outer_graph = currentGraph()
    setCurrentGraph(node.internal_factorgraph)
    node.u.value.m = mean(msg_2.payload)*ones(D)
    node.u.value.V = var(msg_2.payload)*eye(D)
    node.z_old.value = msg_1.payload
    result = execute(node.algo_state_forward)
    for field in fieldnames(outbound_dist)
        setfield!(outbound_dist, field, getfield(result.payload, field))
    end
    setCurrentGraph(outer_graph)

    return outbound_dist
end