module CWorldMCTSVisualize

using Plots, ContinuumWorld, POMDPToolbox, MCTS

export CWorldMCTSVis

struct CWorldMCTSVis
    mdp::CWorld
    s::Vec2
    actions::Vector{Vector{Vec2}}
    next_states::Vector{Vector{Vec2}}
    ns::Vector{Int}
    radii::Vector{Vector{Float64}}
end
CWorldMCTSVis(mdp::CWorld, s::Vec2=Vec2(0,0)) = CWorldMCTSVis(mdp, s, Vec2[], Vec2[], Int[], Float64[])

function circle(x,y,r)
    c = Shape(Plots.partialcircle(0,2π,20,r))
    translate!(c, x, y)
    c
end

@recipe function f(c::CircularRegion)
    @series begin
        plot(circle(c.center[1], c.center[2], c.radius))
    end
end

@recipe function f(vis::CWorldMCTSVis, h::SimHistory)
    mdp = vis.mdp
    xlim := mdp.xlim
    ylim := mdp.ylim
    aspect_ratio := :equal
    @series begin
        label := "reward region"
        [circle(c.center[1], c.center[2], c.radius) for c in mdp.reward_regions]
    end
    @series begin
        label := "path"
        color := :blue
        x = [s[1] for s in state_hist(h)[1:end-1]]
        y = [s[2] for s in state_hist(h)[1:end-1]]
        x, y
    end
    @series begin
        label := "current position"
        marker := :circle
        seriestype := :scatter
        color := :red
        x,y = state_hist(h)[end-1]
        [x], [y]
    end
end

@recipe function f(vis::CWorldMCTSVis, h::SimHistory, index::Int)
    mdp = vis.mdp
    xlim := mdp.xlim
    ylim := mdp.ylim
    aspect_ratio := :equal
    @series begin
        label := "actions"
        alpha := 0.25
        [circle(x,y,r) for ((x,y),r) in zip(vis.next_states[index], vis.radii[index])] 
    end
    @series begin
        label := "reward region"
        [circle(c.center[1], c.center[2], c.radius) for c in mdp.reward_regions]
    end
    @series begin
        label := "path"
        color := :blue
        x = [s[1] for s in state_hist(h)[1:end-1]]
        y = [s[2] for s in state_hist(h)[1:end-1]]
        x, y
    end
    @series begin
        label := "current position"
        marker := :circle
        seriestype := :scatter
        color := :red
        x,y = state_hist(h)[end-1]
        [x], [y]
    end
end

function MCTS.notify_listener(vis::CWorldMCTSVis,dsb::DSBPlanner,s,a,sp,r,snode,sanode,spnode)
    if s == vis.s
        tree = get(dsb.tree)
        sol = dsb.solver
        actions = [tree.a_labels[c] for c in tree.children[snode]]
        if isempty(vis.actions) || (actions != vis.actions[end])  #only push the deltas
            push!(vis.actions, actions)
            push!(vis.next_states, [s+action for action in actions])
            push!(vis.ns, tree.total_n[snode])
            push!(vis.radii, [sol.r0_action/tree.total_n[snode]^sol.lambda_action for action in actions])
        end
    end
end
function MCTS.notify_listener(vis::CWorldMCTSVis,asb::ASBPlanner,s,a,sp,r,snode,sanode,spnode)
    if s == vis.s
        tree = get(asb.tree)
        sol = asb.solver
        actions = [tree.a_labels[c] for c in tree.children[snode]]
        if isempty(vis.actions) || (actions != vis.actions[end])  #only push the deltas
            push!(vis.actions, actions)
            push!(vis.next_states, [s+action for action in actions])
            push!(vis.ns, tree.total_n[snode])
            push!(vis.radii, [tree.a_radius[x] for x in tree.children[snode]])
        end
    end
end

end # module
