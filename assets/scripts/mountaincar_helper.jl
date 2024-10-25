import HypergeometricFunctions: _₂F₁

function create_physics(; engine_force_limit = 0.04, friction_coefficient = 0.1)
    # Engine force as function of action
    Fa = (a::Real) -> engine_force_limit * tanh(a) 

    # Friction force as function of velocity
    Ff = (y_dot::Real) -> -friction_coefficient * y_dot 
    
    # Gravitational force (horizontal component) as function of position
    Fg = (y::Real) -> begin
        if y < 0
            0.05*(-2*y - 1)
        else
            0.05*(-(1 + 5*y^2)^(-0.5) - (y^2)*(1 + 5*y^2)^(-3/2) - (y^4)/16)
        end
    end
    
    # The height of the landscape as a function of the horizontal coordinate
    height = (x::Float64) -> begin
        if x < 0
            h = x^2 + x
        else
            h = x * _₂F₁(0.5,0.5,1.5, -5*x^2) + x^3 * _₂F₁(1.5, 1.5, 2.5, -5*x^2) / 3 + x^5 / 80
        end
        return 0.05*h
    end

    return (Fa, Ff, Fg,height)
end;

function create_world(; Fg, Ff, Fa, initial_position = -0.5, initial_velocity = 0.0)

    y_t_min = initial_position
    y_dot_t_min = initial_velocity
    
    y_t = y_t_min
    y_dot_t = y_dot_t_min
    
    execute = (a_t::Float64) -> begin
        # Compute next state
        y_dot_t = y_dot_t_min + Fg(y_t_min) + Ff(y_dot_t_min) + Fa(a_t)
        y_t = y_t_min + y_dot_t
    
        # Reset state for next step
        y_t_min = y_t
        y_dot_t_min = y_dot_t
    end
    
    observe = () -> begin 
        return [y_t, y_dot_t]
    end
        
    return (execute, observe)
end

function plot_car(y, target; title_plot="trial", fps=12)

    function height(x::Float64)
        if x < 0
            h = x^2 + x
        else
            h = x * _₂F₁(0.5,0.5,1.5, -5*x^2) + x^3 * _₂F₁(1.5, 1.5, 2.5, -5*x^2) / 3 + x^5 / 80
        end
        return 0.05*h
    end

    valley_x = range(-1.3, 1, length=400)
    valley_y = [ height(xs) for xs in valley_x ];

    animation_car = @animate for i in eachindex(y)
        plot(valley_x, valley_y, title = "$title_plot, t=$i", label = "Landscape", color = "black", size = (800, 400))
        scatter!([target[1]], [height(target[1])], label="goal", markersize= 15) 
        scatter!([y[i][1]], [height(y[i][1])], label="car", markersize= 15)  
    end

    file_name = "./ai_agent/ " * title_plot * ".gif"
    gif(animation_car, file_name, fps = fps, show_msg = false);

end

function dzdt(z, a)
    fc = - 0.1 # friction coefficient 
    fl = 0.04 # engine force limit
    function Fg(y::Real) # Gravitational force
        
        if y < 0
            f = 0.05*(-2*y - 1)
        else
            f = 0.05*(-(1 + 5*y^2)^(-0.5) - (y^2)*(1 + 5*y^2)^(-3/2) - (y^4)/16)
        end
        
        return f
    end

    θ̇ = z[2] + Fg(z[1]) + fc * z[2] + fl * tanh(a[1])
    θ = z[1] + θ̇
    z_tp1 = [θ, θ̇ ]
    return z_tp1
end