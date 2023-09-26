import HypergeometricFunctions: _₂F₁

function plot_car(y, title_plot)

    target = [0.5, 0.0]

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


    animation_car = @animate for i in 1:N
    plot(valley_x, valley_y, title = title_plot , label = "Landscape", color = "black", size = (800, 400))
    scatter!([target[1]], [height(target[1])], label="goal", markersize= 15) 
    scatter!([y[i][1]], [height(y[i][1])], label="car", markersize= 15)  
    end

    file_name = "./ai_agent/ " * title_plot * ".gif"
    gif(animation_car, file_name, fps = 24, show_msg = false);

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