# Radial Basis

```@example
using HelFEM, Plots
pb = HelFEM.PolynomialBasis(:lip, 4)
xgrid = [0, 1, 2, 3]
b = HelFEM.RadialBasis(pb, xgrid, 100)
# Calculate points for plotting
rs, bs = HelFEM.quadraturepoints(b), HelFEM.basisvalues(b)
a = @animate for i in 1:length(b)
    plot(layout=(2,1), size=(900, 700), legend=false,
        ylabel = [raw"$b(r)$" raw"$r~b(r)$"]
    )
    plot!(rs, bs, c=:lightgray, subplot=1)
    plot!(rs, bs[:,i], c=:black, subplot=1)
    plot!(rs, bs .* rs, c=:lightgray, subplot=2)
    plot!(rs, bs[:,i] .* rs, c=:black, subplot=2)
    vline!([xgrid xgrid], ls=:dash, c=1)
end
gif(a, fps=1)
```
