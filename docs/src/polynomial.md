# Polynomial bases

```@example
using HelFEM, Plots
let xs = range(-1, 1, length=2001)
    a = @animate for nnodes = 2:15
        b = HelFEM.PolynomialBasis(:legendre, nnodes)
        ys = b(xs)
        plot(xs, ys, label=false, title="Legendre, nnodes=$nnodes")
    end
    gif(a, fps=0.5)
end
```

```@example
using HelFEM, Plots
let xs = range(-1, 1, length=2001)
    a = @animate for nnodes = 2:15
        b = HelFEM.PolynomialBasis(:hermite, nnodes)
        ys = b(xs)
        plot(xs, ys, label=false, title="Hermite, nnodes=$nnodes")
    end
    gif(a, fps=0.5)
end
```

```@example
using HelFEM, Plots
let xs = range(-1, 1, length=2001)
    a = @animate for nnodes = 2:15
        b = HelFEM.PolynomialBasis(:lip, nnodes)
        ys = b(xs)
        plot(xs, ys, label=false, title="LIP, nnodes=$nnodes")
    end
    gif(a, fps=0.5)
end
```

## Reference

```@docs
PolynomialBasis
```
