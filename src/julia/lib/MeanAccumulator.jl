## Object for computing means when elements come one by one.
mutable struct MeanAccumulator
    value::Float64
    n::Int
    function MeanAccumulator(value::AbstractFloat, n::Integer)
        if n < 0
            throw(ArgumentError("`n` must be > 0."));
        end
        if n == 0 && value != 0.0
            throw(ArgumentError("if `n == 0`, so must be `value`"));
        end
        new(value, n);
    end
end
function MeanAccumulator()
    MeanAccumulator(0.0, 0);
end

import Base.accumulate!
function accumulate!(ma::MeanAccumulator, x::AbstractFloat)
    n = ma.n + 1;
    ma.value = ((n - 1) * ma.value + x) / n;
    ma.n = n;
    nothing;
end

function reset!(ma::MeanAccumulator)
    ma.n = 0;
    ma.value = 0.0;
    nothing;
end

value(ma::MeanAccumulator) = ma.value;
