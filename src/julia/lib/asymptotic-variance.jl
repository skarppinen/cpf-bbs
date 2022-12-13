# The codes below are from https://github.com/awllee/MonteCarloMarkovKernels.jl
# src/batchMeans.jl
# src/spectralVariance.jl
# The code can be used to estimate asympotic variance from an MCMC chain.
# If z is a vector of numbers, you can estimate asymptotic variance simply by:
# estimateBM(z)
# estimateSV(z)

"""
MIT License

Copyright (c) 2018 Anthony Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

using Statistics

function estimateBM(xs::Vector{Float64}, b::Int64)
    lxs::Int64 = length(xs)
    a = floor(Int64, lxs/b)
    @assert a > 1
    n::Int64 = a*b
    start::Int64 = length(xs) - n
    overallMean::Float64 = 0.0
    for i = 1:n
        overallMean += xs[start+i]
    end
    overallMean /= n
    acc::Float64 = 0.0
    for i = 1:a
        batchAcc::Float64 = 0.0
        batchStart::Int64 = start + (i-1)*b
        for j = 1:b
            batchAcc += xs[batchStart + j]
        end
        tmp::Float64 = batchAcc/b
        tmp -= overallMean
        acc += tmp * tmp
    end
    return b/(a-1)*acc
end

## Basic batch means estimation of the asymptotic variance
function estimateBM(xs::Vector{Float64})
    return estimateBM(xs, floor(Int64, sqrt(length(xs))))
end

function Î³n(xs::Vector{Float64}, s::Int64)
    n::Int64 = length(xs)
    sa::Int64 = abs(s)
    xbar::Float64 = mean(xs)
    acc::Float64 = 0.0
    for i in 1:n-sa
        @inbounds acc += (xs[i]-xbar)*(xs[i+sa]-xbar)
    end
    return acc/n
end

function estimateSV(xs::Vector{Float64}, b::Int64, ws::Vector{Float64})
    v::Float64 = Î³n(xs, 0)
    for s in 1:b
        v += 2*ws[s]*Î³n(xs, s)
    end
    return v
end

function _wsSimpleTruncation(b::Int64)
    return ones(Float64, b)
end

function _wsBlackmanTukey(b::Int64, a::Float64)
    ws::Vector{Float64} = Vector{Float64}(b)
    for k in 1:b
        ws[k] = 1 - 2*a + 2*a*cos(Ï€*k/b)
    end
    return ws
end

function _wsTukeyHanning(b::Int64)
    return _wsBlackmanTukey(b, 0.25)
end

function _wsParzen(b::Int64, q::Int64)
    ws::Vector{Float64} = Vector{Float64}(undef, b)
    for k in 1:b
        ws[k] = 1 - (k/b)^q
    end
    return ws
end

function _wsModifiedBartlett(b::Int64)
    return _wsParzen(b, 1)
end

function estimateSV(xs::Vector{Float64}, b::Int64, name::Symbol)
    name == :ModifiedBartlett && return estimateSV(xs, b, _wsModifiedBartlett(b))
end

function estimateSV(xs::Vector{Float64}, name::Symbol)
    b::Int64 = floor(Int64, sqrt(length(xs)))
    name == :ModifiedBartlett && return estimateSV(xs, b, _wsModifiedBartlett(b))
end

estimateSV(xs::Vector{Float64}) = estimateSV(xs, :ModifiedBartlett)
