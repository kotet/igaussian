package igaussian

import (
	"math"
	"math/rand"
	"time"
)

type randomNumberGenerator interface {
	Float64() float64
}

type InverseGaussianDistribution struct {
	mu     float64
	lambda float64
	rng    randomNumberGenerator
}

func New(mu float64, lambda float64) *InverseGaussianDistribution {
	return &InverseGaussianDistribution{
		mu:     mu,
		lambda: lambda,
		rng:    rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func NewWithRNG(mu float64, lambda float64, rng randomNumberGenerator) *InverseGaussianDistribution {
	return &InverseGaussianDistribution{
		mu:     mu,
		lambda: lambda,
		rng:    rng,
	}
}

func (ig *InverseGaussianDistribution) Float64() float64 {
	return ig.InverseCDF(ig.rng.Float64())
}

func (ig *InverseGaussianDistribution) CDF(x float64) float64 {
	if 0 < x {
		return inverseGaussianDistributionCDF(x, ig.mu, ig.lambda)
	} else {
		return 0
	}
}

func (ig *InverseGaussianDistribution) PDF(x float64) float64 {
	if 0 < x {
		return inverseGaussianDistributionPDF(x, ig.mu, ig.lambda)
	} else {
		return 0
	}
}

func (ig *InverseGaussianDistribution) InverseCDF(y float64) float64 {
	if y <= 0 || 1 <= y {
		return math.NaN()
	}
	// newton's method
	x := 1.0
	for i := 0; i < 50; i++ {
		a := y - ig.CDF(x)
		b := -ig.PDF(x)
		if math.Abs(a/b) < 0.1 || math.IsInf(a/b, 0) || math.IsNaN(a/b) {
			break
		}
		x = x - a/b
	}

	// bisection method
	less := math.Max(0, x-0.25)
	more := math.Max(0, x-0.25) + 0.5
	for i := 0; i < 50; i++ {
		m := less + (more-less)/2
		if ig.CDF(m) < y {
			less = m
		} else {
			more = m
		}
	}
	return less
}

func standardNormalDistributionCDF(x float64) float64 {
	e := math.Erf(x / math.Sqrt(2))
	return (1 + e) / 2
}

func inverseGaussianDistributionCDF(x float64, mu float64, lambda float64) float64 {
	a := math.Sqrt(lambda/x) * ((x / mu) - 1)
	b := math.Exp(2 * lambda / mu)
	c := -math.Sqrt(lambda/x) * ((x / mu) + 1)
	return standardNormalDistributionCDF(a) + b*standardNormalDistributionCDF(c)
}

func inverseGaussianDistributionPDF(x float64, mu float64, lambda float64) float64 {
	a := math.Sqrt(lambda / (2 * math.Pi * x * x * x))
	b := math.Exp(-(lambda * (x - mu) * (x - mu)) / (2 * mu * mu * x))
	return a * b
}
