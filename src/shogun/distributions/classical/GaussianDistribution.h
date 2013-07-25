/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */
#ifdef HAVE_EIGEN3

#ifndef GAUSSIANDISTRIBUTION_H
#define GAUSSIANDISTRIBUTION_H

#include <shogun/distributions/classical/ProbabilityDistribution.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

/** @brief Dense version of the well-known Gaussian probability distribution,
 * defined as
 * \f[
 * \mathcal{N}_x(\mu,\Sigma)=
 * \frac{1}{\sqrt{|2\pi\Sigma|}}
 * \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu\right)
 * \f]
 *
 * The implementation offers various techniques for representing the covariance
 * matrix \f$\Sigma \f$, such as Cholesky factorisation, SVD-decomposition,
 * low-rank versions, and setting the precision matrix \f$\Sigma^{-1} \f$
 * directly.
 *
 * All factorizations store a matrix m_L, such that the covariance can be
 * computed as LL^T.
 *
 * For Cholesky factorization, the lower factor \f$\Sigma=LL^T\f$ is computed
 * with a eigen3's LDLT (robust Cholesky with pivoting).
 *
 * For SVD factorization U diag(s) V^T, the factor U*s (column-wise product) is
 * stored. SVD factorization may be done in a low rank version in the sense that
 * only the first few Eigenvectors are used for the covariance factor.
 */

enum ECovarianceFactorization
{
	G_CHOLESKY, G_CHOLESKY_PIVOT, G_SVD
};

class CGaussianDistribution: public CProbabilityDistribution
{
public:
	/** Default constructor */
	CGaussianDistribution();

	/** Constructor for which takes Gaussian mean and its covariance matrix.
	 * It is also possible to pass a precomputed matrix factor of the specified
	 * form. In this case, the factorization is not explicitly computed.
	 *
	 * @param mean mean of the Gaussian
	 * @param cov covariance of the Gaussian, or covariance factor
	 * @param factorization factorization type of covariance matrix (default is
	 * Cholesky)
	 * @param cov_is_factor whether cov is a factor of the covariance or not
	 * (default is false). If false, the factorization is explicitly computed
	 * @param low_rank_dimension if a SVD factorization is used, one can
	 * optionally specify the number of dimensions to produce a low-rank
	 * approximation of the covariance. Ignored if 0 (default is 0).
	 *  */
	CGaussianDistribution(SGVector<float64_t> mean, SGMatrix<float64_t> cov,
			ECovarianceFactorization factorization=G_CHOLESKY,
			bool cov_is_factor=false, int32_t low_rank_dimension=0);

	/** Destructor */
	virtual ~CGaussianDistribution();

	/** Samples from the distribution multiple times
	 *
	 * @param num_samples number of samples to generate
	 * @return matrix with samples (column vectors)
	 */
	virtual SGMatrix<float64_t> sample(int32_t num_samples);

	/** Computes the log-pdf for all provided samples
	 *
	 * @param samples samples to compute log-pdf of (column vectors)
	 * @return vector with log-pdfs of given samples
	 */
	virtual SGVector<float64_t> log_pdf(SGMatrix<float64_t> samples);

	/** @return name of the SGSerializable */
	virtual const char* get_name() const
	{
		return "GaussianDistribution";
	}

private:

	/** Initialses and registers parameters */
	void init();

	/** Computes and stores the factorization of the covariance matrix */
	void compute_covariance_factorization(SGMatrix<float64_t> cov,
			ECovarianceFactorization factorization,
			int32_t low_rank_dimension);

protected:
	/** Mean */
	SGVector<float64_t> m_mean;

	/** Lower factor of covariance matrix (depends on factorization type).
	 * Covariance (approximation) is given by LL^T */
	SGMatrix<float64_t> m_L;

	/** Type of the factorization of the covariance matrix */
	ECovarianceFactorization m_factorization;
};

}

#endif // GAUSSIANDISTRIBUTION_H
#endif // HAVE_EIGEN3
