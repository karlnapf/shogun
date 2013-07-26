/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#ifdef HAVE_EIGEN3

#include <shogun/distributions/classical/GaussianDistribution.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CGaussianDistribution::CGaussianDistribution()
{
	init();
}

CGaussianDistribution::CGaussianDistribution(SGVector<float64_t> mean,
		SGMatrix<float64_t> cov,
		ECovarianceFactorization factorization, bool cov_is_factor)
{
	init();

	m_mean=mean;
	m_factorization=factorization;

	if (!cov_is_factor)
		compute_covariance_factorization(cov, factorization);
	else
		m_L=cov;
}

CGaussianDistribution::~CGaussianDistribution()
{

}

SGMatrix<float64_t> CGaussianDistribution::sample(int32_t num_samples)
{
	SG_NOTIMPLEMENTED;
	return SGMatrix<float64_t>();
}

SGVector<float64_t> CGaussianDistribution::log_pdf(SGMatrix<float64_t> samples)
{
	float64_t const_part=-0.5 * m_dimension * CMath::log(2 * CMath::PI);

	/* log-determinant */
	float64_t log_det_part=0;
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	switch (m_factorization)
	{
		case CF_CHOLESKY: case CF_CHOLESKY_PIVOT:
		{
			/* determinant is product of diagonal elements of triangular matrix */
			VectorXd diag=eigen_L.diagonal();
			log_det_part=-diag.array().log().sum();
			break;
		}
		case CF_SVD:
		{
			SG_NOTIMPLEMENTED;
			break;
		}
	}

	/* solve linear system (x-mu)^T Sigma (x-mu) for all given x */
	Map<MatrixXd> eigen_samples(samples.matrix, samples.num_rows, samples.num_cols);
	Map<VectorXd> eigen_mean(m_mean.vector, m_mean.vlen);
	MatrixXd centred=eigen_samples.colwise()-eigen_mean;
	SGVector<float64_t> quadratic_parts(m_dimension);
	Map<VectorXd> eigen_quadratic_parts(quadratic_parts.vector, quadratic_parts.vlen);
	switch (m_factorization)
	{
		case CF_CHOLESKY: case CF_CHOLESKY_PIVOT:
		{
			/* use triangular solver */
			eigen_quadratic_parts=centred.transpose()*
					(eigen_L.triangularView<Lower>().solve(centred));
			break;
		}
		case CF_SVD:
		{
			SG_NOTIMPLEMENTED;
			break;
		}
	}

	eigen_quadratic_parts.array()+=log_det_part+const_part;

	/* contains everything */
	return quadratic_parts;

}

void CGaussianDistribution::init()
{
	m_factorization=CF_CHOLESKY;

	SG_ADD(&m_mean, "mean", "Mean of the Gaussian.", MS_NOT_AVAILABLE);
	SG_ADD(&m_L, "L", "Lower factor of covariance matrix, "
			"depending on the factorization type.", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*)&m_factorization, "factorization", "Type of the "
			"factorization of the covariance matrix.", MS_NOT_AVAILABLE);
}

void CGaussianDistribution::compute_covariance_factorization(
		SGMatrix<float64_t> cov, ECovarianceFactorization factorization)
{
	Map<MatrixXd> eigen_cov(cov.matrix, cov.num_rows, cov.num_cols);
	m_L=SGMatrix<float64_t>(cov.num_rows, cov.num_cols);
	Map<MatrixXd> eigen_factor(m_L.matrix, m_L.num_rows, m_L.num_cols);

	switch (m_factorization)
	{
		case CF_CHOLESKY:
		{
			LLT<MatrixXd> llt(eigen_cov);
			if (llt.info()==NumericalIssue)
				SG_ERROR("Error computing Cholesky\n");

			eigen_factor=llt.matrixL();
			break;
		}

		case CF_CHOLESKY_PIVOT:
		{
			LDLT<MatrixXd> ldlt(eigen_cov);
			if (ldlt.info()==NumericalIssue)
					SG_ERROR("Error computing robust Cholesky (pivoting)\n");

			eigen_factor=ldlt.matrixL();
			break;
		}
		case CF_SVD:
		{
			JacobiSVD<MatrixXd> svd(eigen_cov);
			MatrixXd U=svd.matrixU();
			VectorXd s=svd.singularValues();

			/* square root of covariance using all eigenvectors */
			eigen_factor=U.array().colwise()*s.array();
		}
			break;
		default:
			SG_ERROR("Unknown factorization type: %d\n", m_factorization);
			break;
	}
}

#endif // HAVE_EIGEN3
