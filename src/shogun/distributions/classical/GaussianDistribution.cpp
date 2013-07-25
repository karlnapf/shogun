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
		ECovarianceFactorization factorization, bool cov_is_factor,
		int32_t low_rank_dimension)
{
	init();

	m_mean=mean;
	m_factorization=factorization;

	if (!cov_is_factor)
		compute_covariance_factorization(cov, factorization, low_rank_dimension);
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
	SG_NOTIMPLEMENTED;
	return SGVector<float64_t>();
}

void CGaussianDistribution::init()
{
	m_factorization=G_CHOLESKY;

	SG_ADD(&m_mean, "mean", "Mean of the Gaussian.", MS_NOT_AVAILABLE);
	SG_ADD(&m_L, "L", "Lower factor of covariance matrix, "
			"depending on the factorization type.", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*)&m_factorization, "factorization", "Type of the "
			"factorization of the covariance matrix.", MS_NOT_AVAILABLE);
}

void CGaussianDistribution::compute_covariance_factorization(
		SGMatrix<float64_t> cov, ECovarianceFactorization factorization,
		int32_t low_rank_dimension)
{
	Map<MatrixXd> eigen_cov(cov.matrix, cov.num_rows, cov.num_cols);
	m_L=SGMatrix<float64_t>(cov.num_rows, cov.num_cols);
	Map<MatrixXd> eigen_factor(m_L.matrix, m_L.num_rows, m_L.num_cols);

	switch (m_factorization)
	{
		case G_CHOLESKY:
		{
			LLT<MatrixXd> llt(eigen_cov);
			if (llt.info()==NumericalIssue)
				SG_ERROR("Error computing Cholesky\n");

			eigen_factor=llt.matrixL();
			break;
		}

		case G_CHOLESKY_PIVOT:
		{
			LDLT<MatrixXd> ldlt(eigen_cov);
			if (ldlt.info()==NumericalIssue)
					SG_ERROR("Error computing robust Cholesky (pivoting)\n");

			eigen_factor=ldlt.matrixL();
			break;
		}
		case G_SVD:
		{
			JacobiSVD<MatrixXd> svd(eigen_cov);
			MatrixXd U=svd.matrixU();
			VectorXd s=svd.singularValues();

			if (low_rank_dimension==0)
			{
				/* square root of covariance using all eigenvectors */
				eigen_factor=U.array().colwise()*s.array();
			}
			else
			{
				/* square root of covariance using a subset of eigenvectors */
				MatrixXd U_low_rank=U.block(0, 0, low_rank_dimension, U.cols());
				VectorXd s_low_rank=s.segment(0, low_rank_dimension);
				eigen_factor=U_low_rank.array().colwise()*s_low_rank.array();
			}
		}
			break;
		default:
			SG_ERROR("Unknown factorization type: %d\n", m_factorization);
			break;
	}
}

SGMatrix<float64_t> CGaussianDistribution::get_covariance_factor()
{
	return m_L;
}

#endif // HAVE_EIGEN3
