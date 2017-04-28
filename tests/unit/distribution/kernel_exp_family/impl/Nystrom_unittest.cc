/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Heiko Strathmann
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/distributions/kernel_exp_family/impl/Full.h>
#include <shogun/distributions/kernel_exp_family/impl/Nystrom.h>
#include <shogun/distributions/kernel_exp_family/impl/kernel/Gaussian.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/base/some.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace kernel_exp_family_impl;

using namespace shogun;
using namespace Eigen;

//TEST(kernel_exp_family_impl_Nystrom, compute_xi_norm_2_all_inds_equals_exact)
//{
//	index_t N=5;
//	index_t D=3;
//	SGMatrix<float64_t> X(D,N);
//	for (auto i=0; i<N*D; i++)
//		X.matrix[i]=CMath::randn_float();
//
//	float64_t sigma = 2;
//	float64_t lambda = 1;
//	Nystrom est(X, new kernel::Gaussian(sigma), lambda, N*D);
//	Full est_full(X, new kernel::Gaussian(sigma), lambda);
//
//	// compare against full version
//	EXPECT_NEAR(est.compute_xi_norm_2(), est_full.compute_xi_norm_2(), 1e-12);
//}

//TEST(kernel_exp_family_impl_Nystrom, compute_h_all_inds_equals_full)
//{
//	index_t N=5;
//	index_t D=3;
//	SGMatrix<float64_t> X(D,N);
//	for (auto i=0; i<N*D; i++)
//		X.matrix[i]=CMath::randn_float();
//
//	float64_t sigma = 2;
//	float64_t lambda = 1;
//	Nystrom est(X, N*D, new kernel::Gaussian(sigma), lambda);
//	Full est_full(X, new kernel::Gaussian(sigma), lambda);
//
//	// compare against full version
//	auto h = est.compute_h();
//	auto h_full = est.compute_h();
//
//	ASSERT_EQ(h.vlen, h_full.vlen);
//
//	for (auto i=0; i<N*D; i++)
//		EXPECT_NEAR(h[i], h_full[i], 1e-12);
//}

//TEST(kernel_exp_family_impl_Nystrom, compute_h_half_inds_equals_subsampled_full)
//{
//	index_t N=5;
//	index_t D=3;
//	SGMatrix<float64_t> X(D,N);
//	for (auto i=0; i<N*D; i++)
//		X.matrix[i]=CMath::randn_float();
//
//	float64_t sigma = 2;
//	float64_t lambda = 1;
//
//	index_t m=5;
//	SGVector<index_t> temp(N*D);
//	temp.range_fill();
//	CMath::permute(temp);
//	SGVector<index_t> inds(m);
//	memcpy(inds.vector, temp.vector, sizeof(index_t)*m);
//
//	Nystrom est(X, inds, new kernel::Gaussian(sigma), lambda);
//	Full est_full(X, new kernel::Gaussian(sigma), lambda);
//
//	// compare against full version
//	auto h = est.compute_h();
//	auto h_full = est_full.compute_h();
//
//	ASSERT_EQ(h.vlen, m);
//	ASSERT_EQ(h_full.vlen, N*D);
//	for (auto i=0; i<m; i++)
//		EXPECT_NEAR(h[i], h_full[inds[i]], 1e-12);
//}

TEST(kernel_exp_family_impl_Nystrom, fit)
{
	index_t N=5;
	index_t D=3;
	index_t m=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 1;
	Nystrom est(X, m, new kernel::Gaussian(sigma), lambda);
	est.fit();

	auto beta=est.get_beta();
	ASSERT_EQ(beta.vlen, m*D);
	ASSERT(beta.vector);
}

TEST(kernel_exp_family_impl_Nystrom, log_pdf_almost_all_inds_close_exact)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 1;
	auto m=N-1;
	Nystrom est_nystrom(X, m, new kernel::Gaussian(sigma), lambda);
	Full est(X, new kernel::Gaussian(sigma), lambda);

	est_nystrom.fit();
	est.fit();

	SGVector<float64_t> x(D);
	for (auto i=0; i<D; i++)
		x[i]=CMath::randn_float();

	est.set_data(x);
	est_nystrom.set_data(x);
	auto log_pdf = est.log_pdf(0);
	auto log_pdf_nystrom = est_nystrom.log_pdf(0);

	EXPECT_NEAR(log_pdf, log_pdf_nystrom, 0.1);
}

TEST(kernel_exp_family_impl_Nystrom, grad_all_inds_equals_exact)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 1;
	auto m=N;
	Nystrom est_nystrom(X, m, new kernel::Gaussian(sigma), lambda);
	Full est(X, new kernel::Gaussian(sigma), lambda);
	est_nystrom.fit();
	est.fit();

	SGVector<float64_t> x(D);
	for (auto i=0; i<D; i++)
		x[i]=CMath::randn_float();

	est.set_data(x);
	est_nystrom.set_data(x);
	auto grad = est.grad(0);
	auto grad_nystrom = est_nystrom.grad(0);

	for (auto i=0; i<D; i++)
		EXPECT_NEAR(grad[i], grad_nystrom[i], 1e-8);
}

TEST(kernel_exp_family_impl_Nystrom, grad_almost_all_inds_close_exact)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 1;
	auto m=N-1;
	Nystrom est_nystrom(X, m, new kernel::Gaussian(sigma), lambda);
	Full est(X, new kernel::Gaussian(sigma), lambda);
	est_nystrom.fit();
	est.fit();

	SGVector<float64_t> x(D);
	for (auto i=0; i<D; i++)
		x[i]=CMath::randn_float();

	est.set_data(x);
	est_nystrom.set_data(x);
	auto grad = est.grad(0);
	auto grad_nystrom = est_nystrom.grad(0);

	for (auto i=0; i<D; i++)
		EXPECT_NEAR(grad[i], grad_nystrom[i], 0.3);
}

//TEST(kernel_exp_family_impl_Nystrom, idx_to_ai)
//{
//	index_t D=3;
//
//	index_t idx=0;
//	auto ai=Nystrom::idx_to_ai(idx, D);
//	EXPECT_EQ(ai.first, 0);
//	EXPECT_EQ(ai.second, 0);
//
//	idx=1;
//	ai=Nystrom::idx_to_ai(idx, D);
//	EXPECT_EQ(ai.first, 0);
//	EXPECT_EQ(ai.second, 1);
//
//	idx=2;
//	ai=Nystrom::idx_to_ai(idx, D);
//	EXPECT_EQ(ai.first, 0);
//	EXPECT_EQ(ai.second, 2);
//
//	idx=3;
//	ai=Nystrom::idx_to_ai(idx, D);
//	EXPECT_EQ(ai.first, 1);
//	EXPECT_EQ(ai.second, 0);
//
//	idx=4;
//	ai=Nystrom::idx_to_ai(idx, D);
//	EXPECT_EQ(ai.first, 1);
//	EXPECT_EQ(ai.second, 1);
//}

TEST(kernel_exp_family_impl_Nystrom, fit_all_inds_equals_exact)
{
	index_t N=5;
	index_t D=3;
	auto ND=N*D;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 1;
	Nystrom est_nystrom(X, N, new kernel::Gaussian(sigma), lambda);
	Full est(X, new kernel::Gaussian(sigma), lambda);

	est.fit();
	est_nystrom.fit();

	// compare against full version
	auto result_nystrom=est_nystrom.get_beta();
	auto result=est.get_beta();

	ASSERT_EQ(result.vlen, ND);
	ASSERT_EQ(result_nystrom.vlen, ND);
	ASSERT(result.vector);
	ASSERT(result_nystrom.vector);

	for (auto i=0; i<ND; i++)
		EXPECT_NEAR(result[i], result_nystrom[i], 1e-10);
}

TEST(kernel_exp_family_impl_Nystrom, fit_half_inds_shape)
{
	index_t N=5;
	index_t D=3;
	index_t m=N/2;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 1;
	Nystrom est(X, m, new kernel::Gaussian(sigma), lambda);
	est.fit();

	auto beta=est.get_beta();
	ASSERT_EQ(beta.vlen, m*D);
	ASSERT(beta.vector);
}

TEST(kernel_exp_family_impl_Nystrom, pinv_self_adjoint)
{
	index_t N=3;
	index_t D=2;
	SGMatrix<float64_t> X(D,N);
	X(0,0)=0;
	X(1,0)=1;
	X(0,1)=2;
	X(1,1)=4;
	X(0,2)=3;
	X(1,2)=1;

	SGMatrix<float64_t> S(D, D);

	auto eigen_X = Map<MatrixXd>(X.matrix, D, N);
	auto eigen_S = Map<MatrixXd>(S.matrix, D, D);
	eigen_S = eigen_X*eigen_X.transpose();

	auto pinv = Nystrom::pinv_self_adjoint(S);

	ASSERT_EQ(pinv.num_rows, 2);
	ASSERT_EQ(pinv.num_cols, 2);

	// from numpy.linalg.pinv
	float64_t reference[] = {0.15929204, -0.09734513, -0.09734513,  0.11504425};

	for (auto i=0; i<pinv.num_rows*pinv.num_cols; i++)
		EXPECT_NEAR(pinv[i], reference[i], 1e-8);
}

TEST(kernel_exp_family_impl_Nystrom, log_pdf_all_inds_equals_exact)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 1;
	Nystrom est_nystrom(X, N, new kernel::Gaussian(sigma), lambda);
	Full est(X, new kernel::Gaussian(sigma), lambda);
	est_nystrom.fit();
	est.fit();

	SGVector<float64_t> x(D);
	for (auto i=0; i<D; i++)
		x[i]=CMath::randn_float();

	est.set_data(x);
	est_nystrom.set_data(x);
	auto log_pdf = est.log_pdf(0);
	auto log_pdf_nystrom = est_nystrom.log_pdf(0);

	EXPECT_NEAR(log_pdf, log_pdf_nystrom, 1e-8);
}

TEST(kernel_exp_family_impl_Nystrom, hessian_all_inds_equals_exact)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 1;
	auto m=N;
	Nystrom est_nystrom(X, m, new kernel::Gaussian(sigma), lambda);
	Full est(X, new kernel::Gaussian(sigma), lambda);
	est_nystrom.fit();
	est.fit();

	SGVector<float64_t> x(D);
	for (auto i=0; i<D; i++)
		x[i]=CMath::randn_float();

	est.set_data(x);
	est_nystrom.set_data(x);
	auto hessian = est.hessian(0);
	auto hessian_nystrom = est_nystrom.hessian(0);

	for (auto i=0; i<D*D; i++)
		EXPECT_NEAR(hessian.matrix[i], hessian_nystrom.matrix[i], 1e-8);
}

TEST(kernel_exp_family_impl_Nystrom, hessian_almost_all_inds_execute)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 1;
	auto m=N*.6;
	Nystrom est_nystrom(X, m, new kernel::Gaussian(sigma), lambda);
	Full est(X, new kernel::Gaussian(sigma), lambda);
	est_nystrom.fit();
	est.fit();

	SGVector<float64_t> x(D);
	for (auto i=0; i<D; i++)
		x[i]=CMath::randn_float();

	est.set_data(x);
	est_nystrom.set_data(x);
	auto hessian = est.hessian(0);
	auto hessian_nystrom = est_nystrom.hessian(0);
}

TEST(kernel_exp_family_impl_Nystrom, hessian_diag_equals_hessian)
{
	index_t N=5;
	index_t D=3;
	SGMatrix<float64_t> X(D,N);
	for (auto i=0; i<N*D; i++)
		X.matrix[i]=CMath::randn_float();

	float64_t sigma = 2;
	float64_t lambda = 2;
	auto m=N*.6;
	auto kernel = new kernel::Gaussian(sigma);
	Nystrom est(X, m, kernel, lambda);
	est.fit();

	SGVector<float64_t> x(D);
	x[0] = CMath::randn_float();
	x[1] = CMath::randn_float();

	est.set_data(x);
	auto hessian = est.hessian(0);
	auto hessian_diag = est.hessian_diag(0);

	for (auto i=0; i<D; i++)
		EXPECT_NEAR(hessian_diag[i], hessian(i,i), 1e-8);
}
