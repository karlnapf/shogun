/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Heiko Strathmann, Thoralf Klein
 */

#include <shogun/base/init.h>
#include <shogun/base/range.h>
#include <shogun/classifier/mkl/MKLClassification.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/lib/parameter_observers/ParameterObserverCV.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

void gen_rand_data(SGVector<float64_t> lab, SGMatrix<float64_t> feat,
		float64_t dist)
{
	index_t dims=feat.num_rows;
	index_t num=lab.vlen;

	for (int32_t i=0; i<num; i++)
	{
		if (i<num/2)
		{
			lab[i]=-1.0;

			for (int32_t j=0; j<dims; j++)
				feat(j, i)=CMath::random(0.0, 1.0)+dist;
		}
		else
		{
			lab[i]=1.0;

			for (int32_t j=0; j<dims; j++)
				feat(j, i)=CMath::random(0.0, 1.0)-dist;
		}
	}
	lab.display_vector("lab");
	feat.display_matrix("feat");
}

SGMatrix<float64_t> calculate_weights(
    CParameterObserverCV& obs, int32_t folds, int32_t run, int32_t len)
{
	int32_t column = 0;
	SGMatrix<float64_t> weights(len, folds * run);
	for (auto o : range(obs.get_num_observations()))
	{
		auto obs_storage = obs.get_observation(o);
		for (auto i : range(obs_storage->get<index_t>("num_folds")))
		{
			auto fold = obs_storage->get("folds", i);
			CMKLClassification* machine =
			    (CMKLClassification*)fold->get("trained_machine");
			SG_REF(machine)
			CCombinedKernel* k = (CCombinedKernel*)machine->get_kernel();
			auto w = k->get_subkernel_weights();

			/* Copy the weights inside the matrix */
			/* Each of the columns will represent a set of weights */
			for (auto j = 0; j < w.size(); j++)
			{
				weights.set_element(w[j], j, column);
			}

			SG_UNREF(k)
			SG_UNREF(machine)
			SG_UNREF(fold)
			column++;
		}
		SG_UNREF(obs_storage)
	}
	return weights;
}

void test_mkl_cross_validation()
{
	/* generate random data */
	index_t num=10;
	index_t dims=2;
	float64_t dist=0.5;
	SGVector<float64_t> lab(num);
	SGMatrix<float64_t> feat(dims, num);
	gen_rand_data(lab, feat, dist);

	/*create train labels */
	CLabels* labels=new CBinaryLabels(lab);

	/* create train features */
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(feat);
	SG_REF(features);

	/* create combined features */
	CCombinedFeatures* comb_features=new CCombinedFeatures();
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	SG_REF(comb_features);

	/* create multiple gaussian kernels */
	CCombinedKernel* kernel=new CCombinedKernel();
	kernel->append_kernel(new CGaussianKernel(10, 0.1));
	kernel->append_kernel(new CGaussianKernel(10, 1));
	kernel->append_kernel(new CGaussianKernel(10, 2));
	kernel->init(comb_features, comb_features);
	SG_REF(kernel);

	/* create mkl using libsvm, due to a mem-bug, interleaved is not possible */
	CMKLClassification* svm=new CMKLClassification(new CLibSVM());
	svm->set_interleaved_optimization_enabled(false);
	svm->set_kernel(kernel);
	SG_REF(svm);

	/* create cross-validation instance */
	index_t num_folds=3;
	CSplittingStrategy* split=new CStratifiedCrossValidationSplitting(labels,
			num_folds);
	CEvaluation* eval=new CContingencyTableEvaluation(ACCURACY);
	CCrossValidation* cross=new CCrossValidation(svm, comb_features, labels, split, eval, false);

	/* add print output listener and mkl storage listener */
	CParameterObserverCV mkl_obs{true};
	cross->subscribe(&mkl_obs);

	/* perform cross-validation, this will print loads of information */
	CEvaluationResult* result=cross->evaluate();

	/* print mkl weights */
	auto weights = calculate_weights(mkl_obs, num_folds, 1, 3);
	weights.display_matrix("mkl weights");

	/* print mean and variance of each kernel weight. These could for example
	 * been used to compute confidence intervals */
	CStatistics::matrix_mean(weights, false).display_vector("mean per kernel");
	CStatistics::matrix_variance(weights, false).display_vector("variance per kernel");
	CStatistics::matrix_std_deviation(weights, false).display_vector("std-dev per kernel");

	/* Clear */
	mkl_obs.clear();
	SG_UNREF(result);

	/* again for two runs */
	cross->set_num_runs(2);
	result=cross->evaluate();

	/* print mkl weights */
	SGMatrix<float64_t> weights_2 = calculate_weights(mkl_obs, num_folds, 2, 3);
	weights_2.display_matrix("mkl weights");

	/* print mean and variance of each kernel weight. These could for example
	 * been used to compute confidence intervals */
	CStatistics::matrix_mean(weights_2, false)
	    .display_vector("mean per kernel");
	CStatistics::matrix_variance(weights_2, false)
	    .display_vector("variance per kernel");
	CStatistics::matrix_std_deviation(weights_2, false)
	    .display_vector("std-dev per kernel");

	/* clean up */
	SG_UNREF(result);
	SG_UNREF(cross);
	SG_UNREF(kernel);
	SG_UNREF(features);
	SG_UNREF(comb_features);
	SG_UNREF(svm);
}

int main()
{
	init_shogun_with_defaults();

//	sg_io->set_loglevel(MSG_DEBUG);

	test_mkl_cross_validation();

	exit_shogun();
	return 0;
}

