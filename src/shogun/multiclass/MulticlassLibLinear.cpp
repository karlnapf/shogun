/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Chiyuan Zhang, Giovanni De Toni, 
 *          Evan Shelhamer
 */

#include <shogun/lib/config.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/v_array.h>
#include <shogun/lib/Signal.h>
#include <shogun/labels/MulticlassLabels.h>

using namespace shogun;

CMulticlassLibLinear::CMulticlassLibLinear() :
	RandomMixin<CLinearMulticlassMachine>()
{
	register_parameters();
	init_defaults();
}

CMulticlassLibLinear::CMulticlassLibLinear(float64_t C, CDotFeatures* features, CLabels* labs) :
	RandomMixin<CLinearMulticlassMachine>(new CMulticlassOneVsRestStrategy(),features,(CMachine*)NULL,labs)
{
	register_parameters();
	init_defaults();
	set_C(C);
}

void CMulticlassLibLinear::init_defaults()
{
	set_C(1.0);
	set_epsilon(1e-2);
	set_max_iter(10000);
	set_use_bias(false);
	set_save_train_state(false);
	m_train_state = NULL;
}

void CMulticlassLibLinear::register_parameters()
{
	SG_ADD(&m_C, "C", "regularization constant",ParameterProperties::HYPER);
	SG_ADD(&m_epsilon, "epsilon", "tolerance epsilon");
	SG_ADD(&m_max_iter, "max_iter", "max number of iterations");
	SG_ADD(&m_use_bias, "use_bias", "indicates whether bias should be used");
}

CMulticlassLibLinear::~CMulticlassLibLinear()
{
	reset_train_state();
}

SGVector<int32_t> CMulticlassLibLinear::get_support_vectors() const
{
	if (!m_train_state)
		SG_ERROR("Please enable save_train_state option and train machine.\n")

	ASSERT(m_labels && m_labels->get_label_type() == LT_MULTICLASS)

	int32_t num_vectors = m_features->get_num_vectors();
	int32_t num_classes = ((CMulticlassLabels*) m_labels)->get_num_classes();

	v_array<int32_t> nz_idxs;
	nz_idxs.reserve(num_vectors);

	for (int32_t i=0; i<num_vectors; i++)
	{
		for (int32_t y=0; y<num_classes; y++)
		{
			if (CMath::abs(m_train_state->alpha[i*num_classes+y])>1e-6)
			{
				nz_idxs.push(i);
				break;
			}
		}
	}
	int32_t num_nz = nz_idxs.index();
	nz_idxs.reserve(num_nz);
	return SGVector<int32_t>(nz_idxs.begin,num_nz);
}

SGMatrix<float64_t> CMulticlassLibLinear::obtain_regularizer_matrix() const
{
	return SGMatrix<float64_t>();
}

bool CMulticlassLibLinear::train_machine(CFeatures* data)
{
	if (data)
		set_features((CDotFeatures*)data);

	ASSERT(m_features)
	ASSERT(m_labels && m_labels->get_label_type()==LT_MULTICLASS)
	ASSERT(m_multiclass_strategy)
	init_strategy();

	int32_t num_vectors = m_features->get_num_vectors();
	int32_t num_classes = ((CMulticlassLabels*) m_labels)->get_num_classes();
	int32_t bias_n = m_use_bias ? 1 : 0;

	liblinear_problem mc_problem;
	mc_problem.l = num_vectors;
	mc_problem.n = m_features->get_dim_feature_space() + bias_n;
	mc_problem.y = SG_MALLOC(float64_t, mc_problem.l);
	for (int32_t i=0; i<num_vectors; i++)
		mc_problem.y[i] = ((CMulticlassLabels*) m_labels)->get_int_label(i);

	mc_problem.x = m_features;
	mc_problem.use_bias = m_use_bias;

	SGMatrix<float64_t> w0 = obtain_regularizer_matrix();

	if (!m_train_state)
		m_train_state = new mcsvm_state();

	float64_t* C = SG_MALLOC(float64_t, num_vectors);
	for (int32_t i=0; i<num_vectors; i++)
		C[i] = m_C;

	Solver_MCSVM_CS solver(&mc_problem,num_classes,C,w0.matrix,m_epsilon,
	                       m_max_iter,m_max_train_time,m_train_state);
	solver.solve(m_prng);

	m_machines->reset_array();
	for (int32_t i=0; i<num_classes; i++)
	{
		CLinearMachine* machine = new CLinearMachine();
		SGVector<float64_t> cw(mc_problem.n-bias_n);

		for (int32_t j=0; j<mc_problem.n-bias_n; j++)
			cw[j] = m_train_state->w[j*num_classes+i];

		machine->set_w(cw);

		if (m_use_bias)
			machine->set_bias(m_train_state->w[(mc_problem.n-bias_n)*num_classes+i]);

		m_machines->push_back(machine);
	}

	if (!m_save_train_state)
		reset_train_state();

	SG_FREE(C);
	SG_FREE(mc_problem.y);

	return true;
}
