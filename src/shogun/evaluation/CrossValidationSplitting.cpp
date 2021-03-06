/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Thoralf Klein, Soeren Sonnenburg, 
 *          Fernando Iglesias, Viktor Gal
 */

#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/labels/Labels.h>
#include <shogun/mathematics/RandomNamespace.h>

using namespace shogun;

CCrossValidationSplitting::CCrossValidationSplitting() :
	RandomMixin<CSplittingStrategy>()
{
}

CCrossValidationSplitting::CCrossValidationSplitting(
		CLabels* labels, index_t num_subsets) :
	RandomMixin<CSplittingStrategy>(labels, num_subsets)
{
}

void CCrossValidationSplitting::build_subsets()
{
	REQUIRE(m_labels, "No labels provided.\n");
	/* ensure that subsets are empty and set flag to filled */
	reset_subsets();
	m_is_filled=true;

	/* permute indices */
	SGVector<index_t> indices(m_labels->get_num_labels());
	indices.range_fill();
	random::shuffle(indices, m_prng);

	index_t num_subsets=m_subset_indices->get_num_elements();

	/* distribute indices to subsets */
	index_t current_subset=0;
	for (index_t i=0; i<indices.vlen; ++i)
	{
		/* fill current subset */
		CDynamicArray<index_t>* current=(CDynamicArray<index_t>*)
				m_subset_indices->get_element(current_subset);

		/* add element of current index */
		current->append_element(indices.vector[i]);

		/* unref */
		SG_UNREF(current);

		/* iterate over subsets */
		current_subset=(current_subset+1) % num_subsets;
	}

	/* finally shuffle to avoid that subsets with low indices have more
	 * elements, which happens if the number of class labels is not equal to
	 * the number of subsets (external random state important for threads) */
	random::shuffle(m_subset_indices->begin(), m_subset_indices->end(), m_prng);
}
