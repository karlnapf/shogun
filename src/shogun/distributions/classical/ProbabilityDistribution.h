/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#ifndef PROBABILITYDISTRIBUTION_H
#define PROBABILITYDISTRIBUTION_H

#include <shogun/base/SGObject.h>

namespace shogun
{

template <class T> class SGVector;
template <class T> class SGMatrix;

/** @brief A base class for representing n-dimensional probability distribution
 * over the real numbers (64bit) for which various statistics can be computed
 * and which can be sampled.
 */
class CProbabilityDistribution: public CSGObject
{
public:
	/** Default constructor */
	CProbabilityDistribution();

	/** Constructur that sets the distribution's dimension */
	CProbabilityDistribution(int32_t dimension);

	/** Destructor */
	virtual ~CProbabilityDistribution();

	/** Samples from the distribution multiple times
	 *
	 * @param num_samples number of samples to generate
	 * @return matrix with samples (column vectors)
	 */
	virtual SGMatrix<float64_t> sample(int32_t num_samples);

	/** Samples from the distribution once. Wrapper method.
	 *
	 * @return vector with single sample
	 */
	virtual SGVector<float64_t> sample();

	/** Computes the log-pdf for all provided samples
	 *
	 * @param samples samples to compute log-pdf of (column vectors)
	 * @return vector with log-pdfs of given samples
	 */
	virtual SGVector<float64_t> log_pdf(SGMatrix<float64_t> samples);

	/** Computes the log-pdf for a single provided sample. Wrapper method.
	 *
	 * @param sample sample to compute log-pdf for
	 * @return log-pdf of the given sample
	 */
	virtual float64_t log_pdf(SGVector<float64_t> single_sample);

	/** @return name of the SGSerializable */
	virtual const char* get_name() const=0;

private:

	/** Initialses and registers parameters */
	void init();

protected:
	/** Dimension of the distribution */
	int32_t m_dimension;
};

}

#endif // PROBABILITYDISTRIBUTION_H
