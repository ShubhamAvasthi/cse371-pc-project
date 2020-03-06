#pragma once

#include "Ensemble.h"
#include "mlpack/core.hpp"
#include <atomic>
#include <cstdio>
#include <sstream>
#include <string>

class Bagging : public Ensemble
{
public:
	Bagging(
		const arma::mat train,
		const arma::Row<size_t> train_labels,
		const int _num_classifiers = 4,
		const int sample_size = 100,
		const int _num_classes = 2,
		const int minimum_leaf_size = 5
	) : Ensemble(_num_classifiers, _num_classes)
	{
		train.save("BaggingParallelTrain_" + std::to_string(object_id) + ".bin");
		train_labels.save("BaggingParallelTrainLabels_" + std::to_string(object_id) + ".bin");

		std::stringstream command_ss;
		command_ss << "mpiexec -n " << num_classifiers << " BaggingParallel.exe " \
			<< object_id << ' ' << sample_size << ' ' << num_classes << ' ' << minimum_leaf_size;
		system(command_ss.str().c_str());
		
		std::remove(("BaggingParallelTrain_" + std::to_string(object_id) + ".bin").c_str());
		std::remove(("BaggingParallelTrainLabels_" + std::to_string(object_id) + ".bin").c_str());
	}

	void predict(const arma::mat &X, arma::Row<size_t> &ensemble_predictions)
	{
		X.save("BaggingParallelPredictX_" + std::to_string(object_id) + ".bin");

		std::stringstream command_ss;
		command_ss << "mpiexec -n " << num_classifiers << " BaggingParallelPredict.exe " << object_id;
		system(command_ss.str().c_str());

		std::remove(("BaggingParallelPredictX_" + std::to_string(object_id) + ".bin").c_str());
		
		ensemble_predictions.load(("BaggingParallelPredictions_" + std::to_string(object_id) + ".bin").c_str());
	}
};
