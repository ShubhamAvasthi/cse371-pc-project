#pragma once

#include "Ensemble.h"
#include "mlpack/core.hpp"
#include <atomic>
#include <cstdio>
#include <sstream>
#include <string>

class Stacking : public Ensemble
{
public:
	Stacking(
		const arma::mat train,
		const arma::Row<size_t> train_labels,
		const int _num_classifiers = 2,
		const int sample_size = 100,
		const int _num_classes = 2,
		const int minimum_leaf_size = 5
	) : Ensemble(_num_classifiers, _num_classes)
	{
		train.save("StackingParallelTrain_" + std::to_string(object_id) + ".bin");
		train_labels.save("StackingParallelTrainLabels_" + std::to_string(object_id) + ".bin");

		std::stringstream command_ss;
		command_ss << "mpiexec -n " << num_classifiers << " StackingParallelTrain.exe " \
			<< object_id << ' '<< num_classes;
		system(command_ss.str().c_str());

		std::remove(("StackingParallelTrain_" + std::to_string(object_id) + ".bin").c_str());
		std::remove(("StackingParallelTrainLabels_" + std::to_string(object_id) + ".bin").c_str());
	}

	void predict(const arma::mat &X, arma::Row<size_t> &ensemble_predictions)
	{
		X.save("StackingParallelPredictX_" + std::to_string(object_id) + ".bin");

		std::stringstream command_ss;
		command_ss << "mpiexec -n " << num_classifiers << " StackingParallelPredict.exe " << object_id;
		system(command_ss.str().c_str());

		std::remove(("StackingParallelPredictX_" + std::to_string(object_id) + ".bin").c_str());

		ensemble_predictions.load(("StackingParallelPredictions_" + std::to_string(object_id) + ".bin").c_str());
	}
};
