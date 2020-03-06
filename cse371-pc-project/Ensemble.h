#pragma once

#include "mlpack/core.hpp"
#include <atomic>

class Ensemble
{
protected:
	static std::atomic<int> object_id_cnt;
	const int object_id;
	const int num_classifiers;
	const int num_classes;

public:
	Ensemble(
		const int _num_classifiers = 4,
		const int _num_classes = 2
	) : object_id(object_id_cnt++),
		num_classifiers(_num_classifiers),
		num_classes(_num_classes) {};

	virtual void predict(const arma::mat &X, arma::Row<size_t> &ensemble_predictions) {}

	double get_score(const arma::mat &test, const arma::Row<size_t> &labels)
	{
		arma::Row<size_t> predictions;
		predict(test, predictions);
		const size_t correct = arma::accu(predictions == labels);
		return correct / double(labels.n_elem);
	}
};

std::atomic<int> Ensemble::object_id_cnt = 0;
