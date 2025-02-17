#include <algorithm>
#include <random>
#include <functional>

#include "mkl.h"

#include "../svd_rank_one_update/fmm.h"
#include <numeric>
#include <iostream>
#include <fstream>
#include "../svd_rank_one_update/svd_rank_one_update.h"
#include "svd_rank_one_update_tests.h"
#include <string>
#include <sstream>
#include <chrono>

using namespace std;

// [dimv, nx] * [nx, ny] = [dimv, ny]
// [ny, nx] * [nx, dimv] = [ny, dimv]

vector<double> fmm_matrix(const vector<double>& x, const vector<double>& y, const vector<double>& v) {
	const auto nx = x.size();
	const auto ny = y.size();
	const auto dimv = v.size() / nx;
	vector<double> res(ny*dimv, 0.);
	vector<double> cauchy(ny*nx, 0.);

	for (uint64_t y_i = 0; y_i < ny; ++y_i)
		for (uint64_t x_i = 0; x_i < nx; ++x_i)
			cauchy[y_i*nx + x_i] = 1. / (y[y_i] - x[x_i]);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		ny, dimv, nx, 1.,
		cauchy.data(), nx, v.data(), dimv,
		0., res.data(), dimv);
	return res;
}

vector<double> fmm_ideal(const vector<double>& x, const vector<double>& y, const vector<double>& v) {
	const auto nx = x.size();
	const auto ny = y.size();
	const auto dimv = v.size() / nx;
	vector<double> res(ny*dimv, 0.);

	const auto add_vec = [&v, &res, &dimv](const int i1, const int i2, const double div) {
		cblas_daxpy(dimv, 1. / div,
			v  .data() + i2 * dimv, 1,
			res.data() + i1 * dimv, 1);
	};

	for(uint64_t y_i = 0; y_i < ny; ++y_i) {
		auto li = 0;
		auto ri = nx - 1;
		auto ldiff = x[li] - y[y_i];
		auto rdiff = y[y_i] - x[ri];
		while (ldiff > 0 || rdiff > 0) {
			if(li == ri) {
				if (ldiff < 0)	add_vec(y_i, ri,  rdiff);
				else			add_vec(y_i, li, -ldiff);
				break;
			}
			if (ldiff < 0) {
				add_vec(y_i, ri, rdiff);
				rdiff = y[y_i] - x[--ri];
			}
			else if (rdiff < 0) {
				add_vec(y_i, li, -ldiff);
				ldiff = x[++li] - y[y_i];
			}
			else if (ldiff > rdiff) {
				add_vec(y_i, li, -ldiff);
				ldiff = x[++li] - y[y_i];
			}
			else {
				add_vec(y_i, ri, rdiff);
				rdiff = y[y_i] - x[--ri];
			}
		}
	}
	return res;
}

struct sample {
	vector<double> x, y, v, r;
	uint64_t nx, ny, dimv;
	vector<double*> vrows;
};

struct sample_result {
	double error, peak_ratio;
	int iteration;
};

vector<sample> load_samples(const string& filename) {
	ifstream is;
	is.open(filename, ios::binary);
	is.seekg(0, ios::end);
	const auto length = is.tellg();
	is.seekg(0, ios::beg);
	vector<char> buffer(length);
	is.read(buffer.data(), length);
	is.close();

	auto ptr = buffer.data();
	const auto nsamples = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;
	vector<sample> samples(nsamples);
	for (auto i = 0; i<nsamples; ++i) {
		auto &s = samples[i];
		s.nx = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;
		s.ny = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;
		s.dimv = *reinterpret_cast<uint64_t*>(ptr); ptr += 8;
		s.x.resize(s.nx);
		s.y.resize(s.ny);
		s.v.resize(s.nx*s.dimv);
		s.r.resize(s.ny*s.dimv);
		s.vrows.resize(s.nx);
		for (auto j = 0; j < s.x.size(); ++j, ptr += 8) s.x[j] = *reinterpret_cast<double*>(ptr);
		for (auto j = 0; j < s.y.size(); ++j, ptr += 8) s.y[j] = *reinterpret_cast<double*>(ptr);
		for (auto j = 0; j < s.v.size(); ++j, ptr += 8) s.v[j] = *reinterpret_cast<double*>(ptr);
		for (auto j = 0; j < s.r.size(); ++j, ptr += 8) s.r[j] = *reinterpret_cast<double*>(ptr);
		for (auto j = 0; j < s.nx; ++j) s.vrows[j] = s.v.data() + s.dimv*j;
	}
	return samples;
}

void test_accuracy(
	const vector<sample>& samples, 
	function<vector<double>(const sample& )> test_func) 
{
	vector<sample_result> results;
	for (auto sample_ind = 0; sample_ind < samples.size(); ++sample_ind) {
		auto& sample = samples[sample_ind];
		const auto ideal_norm = matrix_norm(sample.r);
		const auto result = test_func(sample);
		const auto error = matrix_norm(result, sample.r) / ideal_norm;
		vector<double> abs_diff(result.size());
		transform(result.begin(), result.end(), sample.r.begin(), abs_diff.begin(),
			[](auto x, auto y) { return abs(x - y); });
		const auto peak_ratio = *max_element(abs_diff.begin(), abs_diff.end()) / ideal_norm;
		results.push_back(sample_result{ error, peak_ratio, sample_ind });
	}
	//
	sort(results.begin(), results.end(), [](const auto& x, const auto& y) { return x.error < y.error; });
	cout << "min error: " << results[0].error << " at iteraion " << results[0].iteration << endl;
	cout << "max error: " << results[results.size() - 1].error << " at iteraion " << results[results.size() - 1].iteration << endl;
	cout << "median error: " << results[results.size() / 2].error << endl;
	auto sum = accumulate(results.begin(), results.end(), 0., [](auto acc, const auto& r) { return acc + r.error; });
	auto mean = sum / results.size();
	auto sq_sum = inner_product(results.begin(), results.end(), results.begin(), 0., plus<double>(),
		[](const auto& x, const auto& y) { return x.error*y.error; });
	auto stdev = sqrt(sq_sum / results.size() - mean * mean);
	cout << "mean error: " << mean << endl;
	cout << "stdev: " << stdev << endl;

	cout << "peak ratio: " << endl;
	sort(results.begin(), results.end(), [](const auto& x, const auto& y) { return x.peak_ratio < y.peak_ratio; });
	cout << "min: " << results[0].peak_ratio << " at iteraion " << results[0].iteration << endl;
	cout << "max: " << results[results.size() - 1].peak_ratio << " at iteraion " << results[results.size() - 1].iteration << endl;
	cout << "median: " << results[results.size() / 2].peak_ratio << endl;
	sum = accumulate(results.begin(), results.end(), 0., [](auto acc, const auto& r) { return acc + r.peak_ratio; });
	mean = sum / results.size();
	sq_sum = inner_product(results.begin(), results.end(), results.begin(), 0., plus<double>(),
		[](const auto& x, const auto& y) { return x.peak_ratio*y.peak_ratio; });
	stdev = sqrt(sq_sum / results.size() - mean * mean);
	cout << "mean " << mean << endl;
	cout << "stdev: " << stdev << endl;
}

double mean_accuracy(
	const vector<sample>& samples,
	function<vector<double>(const sample&)> test_func)
{
	vector<sample_result> results;
	for (auto sample_ind = 0; sample_ind < samples.size(); ++sample_ind) {
		auto& sample = samples[sample_ind];
		const auto ideal_norm = matrix_norm(sample.r);
		const auto result = test_func(sample);
		const auto error = matrix_norm(result, sample.r) / ideal_norm;
		vector<double> abs_diff(result.size());
		transform(result.begin(), result.end(), sample.r.begin(), abs_diff.begin(),
			[](auto x, auto y) { return abs(x - y); });
		const auto peak_ratio = *max_element(abs_diff.begin(), abs_diff.end()) / ideal_norm;
		results.push_back(sample_result{ error, peak_ratio, sample_ind });
	}
	auto sum = accumulate(results.begin(), results.end(), 0., [](auto acc, const auto& r) { return acc + r.error; });
	auto mean = sum / results.size();
	return mean;
}

void uniform_test_100() {
	const auto samples = load_samples("../tests/fmm_uniform_test_100");
	test_accuracy(samples, [](const sample& s) { return fmm_matrix(s.x, s.y, s.v); });
	test_accuracy(samples, [](const sample& s) { return fmm_ideal(s.x, s.y, s.v); });

	cout << endl << "FMM" << endl;
	for (int p = 2; p < 25; ++p) {
		cout << endl << "p = " << p << endl;
		fmm_context ctx;
		init_fmm_context(&ctx, p);
		test_accuracy(samples, [&ctx](const sample& s) {
			vector<double> r(s.ny*s.dimv, 0.);
			vector<double*> rrows(s.ny);
			for (auto j = 0; j < s.ny; ++j)
				rrows[j] = r.data() + s.dimv*j;
			fmm_task task = {
				s.x.data(),
				s.y.data(),
				s.vrows.data(),
				rrows.data(),
				s.nx, s.ny, s.dimv, 0, false };
			fmm(&task, &ctx);
			return r;
		});
		free_fmm_context(&ctx);
	}

	cout << endl << "FMM accurate" << endl;
	for (int p = 2; p < 25; ++p) {
		cout << endl << "p = " << p << endl;
		fmm_context ctx;
		init_fmm_context(&ctx, p);
		test_accuracy(samples, [&ctx](const sample& s) {
			vector<double> r(s.ny*s.dimv, 0.);
			vector<double*> rrows(s.ny);
			for (auto j = 0; j < s.ny; ++j)
				rrows[j] = r.data() + s.dimv*j;
			fmm_task task = {
				s.x.data(),
				s.y.data(),
				s.vrows.data(),
				rrows.data(),
				s.nx, s.ny, s.dimv, 0, true };
			fmm(&task, &ctx);
			return r;
		});
		free_fmm_context(&ctx);
	}
}

void uniform_test_100_mean_relative_accuracy() {
	const auto samples = load_samples("../tests/fmm_uniform_test_100");

	ofstream ofs("../tests/fmm_uniform_test_100_mean_rel_acc.txt");
	for (int p = 2; p < 25; ++p) {
		ofs << p << ", ";
		fmm_context ctx;
		init_fmm_context(&ctx, p);
		ofs << mean_accuracy(samples, [&ctx](const sample& s) {
			vector<double> r(s.ny*s.dimv, 0.);
			vector<double*> rrows(s.ny);
			for (auto j = 0; j < s.ny; ++j)
				rrows[j] = r.data() + s.dimv*j;
			fmm_task task = {
				s.x.data(),
				s.y.data(),
				s.vrows.data(),
				rrows.data(),
				s.nx, s.ny, s.dimv, 0, true };
			fmm(&task, &ctx);
			return r;
		});
		if (p < 24) ofs << ", ";
		free_fmm_context(&ctx);
	}
}

double benchmark(int n_iterations, function<void()> func)
{
	const auto t1 = chrono::high_resolution_clock::now();
	for (int i = 0; i < n_iterations; ++i)
		func();
	const auto t2 = chrono::high_resolution_clock::now();
	const double total_seconds = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
	return total_seconds / n_iterations;
}

void fmm_benchmark() {
	vector<int> sizes = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000 };
	const double benchmark_size = 1e7;
	ofstream ofs("../tests/fmm_benchmark.txt");

	cout << endl << "fmm benchmark:" << endl;
	cout << "task dim, ffm seconds, dgemm seconds" << endl;
	//outfile << "single_thread_simple";
	fmm_context ctx;
	init_fmm_context(&ctx, 17);
	uniform_real_distribution<double>distribution;
	default_random_engine generator;

	for (auto s : sizes) {
		ofs << s << ", ";
		cout << s << ", ";
		vector<double> v(s*s), x(s), y(s), r(s*s);
		vector<double*> vrows(s), rrows(s);
		generate(v.begin(), v.end(), [&]() { return distribution(generator); });
		generate(x.begin(), x.end(), [&]() { return distribution(generator); });
		generate(y.begin(), y.end(), [&]() { return distribution(generator); });
		sort(x.begin(), x.end(), [](auto a, auto b) {return a > b; });
		sort(y.begin(), y.end(), [](auto a, auto b) {return a > b; });

		for (auto j = 0; j < s; ++j) {
			vrows[j] = v.data() + s * j;
			rrows[j] = r.data() + s * j;
		}
		const int n_iterations = max(1, static_cast<int>(benchmark_size / s / s / s));

		const auto time = benchmark(n_iterations, [&]() {
			fmm_task task = {
				x.data(),
				y.data(),
				vrows.data(),
				rrows.data(),
				s, s, s, 0, true };
			fmm(&task, &ctx);
		});

		const auto time2 = benchmark(n_iterations, [&]() {
			vector<double> cauchy(s*s, 0.);
			for (uint64_t y_i = 0; y_i < s; ++y_i)
				for (uint64_t x_i = 0; x_i < s; ++x_i)
					cauchy[y_i*s + x_i] = 1. / (y[y_i] - x[x_i]);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				s, s, s, 1.,
				cauchy.data(), s, v.data(), s,
				0., r.data(), s);
		});

		ofs << time << ", " << time2 << ", ";
		cout << time << ", " << time2 << endl;
	}
}


int main()
{
	//simple_test();
	//uniform_test_100();
	//uniform_test_100_mean_relative_accuracy();
	fmm_benchmark();
}

