#pragma once

#include <mkl_trans.h>
#include <mkl_lapacke.h>
#include <vector>
#include <algorithm>
#include <mkl_cblas.h>
#include <iostream>
#include <numeric>
#include "../svd_rank_one_update/svd_rank_one_update.h"

using namespace std;

inline double matrix_norm(const vector<double>& m) {
	return sqrt(accumulate(m.begin(), m.end(), 0., [](auto acc, auto x) { return acc + x * x; }) / m.size());
}
inline double matrix_norm(const vector<double>& a, const vector<double>& b) {
	vector<double> c(a.size());
	transform(a.begin(), a.end(), b.begin(), c.begin(), minus<>());
	return matrix_norm(c);
}

inline bool svd(const int m, const int n, const vector<double>& a, vector<double>& ut, vector<double>& s, vector<double>& vt) {
	const auto d = std::min(m, n);
	std::vector<double> superb(d);
	auto a_copy = a;
	ut.resize(m*m);
	s.resize(d);
	vt.resize(n*n);

	const auto info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n,
		a_copy.data(), n, s.data(), ut.data(), d, vt.data(), n, superb.data());
	mkl_dimatcopy('R', 'T', m, m, 1., ut.data(), m, m);
	return info == 0;
}


struct simple_sample {
	int m, n;
	vector<double> m1, m2, a, b;
};

const vector<simple_sample> standart_samples = {
	simple_sample{ 2, 2, { 1., 1., 2., 4. },
						 { 1., 2., 2., 4. }, {1., 0.}, {0., 1.} },
	simple_sample{ 2, 2, { 1., 2., 2., 4. },
						 { 1., 3., 2., 4. }, {1., 0.}, {0., 1.} },
	simple_sample{ 2, 2, { 1., 2., 2., 4. },
						 { 1., 100002., 2., 4. }, {100000., 0.}, {0., 1.} },
	simple_sample{ 2, 2, { 1., -99998., 2., 4. },
						 { 1.,      2., 2., 4. }, {100000., 0.}, {0., 1.} },
	simple_sample{ 2, 3, { 1., 2., 2., 2., 4., 6. },
						 { 1., 2., 3., 2., 4., 6. }, {1., 0.}, {0., 0., 1.} },
	simple_sample{ 2, 3, { 1., 2., 3., 2., 4., 6. },
						 { 1., 2., 4., 2., 4., 6. }, {1., 0.}, {0., 0., 1.} },
	simple_sample{ 2, 3, { 1., 2.,      3., 2., 4., 6. },
						 { 1., 2., 100003., 2., 4., 6. }, {100000., 0.}, {0., 0., 1.} },
	simple_sample{ 2, 3, { 1., 2., -99997., 2., 4., 6. },
						 { 1., 2.,      3., 2., 4., 6. }, {100000., 0.}, {0., 0., 1.} }
	};

inline void simple_test() {
	for (size_t ind = 0; ind<standart_samples.size(); ++ind) {
		cout << "sample " << ind << ":" << endl;
		const auto& m1 = standart_samples[ind].m1;
		const auto& m2 = standart_samples[ind].m2;
		const auto& a  = standart_samples[ind].a;
		const auto& b  = standart_samples[ind].b;
		const auto m   = standart_samples[ind].m;
		const auto n   = standart_samples[ind].n;

		const auto d = min(m, n);

		vector<double> ut, s, vt;
		svd(m, n, m1, ut, s, vt);

		vector<double> ut2, s2, vt2;
		svd(m, n, m2, ut2, s2, vt2);

		vector<double> extS(m*n, 0);

		for (int i = 0; i < d; ++i)
			extS[(1 + n)*i] = s[i];
		vector<double> T(m*n), R(m*n);
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, m, 1., ut.data(), m, extS.data(), n, 0., T.data(), n);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1., T.data(), n, vt.data(), n, 0., R.data(), n);
		cout << "norm origin: " << matrix_norm(m1, R) << endl;

		for (int i = 0; i < d; ++i)
			extS[(1 + n)*i] = s2[i];
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, m, 1., ut2.data(), m, extS.data(), n, 0., T.data(), n);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1., T.data(), n, vt2.data(), n, 0., R.data(), n);
		cout << "norm target: " << matrix_norm(m2, R) << endl;

		vector<double> ut3(m*m), s3(d), vt3(n*n);
		auto r = svd_rank_one_update(m, n,
			m1.data(),
			ut.data(), s.data(), vt.data(),
			a.data(), b.data(),
			ut3.data(), s3.data(), vt3.data());
		for (auto i = 0; i < d; ++i)
			extS[(1 + n)*i] = s3[i];

		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, m, 1., ut3.data(), m, extS.data(), n, 0., T.data(), n);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1., T.data(), n, vt3.data(), n, 0., R.data(), n);
		cout << "norm updated: " << matrix_norm(m2, R) << endl;
	}
}