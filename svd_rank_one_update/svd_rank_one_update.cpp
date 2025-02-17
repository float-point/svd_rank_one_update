
#include <vector>

#include "secular_equation.h"
#include <mkl_cblas.h>
#include "fmm.h"
#include <mkl.h>
#include <algorithm>
#include <numeric>
#include <tuple>

using namespace std;

#define CHECK
//#define DEBUG_PRINT

#ifdef DEBUG_PRINT
	#include <ostream>
	#include <iostream>
#endif

void fmm_matrix(const int nx, const int ny, const int dimv, 
	const double* x, const double* y, const double* const* v, double* const* new_v) {
	for (auto i = 0; i < ny; ++i) {
		memset(new_v[i], 0, dimv * sizeof(double));
		for (auto j = 0; j < nx; ++j)
			cblas_daxpy(dimv, 1. / (y[i] - x[j]), v[j], 1, new_v[i], 1);
	}
}

bool 
rank_one_update_with_deflation(
	double* U, // because of scaling by a_line
	uint64_t d,
	uint64_t m,
	vector<double>& a, 
	const double* D,
	double* new_U,
	double* new_D,
	double rho,
	vector<double>& cauchy,
	const fmm_context& ctx)
{
	vector<double> a_line(m);
	cblas_dgemv(CblasRowMajor, CblasNoTrans, m, m, 1., U, m, a.data(), 1, 0., a_line.data(), 1);

	const auto a_line_len = cblas_dnrm2(m, a_line.data(), 1);
	cblas_dscal(m, 1. / a_line_len, a_line.data(), 1);
	rho *= a_line_len * a_line_len;

	vector<int> non_zero_indexes, zero_indexes;
	non_zero_indexes.reserve(m);
	for (auto i = 0; i < m; ++i)
		if (a_line[i] != 0.0) // todo: eps
			non_zero_indexes.push_back(i);
		else
			zero_indexes.push_back(i);

	if (non_zero_indexes.size() == 0)
		return true; // todo

	vector<tuple<int, int, double, double>> rotations;
	auto o = non_zero_indexes[0];
	for(auto ind = 1; ind< non_zero_indexes.size();++ind) {
		auto i = non_zero_indexes[ind];
		if (D[o] - D[i] != 0.0) { // todo: eps
			o = i;
			continue;
		}
		double c, s;
		cblas_drotg(&a_line[o], &a_line[i], &c, &s);
		a_line[i] = 0.;
		rotations.emplace_back(o, i, c, s);
		non_zero_indexes[i] = -1;
		zero_indexes.push_back(i);
	}
	auto nnz = 0;
	for (auto ind = 0; ind < non_zero_indexes.size(); ++ind)
		if (non_zero_indexes[ind] != -1)
			non_zero_indexes[nnz++] = non_zero_indexes[ind];
	non_zero_indexes.resize(nnz);
	// после выполнения поворотов гивенса норма вектора a_line должна сохраниться
	// но на всякий случай дополнительно отнормируем его позже (или не надо?)
	// порядок собственных значений сохраняется
	
	vector<double> a_line_deflated(nnz), D_deflated(nnz);
	for (auto nnz_ind = 0; nnz_ind<nnz; ++nnz_ind) {
		const auto ind = non_zero_indexes[nnz_ind];
		a_line_deflated[nnz_ind] = a_line[ind];
		     D_deflated[nnz_ind] =      D[ind];
	}
	vector<double> new_D_deflated(nnz);
	if (!solve_secular_equation_stor(D_deflated, a_line_deflated, rho, new_D_deflated))
		return false;
	vector<double> t_D(m);
	for (auto ind : zero_indexes) t_D[ind] = D[ind];
	for (auto nnz_ind = 0; nnz_ind<nnz; ++nnz_ind) {
		const auto ind = non_zero_indexes[nnz_ind];
		t_D[ind] = new_D_deflated[nnz_ind];
	}

	vector<size_t> perm(m), iperm(m);
	iota(perm.begin(), perm.end(), 0);
	sort(perm.begin(), perm.end(), [t_D](auto i1, auto i2) {return t_D[i1] > t_D[i2]; });
	for (size_t i = 0; i < m; ++i) iperm[perm[i]] = i;
	for (size_t i = 0; i < m; ++i) new_D[iperm[i]] = t_D[i];
	for (auto ind : zero_indexes)
		memcpy(new_U + iperm[ind] * m, U + ind * m, m * sizeof(double));

	vector<double*> U_deflated(nnz), new_U_deflated(nnz);
	for(auto i = 0; i < nnz; ++i) {
		const auto old_ind = non_zero_indexes[i];
		     U_deflated[i] =     U +       old_ind   * m;
		 new_U_deflated[i] = new_U + iperm[old_ind]  * m;
		 cblas_dscal(m, a_line_deflated[i], U_deflated[i], 1);
	}

	fmm_matrix(nnz, nnz, m, D_deflated.data(), new_D_deflated.data(), U_deflated.data(), new_U_deflated.data());
	// todo: ensure valid order
	//fmm_task task = {
	//	D_deflated.data(),
	//	new_D_deflated.data(),
	//	U_deflated.data(),
	//	new_U_deflated.data(),
	//	nnz, nnz, n, 0, true, cauchy.data() };
	//fmm(&task, &ctx);

	//for (auto i = 0; i < nnz; ++i) // todo: mkl_ddiamm
	//	cblas_dscal(n, 1. / cblas_dnrm2(n, cauchy.data() + i * n, 1), new_U_deflated[i], 1);
	for (auto i = 0; i < nnz; ++i) {
		auto sum = 0.;
		for(auto j = 0; j < nnz; ++j) {
			const auto c = a_line_deflated[j]/(D_deflated[j] - new_D_deflated[i]);
			sum += c * c;
		}
		cblas_dscal(m, 1. / sqrt(sum), new_U_deflated[i], 1);
	}

	for(auto ind=static_cast<int>(rotations.size())-1; ind >= 0; --ind) {
		const auto o = get<0>(rotations[ind]);
		const auto i = get<1>(rotations[ind]);
		const auto c = get<2>(rotations[ind]);
		const auto s = get<3>(rotations[ind]);
		cblas_drot(m, new_U + iperm[o] * m, 1, new_U + iperm[i] * m, 1, c, -s);
	}
	return true;
}

double sqrt_spec(double x) {
	if (abs(x) < 10.)
		return sqrt(1. + 1. / x);
	vector<double> powers(16);
	powers[0] = 1.; powers[1] = 1./x;
	for (int i = 2; i < 16; ++i)
		powers[i] = powers[i - 1] / x;
	auto res = 0.;
	res += 334305. / 67108864. * powers[15];
	res += -185725. / 33554432. * powers[14];
	res += 52003. / 8388608. * powers[13];
	res += -29393. / 4194304. * powers[12];
	res += 4199. / 524288. * powers[11];
	res += -2431. / 262144. * powers[10];
	res += 715. / 65536. * powers[9];
	res += -429. / 32768. * powers[8];
	res += 33. / 2048. *powers[7];
	res += -21. / 1024. *powers[6];
	res += 7. / 256. * powers[5];
	res += -5. / 128. *powers[4];
	res += powers[3] / 16.;
	res += -powers[2] / 8.;
	res += powers[1] / 2.;
	res += 1.;
	return res;
}

void schur_spec_decomposition(double a, double& rho1, double& rho2, double& m1, double& m2) {
	a /= 2;
	const auto a_sq = a * a;
	rho1 = a * (1. + sqrt_spec(a_sq));
	rho2 = -1 / rho1;
	const auto rho1_sq = rho1 * rho1;
	const auto rho2_sq = rho2 * rho2;
	m1 = 1. / rho1 / sqrt_spec(rho1_sq);
	m2 = 1. / rho2 / sqrt_spec(rho2_sq);
}

int 
svd_rank_one_update(
	const int m, const int n, 
	const double *A,  // in  [m x n] matrix
	const double *UT, // in  [d x m] matrix
	const double *S,  // in  [d] vector
	const double *VT, // in  [d x n] matrix
	const double *a,  // in  [m] vector
	const double *b,  // in  [n] vector
	double *UT2,      // out [d x m] matrix
	double *S2,	      // out [d] vector
	double *VT2       // out [d x n] matrix
) 
{
	const auto d = min(m, n);
	const auto max_dim = max(m, n);
	vector<double> D(max_dim, 0.0); // with reserve
	vector<double> a_wave(n), b_wave(m);

	// a_wave = A'.a
	memset(D.data(), 0, max_dim * sizeof(double));
	cblas_dgemv(CblasRowMajor, CblasNoTrans, m, m, 1., UT, m, a, 1, 0., D.data(), 1); // temporarily use D
	vdMul(d, D.data(), S, D.data());
	cblas_dgemv(CblasRowMajor, CblasTrans, n, n, 1., VT, n, D.data(), 1, 0., a_wave.data(), 1);

	// b_wave = A.b
	memset(D.data(), 0, max_dim * sizeof(double));
	cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1., VT, n, b, 1, 0., D.data(), 1); // temporarily use D
	vdMul(d, D.data(), S, D.data());
	cblas_dgemv(CblasRowMajor, CblasTrans, m, m, 1., UT, m, D.data(), 1, 0., b_wave.data(), 1);

	// alpha = a'.a, beta = b'.b
	const auto alpha = cblas_ddot(m, a, 1, a, 1);
	const auto  beta = cblas_ddot(n, b, 1, b, 1);
	
	// D = S^2
	memset(D.data(), 0, max_dim * sizeof(double));
	vdMul(d, S, S, D.data());

	double rho1, rho2, mu1, mu2;
	schur_spec_decomposition(beta , rho1, rho2, mu1, mu2);
	vector<double> a1(m, 0.), b1(m, 0.);
	cblas_daxpy(m, rho1*mu1, a            , 1, a1.data(), 1);
	cblas_daxpy(m,      mu1, b_wave.data(), 1, a1.data(), 1);
	cblas_daxpy(m, rho2*mu2, a            , 1, b1.data(), 1);
	cblas_daxpy(m,	    mu2, b_wave.data(), 1, b1.data(), 1);

	vector<double> tUV(max_dim*max_dim), tD(max_dim), D_res(max_dim), cauchy_storage(max_dim*max_dim);
	fmm_context fmm_context;
	init_fmm_context(&fmm_context, 20);

	const auto max_d_m = max(d, m);
	memcpy(UT2, UT, m*m * sizeof(double)); // because of scaling inside rank_one_update_with_deflation
	if (!rank_one_update_with_deflation(UT2, max_d_m, m, b1, D.data(), tUV.data(),tD.data(), rho2, cauchy_storage, fmm_context))
		return 1;
#ifdef DEBUG_PRINT
	cout << "temp U:" << endl;
	for (int i = 0; i < m; ++i) {
		cout << "  ";
		for (int j = 0; j < m; ++j)
			cout << tUV[i*m + j] << ", ";
		cout << endl;
	}
	cout << "temp D:" << endl << "  ";
	for (int i = 0; i < m; ++i)
		cout << tD[i] << ", ";
	cout << endl;
#endif

	if (!rank_one_update_with_deflation(tUV.data(), max_d_m, m, a1, tD.data(), UT2, D_res.data(), rho1, cauchy_storage, fmm_context))
		return 1;

#ifdef DEBUG_PRINT
	cout << "U:" << endl;
	for (int i = 0; i < m; ++i) {
		cout << "  ";
		for (int j = 0; j < m; ++j)
			cout << UT2[i*m + j] << ", ";
		cout << endl;
	}
	cout << "UD:" << endl << "  ";
	for (int i = 0; i < m; ++i)
		cout << D_res[i] << ", ";
	cout << endl;
#endif

	schur_spec_decomposition(alpha, rho1, rho2, mu1, mu2);
	vector<double> a2(n, 0.), b2(n, 0.);
	cblas_daxpy(n, rho1*mu1, b            , 1, a2.data(), 1);
	cblas_daxpy(n,      mu1, a_wave.data(), 1, a2.data(), 1);
	cblas_daxpy(n, rho2*mu2, b            , 1, b2.data(), 1);
	cblas_daxpy(n,      mu2, a_wave.data(), 1, b2.data(), 1);

	const auto max_d_n = max(d, n);
	memcpy(VT2, VT, n*n * sizeof(double)); // because of scaling inside rank_one_update_with_deflation
	if (!rank_one_update_with_deflation(VT2, max_d_n, n, b2, D.data(), tUV.data(), tD.data(), rho2, cauchy_storage, fmm_context))
		return 1;

#ifdef DEBUG_PRINT
	cout << "temp V:" << endl;
	for (int i = 0; i < n; ++i) {
		cout << "  ";
		for (int j = 0; j < n; ++j)
			cout << tUV[i*n + j] << ", ";
		cout << endl;
	}
	cout << "temp D:" << endl << "  ";
	for (int i = 0; i < n; ++i)
		cout << tD[i] << ", ";
	cout << endl;
#endif
	if (!rank_one_update_with_deflation(tUV.data(), max_d_n, n, a2, tD.data(), VT2, D_res.data(), rho1, cauchy_storage, fmm_context))
		return 1;
#ifdef DEBUG_PRINT
	cout << "V:" << endl;
	for (int i = 0; i < n; ++i) {
		cout << "  ";
		for (int j = 0; j < n; ++j)
			cout << VT2[i*n + j] << ", ";
		cout << endl;
	}
	cout << "VD:" << endl << "  ";
	for (int i = 0; i < n; ++i)
		cout << D_res[i] << ", ";
	cout << endl;
#endif

	free_fmm_context(&fmm_context);

	for (auto i = 0; i < d; ++i)
		if (D_res[i] < 0) {
			cblas_dscal(n, -1.0, VT2 + i * n, 1);
			D_res[i] = -D_res[i];
		}

	vdSqrt(d, D_res.data(), S2);

	// sign correction
	// https://math.stackexchange.com/questions/2359992/how-to-resolve-the-sign-issue-in-a-svd-problem
	for(auto i = 0; i < d; ++i) {
		if (S[i] == 0.) continue;

		auto abs_max_Ui = abs(UT2[i*m + 0]);
		auto arg_abs_max_Ui = 0;
		for(auto j=1; j < m; ++j) {
			const auto abs_Ui_j = abs(UT2[i*m + j]);
			if(abs_Ui_j > abs_max_Ui) {
				abs_max_Ui = abs_Ui_j;
				arg_abs_max_Ui = j;
			}
		}
		memcpy(a2.data(), A + arg_abs_max_Ui * n, n * sizeof(double));  // reuse a2 vector
		cblas_daxpy(n, a[arg_abs_max_Ui], b, 1, a2.data(), 1);
		const auto desired_UT2_i_j = cblas_ddot(n, a2.data(), 1, VT2 + i*n, 1);
		if (desired_UT2_i_j * UT2[i*m + arg_abs_max_Ui] < 0)
			cblas_dscal(n, -1.0, VT2 + i * n, 1);
	}
	return 0;
}