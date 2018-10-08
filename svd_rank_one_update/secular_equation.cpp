#include <numeric>
#include <algorithm>
#ifdef _DEBUG
#include <iostream>
#endif

#include <mkl.h>

#include "secular_equation.h"
using namespace  std;

bool 
solve_secular_equation(
	const vector<double>& lambdas, // in decreasing order
	const vector<double>& a,	   
	const double rho, 
	vector<double>& mu) // in decreasing order 
{
	const auto n = static_cast<int>(lambdas.size());
	mu.resize(n);

	// debug workaround
	//vector<double> mat(n*n, 0.0);
	//for (int i = 0; i < n; ++i) {
	//	mat[(n + 1)*i] = lambdas[i];
	//	for (int j = 0; j < n; ++j)
	//		mat[n*i + j] += a[i] * a[j] * rho;
	//}
	//auto info2 = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, mat.data(), n, mu.data());
	//if (info2 > 0) {
	//	printf("The algorithm failed to compute eigenvalues.\n");
	//	return false;
	//}
	////sort(mu.begin(), mu.end(), greater<>());
	//return true;

	const auto a_len_sq = inner_product(a.begin(), a.end(), a.begin(), 0.);
	const auto a_len = sqrt(a_len_sq);
	vector<double> z(n, 0.);
	cblas_daxpy(a.size(), 1. / a_len, a.data(), 1, z.data(), 1);

	const auto inverse = rho < 0.;
	const auto _rho = abs(rho) * a_len_sq;

	auto d = lambdas;
	if(inverse) {
		cblas_dscal(n, -1., d.data(), 1);
	}
	else {
		for (auto i = 0; i < n / 2; ++i) swap(d[i], d[n - 1 - i]);
		for (auto i = 0; i < n / 2; ++i) swap(z[i], z[n - 1 - i]);
	}

	vector<double> deltas(n);
	int info;
	for(auto i=0;i<n;++i) {
		dlaed4(&n, &i, d.data(), z.data(), deltas.data(), &_rho, &mu[i], &info);
		if (info != 0)
			return false;
	}
	if (inverse) {
		// todo: should we multiply mu by -1 ?
		cblas_dscal(n, -1., mu.data(), 1);
		for (auto i = 0; i < n / 2; ++i) swap(mu[i], mu[n - 1 - i]);
	}
	else {
		//for (auto i = 0; i < n / 2; ++i) swap(mu[i], mu[n - 1 - i]);
	}
		

#ifdef _DEBUG
	const auto rerror = rerror_secular_equation_solution(mu, lambdas, a, rho);
	if (rerror > DBL_EPSILON)
		cout << "secular solution error: " << rerror << endl;
#endif
	return true;
}

double 
rerror_secular_equation_solution(
	const vector<double>& lambdas,
	const vector<double>& d, 
	const vector<double>& z,
	const double rho) 
{
	const auto n = d.size();
	vector<double> m(n*n, 0.);
	for (size_t i = 0; i < n; ++i) {
		m[(1 + n)*i] += d[i];
		for (size_t j = 0; j < n; ++j)
			m[i*n + j] += rho * z[i] * z[j];
	}
	vector<double> eigenvalues(n);
	auto res = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', n, m.data(), n, eigenvalues.data());
	for (auto i = 0; i < n / 2; ++i) swap(eigenvalues[i], eigenvalues[n - 1 - i]);
	const auto norm = cblas_dnrm2(n, eigenvalues.data(), 1);
	cblas_daxpy(n, -1., lambdas.data(), 1, eigenvalues.data(), 1);
	const auto err = cblas_dnrm2(n, eigenvalues.data(), 1);
	return err / norm;
}

double bisect_sym_dpr1(size_t n, const vector<double>& AD, const vector<double>& Au, double Ar, char side) {
	// COMPUTES: the leftmost(for side = 'L') or the rightmost(for side = 'R') eigenvalue
	// of a SymDPR1 matrix A = diagm(A.D) + A.r*A.u*(A.u)' by bisection.
	// RETURNS: the eigenvalue
	if (n == 1)
		return AD[0] + Ar * Au[0];

	// u2 = Au * * 2
	vector<double> u2(n); // todo: optimize this - move allocation to parent
	vdMul(n, Au.data(), Au.data(), u2.data());

	//indD = np.argsort(AD)[::-1]
	// todo: optimize this - should be ordered
	vector<size_t> ind_d(n);
	iota(ind_d.begin(), ind_d.end(), 0);
	sort(ind_d.begin(), ind_d.end(), [&AD](auto i1, auto i2) {return AD[i1] > AD[i2]; });

	// Determine the starting interval for bisection, [left; right]
	double left, right;
	if (Ar > 0.) {
		if (side == 'L') {
			left  = AD[ind_d[n - 1]];
			right = AD[ind_d[n - 2]];
		}
		else {
			left  = AD[ind_d[0]];
			right = AD[ind_d[0]] + Ar * accumulate(u2.begin(), u2.end(), 0.);
		}
	}
	else { // rho<=0
		if (side == 'L') {
			left  = AD[ind_d[n - 1]] + Ar * accumulate(u2.begin(), u2.end(), 0.);
			right = AD[ind_d[n - 1]];
		}
		else {
			left  = AD[ind_d[1]]; 
			right = AD[ind_d[0]];
		}
	}

    // Bisection
	auto middle = (left + right) / 2.;
    while (right - left > 2.0*DBL_EPSILON*max(abs(left),abs(right))) {
		// Fmiddle = 1.0 + Ar * np.sum(u2 / (AD - middle))
		auto f_middle = 0.0;
		for (size_t i = 0; i < n; ++i)
			f_middle += u2[i] / (AD[i] - middle);
		f_middle = 1.0 + Ar * f_middle;
		if (Ar * f_middle < 0.0)
			left = middle;
		else
			right = middle;
		middle = (left + right) / 2.0;
    }
    return right;
}

double bisect_sym_arrow(size_t n, const vector<double>& AD, const vector<double>& Az, double Aa, char side) {
	// COMPUTES: the leftmost (for side='L') or the rightmost (for side='R') eigenvalue
    // of a SymArrow A = [diag (D) z; z'] by bisection.
    // RETURNS: the eigenvalue
	if (n == 0) return Aa;

	// z2 = Az ** 2
	vector<double> z2(n); // todo: optimize this - move allocation to parent
	vdMul(n, Az.data(), Az.data(), z2.data());

	// Determine the starting interval for bisection, [left; right]
    // left, right = side == 'L' ? {minimum([A.D-abs(A.z),A.a-sum(abs(A.z))]), minimum(A.D)} : 
    //   {maximum(A.D),maximum([A.D+abs.(A.z),A.a+sum(abs,A.z)])}
	vector<double> absAz(n);
	vdAbs(n, Az.data(), absAz.data());
	const auto abs_Az_sum = accumulate(absAz.begin(), absAz.end(), 0.);

	double left, right;
    if (side == 'L'){
		right = *min_element(AD.begin(), AD.end());
		vector<double> v1(n); // todo: do without allocation
		vdSub(n, AD.data(), absAz.data(), v1.data());
		left = *min_element(v1.begin(), v1.end());
		left = min(left, Aa - abs_Az_sum);
	}
    else{
		left = *max_element(AD.begin(), AD.end());
		vector<double> v1(n); // todo: do without allocation
		vdAdd(n, AD.data(), absAz.data(), v1.data());
        right = *max_element(v1.begin(), v1.end());
		right = max(right, Aa + abs_Az_sum);
    }
    // Bisection
	auto middle = (left + right) / 2.0;
	//auto count = 0;
    
    while ((right-left) > 2.0 * DBL_EPSILON * max(abs(left), abs(right))) {
		// Fmiddle = Aa - middle - np.sum(z2 / (AD - middle))
		auto f_middle = 0.0;
		for (size_t i = 0; i < n; ++i)
			f_middle += z2[i] / (AD[i] - middle);
		f_middle = Aa - middle - f_middle;

		if (f_middle > 0.0)
			left = middle;
		else
			right = middle;
		middle = (left + right) / 2.0;
    }
        
    // Return the eigenvalue
	return right;
}
    
void inv(const vector<double>& AD, const vector<double>& Au, double Ar, size_t i, double tolb, double tolz,
	vector<double>& D, vector<double>& z, double& b){
    // COMPUTES: inverse of a shifted SymDPR1 matrix A=diagm(A.D)+A.r*A.u*A.u',
    // inv(A-A.D[i]*I) which is a SymArrow.
    // Uses higher precision to compute top of the arrow element accurately, if
    // needed. 
    // tols=[tolb,tolz] are tolerances, usually [1e3, 10*n]
    // [0.0,0.0] forces DoubleDouble, [1e50,1e50] would never use it
    // RETURNS:  SymArrow(D,z,b,i), Kb, Kz, Qout
    // Kb - condition Kb, Kz - condition Kz, Qout = 1 / 0 - double was / was not used 
	const auto n = AD.size();
    const auto wz = 1.0/Au[i];
	const auto shift = AD[i];
	for(size_t k=0;k<i; ++k) {
		D[k] = 1.0 / (AD[k] - shift);
		z[k] = -Au[k] * D[k] * wz;
	}
	for (auto k = i+1; k < n; ++k) {
		D[k - 1] = 1.0 / (AD[k] - shift);
		z[k - 1] = -Au[k] * D[k - 1] * wz;
	}

    // compute the sum in a plain loop
	double P = 0., Q = 0.;
	for (size_t k = 0; k < i; ++k)
		if (D[k] > 0.) P += (Au[k] * Au[k])*D[k];
		else		   Q += (Au[k] * Au[k])*D[k];
	for (auto k = i; k < n - 1; ++k)
		if (D[k] > 0.) P += (Au[k + 1] * Au[k + 1])*D[k];
		else		   Q += (Au[k + 1] * Au[k + 1])*D[k];
	if (Ar > 0) P += 1.0 / Ar;
	else		Q += 1.0 / Ar;

	b = (P + Q)*wz*wz;
}

void inv_shift(const vector<double>& AD, const vector<double>& Au, double Ar, double shift, double tolr,
	vector<double>& D, vector<double>& u, double& rho){

    // COMPUTES: inverse of the shifted SymDPR1 A = diagm(A.D)+A.r*A.u*A.u', 
    // inv(A-shift*I) = D + rho*u*u', shift!=A.D[i], which is again a SymDPR1
    // uses DoubleDouble to compute A.r accurately, if needed. 
    // tolr is tolerance, usually 1e3,  0.0 forces Double, 1e50 would never use it
    // RETURNS: SymDPR1(D,u,rho), Krho, Qout
    // Krho - condition Krho, Qout = 1 / 0 - Double was / was not used

	const auto n = AD.size();
	for (size_t i = 0; i < n; ++i)
		D[i] = 1. / (AD[i] - shift);
	vdMul(n, Au.data(), D.data(), u.data());

    // compute gamma and Kgamma
    //--- compute the sum in a plain loop
	double P = 0., Q = 0.;
	for (size_t k = 0; k < n; ++k)
		if (D[k] > 0.) P += (Au[k] * Au[k])*D[k];
		else		   Q += (Au[k] * Au[k])*D[k];
	if (Ar > 0) P += 1.0 / Ar;
	else        Q += 1.0 / Ar;
	const auto P_plus_Q = P + Q;
	rho = P_plus_Q == 0. ? INFINITY : -1.0 / P_plus_Q;
}

template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

double eig(const vector<double>& AD, const vector<double>& Au, double Ar, size_t k, 
	double tolb = 1e3, double tolz = 10., double tolnu = 1e3, double tolrho = 1e3, double tolmu=1e3){
    // COMPUTES: k-th eigenpair of an ordered irreducible SymDPR1 
    // A = diagm(A.D)+A.r*A.u*A.u', A.r > 0
    // tols=[tolb,tolz,tolnu,tolrho,tolmu] = [1e3,10.0*n,1e3,1e3,1e3]
    // RETURNS: mu, v, Sind, Kb, Kz, Knu, Krho, Qout
    // mu - k-th eigenvalue in descending order
    // v - mu's normalized eigenvector
    // Kb, Kz, Knu, Krho - condition numbers
    // Qout = 1 / 0 - Double was / was not used 
	const auto n = AD.size();
	tolz *= n;

	double Kb = 0., Kz = 0., Krho = 0.; // Kz is former kappa_nu
    // Determine the shift sigma, the shift index i, and whether mu 
    // is on the left or the right side of the nearest pole
    // Exterior eigenvalues (k = 1 or k = n):

	//auto sigma = AD[k];
	//const auto i = k;
	//const auto side = 'R';
	vector<double> Au2(n); // todo: propagate to all childs
	vdMul(n, Au.data(), Au.data(), Au2.data());

	auto sigma = AD[0];
	auto i = 0;
	auto side = 'R';
	if (k > 0){
        // Interior eigenvalues
		vector<double> Dtemp(n);
		for (size_t j = 0; j < n; ++j)
			Dtemp[j] = AD[j] - AD[k]; // is it valid???
		const auto middle = Dtemp[k - 1] / 2.;
		auto Fmiddle = 0.;
		for (size_t j = 0; j < n; ++j)
			Fmiddle += Au2[j] / (Dtemp[j] - middle);
		Fmiddle = 1.0 + Ar * Fmiddle;
		if (Fmiddle > 0.) {
			sigma = AD[k];
			i = k;
			side = 'R';
		}
		else {
			sigma = AD[k - 1];
			i = k - 1;
			side = 'L';
		}
	}
    //if verbose:
    //    print('side = ', side, ', sigma = ', i)

    // Compute the inverse of the shifted matrix, A_i^(-1), Kb and Kz
	vector<double> AinvD(n - 1), Ainvz(n - 1);
	double Ainva, mu;
	inv(AD, Au, Ar, i, tolb, tolz, AinvD, Ainvz, Ainva);

    // Compute the eigenvalue of the inverse shifted matrix
	auto nu = bisect_sym_arrow(n - 1, AinvD, Ainvz, Ainva, side);

	if (abs(nu) == INFINITY)
		// this is nonstandard
		// Deflation in dpr1eig (nu=Inf)
		return mu = sigma;
 
        // standard case, full computation
        // nu1 is the F- or 1-norm of the inverse of the shifted matrix
        // nu10=maximum([sum(abs,Ainv.z)+abs(Ainv.a), maximum(abs.(Ainv.D)+abs.(Ainv.z))])
	auto nu1 = 0.;
	for (size_t k = 0; k < n - 1; ++k)
		nu1 = max(nu1, abs(AinvD[k]) + abs(Ainvz[k]));
	nu1 = max(accumulate(Ainvz.begin(), Ainvz.end(), 0., [](double a, double x) { return a + abs(x); }) + abs(Ainva), nu1);

	AinvD.resize(n);
	Ainvz.resize(n);

	auto Knu = nu1 / abs(nu);
    while (Knu > tolnu){
        // Remedies according to Remark 3 - we shift between original
        // eigenvalues and compute DPR1 matrix
        // 1/nu1+sigma, 1/nu+sigma
        // print("Remedy 3 ")
		nu = side == 'R' ? abs(nu) : -abs(nu);
		nu1 = -sgn(nu) * nu1;
		const auto sigma1 = (nu1 + nu) / (2.0*nu*nu1) + sigma;
		double Ainvr;
        // Compute the inverse of the shifted arrowhead (DPR1)
		inv_shift(AD, Au, Ar, sigma1, tolrho, AinvD, Ainvz, Ainvr);
        // Compute the eigenvalue by bisect for DPR1
        // Note: instead of bisection could use dlaed4 (need a wrapper) but
        // it is not faster. There norm(u)==1
		nu = bisect_sym_dpr1(n, AinvD, Ainvz, Ainvr, side);
		const auto max_abs_AinvD = accumulate(AinvD.begin(), AinvD.end(), 0., [](double a, double x) { return max(a, abs(x)); });
		nu1 = max_abs_AinvD + abs(Ainvr)*cblas_ddot(n, Ainvz.data(), 1, Ainvz.data(), 1);
			
		Knu = nu1 / abs(nu);
		sigma = sigma1;
    }
	mu = 1. / nu + sigma;
	if (mu == 0.0) // todo: возможно стоит попробовать пересчитать из обратной матрицы
		return mu;

    // Remedy according to Remark 1 - we recompute the the eigenvalue
    // near zero from the inverse of the original matrix (a DPR1 matrix).  
	if ((abs(AD[i]) + abs(1.0 / nu)) / abs(mu) > tolmu) {
		if  (k == 0 && AD[0] < 0.0 || 
			side == 'L' && sgn(AD[i]) + sgn(AD[i + 1]) == 0 ||
			i>0 && side == 'R' && sgn(AD[i]) + sgn(AD[i - 1]) == 0) {
			// print("Remedy 1 ")
			// Compute the inverse of the original arrowhead(DPR1)
			inv_shift(AD, Au, Ar, 0., tolrho, AinvD, Ainvz, Ainva);
			if (Ainva == INFINITY)
				mu = 0.0;
			else {
				// в оригинальной работе используется отношение Релея, но я не хочу считать собственный вектор
				// todo: найти способ искать максимальное по модулю собственное значение
				const auto nu_1 = bisect_sym_dpr1(n, AinvD, Ainvz, Ainva, 'R');
				const auto nu_2 = bisect_sym_dpr1(n, AinvD, Ainvz, Ainva, 'L');
				mu = 1.0 / (abs(nu_1) > abs(nu_2) ? nu_1 : nu_2);
			}
		}
	}
	// check side
#ifdef _DEBUG
	if (k > 0 && ((abs(mu - AD[k]) < abs(mu - AD[k-1])) != (side == 'R')))
		cout << "invalid side !!!!" << endl;
#endif
    return mu;
}
    
    
bool solve_secular_equation_stor(
	const std::vector<double>& lambdas, 
	const std::vector<double>& a, 
	double rho,
	std::vector<double>& mu) {
	const auto n = static_cast<int>(lambdas.size());
	mu.resize(n);


	const auto a_len_sq = inner_product(a.begin(), a.end(), a.begin(), 0.);
	const auto a_len = sqrt(a_len_sq);
	vector<double> z(n, 0.);
	cblas_daxpy(a.size(), 1. / a_len, a.data(), 1, z.data(), 1);

	const auto inverse = rho < 0.;
	const auto _rho = abs(rho) * a_len_sq;

	auto d = lambdas;
	if (inverse) {
		cblas_dscal(n, -1., d.data(), 1);
		for (auto i = 0; i < n / 2; ++i) swap(d[i], d[n - 1 - i]);
		for (auto i = 0; i < n / 2; ++i) swap(z[i], z[n - 1 - i]);
	}
#ifdef _DEBUG
	for (auto i = 1; i < n; ++i)
		if (d[i - 1] <= d[i])
			cout << "secular arguments wrong order!" << endl;
	if (_rho <= 0.0)
		cout << "secular wrong _rho!" << endl;
#endif
	for (auto i = 0; i < n; ++i)
		mu[i] = eig(d, z, _rho, i);
	if (inverse) {
		cblas_dscal(n, -1., mu.data(), 1);
		for (auto i = 0; i < n / 2; ++i) swap(mu[i], mu[n - 1 - i]);
	}

#ifdef _DEBUG
	for (auto i = 1; i < n; ++i)
		if (mu[i - 1] < mu[i])
			cout << "secular solution wrong order!" << endl;
	const auto rerror = rerror_secular_equation_solution(mu, lambdas, a, rho);
	if (rerror > DBL_EPSILON)
		cout << "secular solution error: " << rerror << endl;
#endif
	return true;
}
