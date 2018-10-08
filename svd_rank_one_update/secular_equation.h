#pragma once
#include <vector>

bool 
solve_secular_equation(
	const std::vector<double>& lambdas, 
	const std::vector<double>& a, 
	double rho, 
	std::vector<double>& mu);

double
rerror_secular_equation_solution(
	const std::vector<double>& lambdas,
	const std::vector<double>& d,
	const std::vector<double>& z,
	const double rho);

bool
solve_secular_equation_stor(
	const std::vector<double>& lambdas,
	const std::vector<double>& a,
	double rho,
	std::vector<double>& mu);