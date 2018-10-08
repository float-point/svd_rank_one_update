#pragma once


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
);