#pragma once
#include <stdint.h>

struct fmm_task
{
	const double *x, *y;
	const double* const* v;
	double* const* res;
	uint64_t nx, ny, dimv, s;
	bool precise_neighbours_interactions;
	double* c;
};

struct fmm_context
{
	uint64_t p;
	double *t, *Ml, *Mr, *Sl, *Sr, *T1, *T2, *T3, *T4;
};

void init_fmm_context(struct fmm_context* ctx, uint64_t p);
void free_fmm_context(struct fmm_context* ctx);

int fmm(const fmm_task *task, const fmm_context* ctx);
int fmm(const fmm_task *task, uint64_t p);