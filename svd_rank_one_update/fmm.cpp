#define _USE_MATH_DEFINES
#include <cmath>

#include <vector>
#include <tuple>
#include <algorithm>

#include "mkl.h"

#include "fmm.h"
#include <cassert>

#define ALIGNMENT 256

using namespace std;

double u(const fmm_context& ctx, uint64_t j, double t)
{
	auto res = 1.;
	for (uint64_t i = 0; i < ctx.p; ++i)
		if (i != j)
			res *= (t - ctx.t[i]) / (ctx.t[j] - ctx.t[i]);
	return res;
}

struct interval
{
	uint64_t level = 0, index = 0;
	uint64_t x_start = 0, x_end = 0;
	uint64_t y_start = 0, y_end = 0;
	interval *lchild = nullptr, *rchild = nullptr, *nl = nullptr, *nr = nullptr;
	double mid = 0., r = 0.;
	double *ffe = nullptr, *le = nullptr;
	
	~interval()
	{
		if (ffe != nullptr) mkl_free(ffe);
		if ( le != nullptr) mkl_free(le);
	}

	bool childless() const { return lchild == nullptr; }
	double* get_ffe(const fmm_context& ctx, const fmm_task& task)
	{
		if (ffe != nullptr)
			return ffe;

		const auto size = sizeof(double)*ctx.p*task.dimv;
		ffe = static_cast<double*>(mkl_malloc(size, ALIGNMENT));
		if(!childless())
		{
			//self.ffe = ctx.Ml.dot(self.lchild.get_ffe(ctx)) \
			//	+ ctx.Mr.dot(self.rchild.get_ffe(ctx))
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				ctx.p, task.dimv, ctx.p, 1.,
				ctx.Ml, ctx.p, lchild->get_ffe(ctx, task), task.dimv,
				0., ffe, task.dimv);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				ctx.p, task.dimv, ctx.p, 1.,
				ctx.Mr, ctx.p, rchild->get_ffe(ctx, task), task.dimv,
				1., ffe, task.dimv);
			return ffe;
		}
		memset(ffe, 0, size);
		for (auto k = x_start; k < x_end; ++k)
			for (uint64_t j = 0; j < ctx.p; ++j)
				cblas_daxpy(task.dimv, 1. / (3 * r / ctx.t[j] - (task.x[k] - mid)), 
					task.v[k], 1, 
					ffe     + j*task.dimv, 1);
				//ffe[j] += task.v[k] * (1. / (3 * r / ctx.t[j] - (task.x[k] - mid)));
		return ffe;
	}

	void link()
	{
		if(childless()) return;
		lchild->nr = rchild;
		rchild->nl = lchild;
		lchild->nl = nl == nullptr || nl->childless() ? nl : nl->rchild;
		rchild->nr = nr == nullptr || nr->childless() ? nr : nr->lchild;
	}

	void add_le_from_points(const fmm_context& ctx, const fmm_task& task, const interval* another) const {
		for (auto x_i = another->x_start; x_i < another->x_end; ++x_i)
			for (uint64_t j = 0; j < ctx.p; ++j)
				// self.le[j] +=  ctx.v[x_i]/(self.r*ctx.t[j]-(ctx.x[x_i]-self.mid))
				cblas_daxpy(task.dimv, 1. / (r*ctx.t[j] - (task.x[x_i] - mid)),
					task.v[x_i], 1,
					le + j * task.dimv, 1);
	}
	void eval_ffe(const fmm_context& ctx, const fmm_task& task, const interval* another) const {
		for (auto y_i = y_start; y_i < y_end; ++y_i)
			for (uint64_t j = 0; j < ctx.p; ++j)
				// ctx.res1[y_i] += another.ffe[j]*ctx.u(j,3*another.r/(ctx.y[y_i]-another.mid))
				cblas_daxpy(task.dimv, u(ctx, j, 3 * another->r / (task.y[y_i] - another->mid)),
					another->ffe + j * task.dimv, 1,
					task.res[y_i], 1);
	}

	void calc_le(const fmm_context& ctx, const fmm_task& task)
	{
		if(childless()){
			if (le == nullptr)
				return;
			for (auto y_k = y_start; y_k < y_end; ++y_k)
				for (uint64_t j = 0; j < ctx.p; ++j)
					// task.res[y_k] += le[j] * u(ctx, j, (task.y[y_k] - mid) / r);
					cblas_daxpy(task.dimv, u(ctx, j, (task.y[y_k] - mid) / r),
						le + j * task.dimv, 1,
						task.res[y_k], 1);
			return;
		}

		const auto size = sizeof(double)*ctx.p*task.dimv;
		lchild->le = static_cast<double*>(mkl_malloc(size, ALIGNMENT));
		rchild->le = static_cast<double*>(mkl_malloc(size, ALIGNMENT));

		if(le != nullptr){
            // self.lchild.le = ctx.Sl.dot(self.le)
            // self.rchild.le = ctx.Sr.dot(self.le)
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				ctx.p, task.dimv, ctx.p, 1.,
				ctx.Sl, ctx.p, le, task.dimv,
				0., lchild->le, task.dimv);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				ctx.p, task.dimv, ctx.p, 1.,
				ctx.Sr, ctx.p, le, task.dimv,
				0., rchild->le, task.dimv);
		}
		else{
			memset(lchild->le, 0, size);
			memset(rchild->le, 0, size);
		}

		if (nl != nullptr){
			if (nl->childless()) {
				rchild->add_le_from_points(ctx, task, nl);
				nl->eval_ffe(ctx, task, rchild);
			}
			else {
				// self.rchild.le += ctx.T1.dot(self.nl.lchild.ffe)
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					ctx.p, task.dimv, ctx.p, 1.,
					ctx.T1, ctx.p, nl->lchild->get_ffe(ctx, task), task.dimv,
					1., rchild->le, task.dimv);
			}
		}
		if (nr != nullptr) {
			if (nr->childless()) {
				lchild->add_le_from_points(ctx, task, nr);
				nr->eval_ffe(ctx, task, lchild);
			}
			else {
				// self.lchild.le += ctx.T4.dot(self.nr.rchild.ffe)
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					ctx.p, task.dimv, ctx.p, 1.,
					ctx.T4, ctx.p, nr->rchild->get_ffe(ctx, task), task.dimv,
					1., lchild->le, task.dimv);
			}
		}
		if (nl != nullptr && !nl->childless()) {
            // self.lchild.le += ctx.T2.dot(self.nl.lchild.ffe)
            // self.rchild.le += ctx.T2.dot(self.nl.rchild.ffe)
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				ctx.p, task.dimv, ctx.p, 1.,
				ctx.T2, ctx.p, nl->lchild->get_ffe(ctx, task), task.dimv,
				1., lchild->le, task.dimv);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				ctx.p, task.dimv, ctx.p, 1.,
				ctx.T2, ctx.p, nl->rchild->get_ffe(ctx, task), task.dimv,
				1., rchild->le, task.dimv);
		}
		if (nr != nullptr && !nr->childless()) {
			// self.lchild.le += ctx.T3.dot(self.nr.lchild.ffe)
			// self.rchild.le += ctx.T3.dot(self.nr.rchild.ffe)
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				ctx.p, task.dimv, ctx.p, 1.,
				ctx.T3, ctx.p, nr->lchild->get_ffe(ctx, task), task.dimv,
				1., lchild->le, task.dimv);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				ctx.p, task.dimv, ctx.p, 1.,
				ctx.T3, ctx.p, nr->rchild->get_ffe(ctx, task), task.dimv,
				1., rchild->le, task.dimv);
		}
	}

	void neighbour_interactions(const fmm_task& task, const interval* another) const {
		for (auto y_i = y_start; y_i < y_end; ++y_i)
			for (auto x_i = another->x_start; x_i < another->x_end; ++x_i)
				// task.res[y_i] += task.v[x_i] / (task.y[y_i] - task.x[x_i]);
				cblas_daxpy(task.dimv, 1./ (task.y[y_i] - task.x[x_i]),
					task.v  [x_i], 1,
					task.res[y_i], 1);
	}

	void eval_neighbour_interactions(const fmm_context& ctx, const fmm_task& task) {
		auto left = nl, right = nr;
		if (left  != nullptr)
			while (!left ->childless()) left  = left ->rchild;
		if (right != nullptr)
			while (!right->childless()) right = right->lchild;

		if (!task.precise_neighbours_interactions) {
			if (left  != nullptr) neighbour_interactions(task, left );
			if (right != nullptr) neighbour_interactions(task, right);
			neighbour_interactions(task, this);
			return;
		}
		const auto li_start = static_cast<int64_t>(left  == nullptr ? x_start : left ->x_start);
        const auto ri_start = static_cast<int64_t>(right == nullptr ? x_end   : right->x_end  ) - 1;
		if (li_start > ri_start) return;

		const auto add_vec = [&task](const int i1, const int i2, const double div) {
			cblas_daxpy(task.dimv, 1. / div,
				task.v  [i2], 1,
				task.res[i1], 1);
		};

		for (auto y_i = y_start; y_i < y_end; ++y_i){
			auto li = li_start, ri = ri_start;
			auto ldiff = task.x[li] - task.y[y_i], rdiff = task.y[y_i] - task.x[ri];
            while (ldiff > 0 || rdiff > 0){
				if (li == ri) {
					     if (ldiff < 0) add_vec(y_i, ri,  rdiff);// task.res[y_i] -= task.v[ri] / rdiff;
					else if (rdiff < 0) add_vec(y_i, li, -ldiff);// task.res[y_i] += task.v[li] / ldiff;
					break;
				}
				if (ldiff < 0) {
					add_vec(y_i, ri, rdiff);// task.res[y_i] -= task.v[ri] / rdiff;
					rdiff = task.y[y_i] - task.x[--ri];
				}
				else if (rdiff < 0) {
					add_vec(y_i, li, -ldiff); // task.res[y_i] += task.v[li] / ldiff;
					ldiff = task.x[++li] - task.y[y_i];
				}
				else if (ldiff > rdiff) {
					add_vec(y_i, li, -ldiff); // task.res[y_i] += task.v[li] / ldiff;
					ldiff = task.x[++li] - task.y[y_i];
				}
				else {
					add_vec(y_i, ri, rdiff);// task.res[y_i] -= task.v[ri] / rdiff;
					rdiff = task.y[y_i] - task.x[--ri];
				}
			}
		}
	}
};

void init_fmm_context(fmm_context* ctx, uint64_t p)
{
	ctx->p = p;
	const auto size = sizeof(double)*p*p;

	ctx->t  = static_cast<double*>(mkl_malloc(size, ALIGNMENT));
	ctx->Ml = static_cast<double*>(mkl_malloc(size, ALIGNMENT));
	ctx->Mr = static_cast<double*>(mkl_malloc(size, ALIGNMENT));
	ctx->Sl = static_cast<double*>(mkl_malloc(size, ALIGNMENT));
	ctx->Sr = static_cast<double*>(mkl_malloc(size, ALIGNMENT));
	ctx->T1 = static_cast<double*>(mkl_malloc(size, ALIGNMENT));
	ctx->T2 = static_cast<double*>(mkl_malloc(size, ALIGNMENT));
	ctx->T3 = static_cast<double*>(mkl_malloc(size, ALIGNMENT));
	ctx->T4 = static_cast<double*>(mkl_malloc(size, ALIGNMENT));

	for (uint64_t i = 0; i < ctx->p; ++i)
		ctx->t[i] = cos((2 * i + 1) * M_PI_2 / ctx->p);

	for (uint64_t i = 0; i < ctx->p; ++i)
		for (uint64_t j = 0; j < ctx->p; ++j) {
			ctx->Ml[i*ctx->p + j] = u(*ctx, j, 3 * ctx->t[i] / (6 - ctx->t[i]));
			ctx->Mr[i*ctx->p + j] = u(*ctx, j, 3 * ctx->t[i] / (6 + ctx->t[i]));
			ctx->T1[i*ctx->p + j] = u(*ctx, j, 3 / (ctx->t[i] - 6));
			ctx->T2[i*ctx->p + j] = u(*ctx, j, 3 / (ctx->t[i] - 4));
			ctx->T3[i*ctx->p + j] = u(*ctx, j, 3 / (ctx->t[i] + 4));
			ctx->T4[i*ctx->p + j] = u(*ctx, j, 3 / (ctx->t[i] + 6));
			ctx->Sl[i*ctx->p + j] = u(*ctx, j, (ctx->t[i] + 1) / 2);
			ctx->Sr[i*ctx->p + j] = u(*ctx, j, (ctx->t[i] - 1) / 2);
		}
}

void free_fmm_context(fmm_context* ctx)
{
	mkl_free(ctx->t);
	mkl_free(ctx->Ml);
	mkl_free(ctx->Mr);
	mkl_free(ctx->Sl);
	mkl_free(ctx->Sr);
	mkl_free(ctx->T1);
	mkl_free(ctx->T2);
	mkl_free(ctx->T3);
	mkl_free(ctx->T4);

	ctx->t  = nullptr;
	ctx->Ml = nullptr;
	ctx->Mr = nullptr;
	ctx->Sl = nullptr;
	ctx->Sr = nullptr;
	ctx->T1 = nullptr;
	ctx->T2 = nullptr;
	ctx->T3 = nullptr;
	ctx->T4 = nullptr;
}



void fmm(const fmm_task &task, const fmm_context& ctx)
{
	const auto min_xy = min(task.x[task.nx - 1], task.y[task.ny - 1]);
	const auto max_xy = max(task.x[0], task.y[0]);
	const auto delta_xy = max_xy - min_xy;
	const auto max_points_inside_interval = task.s == 0 ? 2 * ctx.p : task.s;

	vector<interval*> intervals{ new interval{ 0, 0, 0, task.nx, 0, task.ny } };
	for(uint64_t queue_pos = 0; queue_pos < intervals.size(); ++queue_pos)
	{
		const auto i = intervals[queue_pos];

		const auto n = (i->x_end - i->x_start) + (i->y_end - i->y_start);
		i->r = delta_xy / (1 << (i->level + 1));
		i->mid = max_xy - (2 * i->index + 1)*i->r;

		if (n <= max_points_inside_interval)
			continue;

		uint64_t x_mid = i->x_start, y_mid = i->y_start;
		for (; x_mid < i->x_end && task.x[x_mid] > i->mid; ++x_mid) {}
		for (; y_mid < i->y_end && task.y[y_mid] > i->mid; ++y_mid) {}

		intervals.push_back(i->lchild = new interval{ i->level + 1, i->index * 2    , i->x_start,    x_mid, i->y_start,    y_mid });
		intervals.push_back(i->rchild = new interval{ i->level + 1, i->index * 2 + 1,    x_mid  , i->x_end,    y_mid  , i->y_end });
	}
	
	for (auto* i : intervals) i->link();
	for (auto* i : intervals) i->calc_le(ctx, task);

	const auto cauchy_matrix = task.c == nullptr 
							 ? static_cast<double*>(mkl_malloc(task.nx*task.ny*sizeof(double), ALIGNMENT))
							 : task.c;
	auto cm = cauchy_matrix;
	for (uint64_t y_i = 0; y_i < task.ny; ++y_i)
		for (uint64_t x_i = 0; x_i < task.nx; ++x_i) {
			assert(task.y[y_i] != task.x[x_i]);
			*cm++ = 1. / (task.y[y_i] - task.x[x_i]);
		}

	auto leaf = intervals[0];
	while (!leaf->childless()) 
		leaf = leaf->lchild;
	while(true) {
		leaf->eval_neighbour_interactions(ctx, task);
		for(leaf = leaf->nr; leaf!=nullptr && !leaf->childless(); leaf = leaf->lchild){}
		if (leaf == nullptr)
			break;
	}

	for (auto* i : intervals)
		delete i;
	if (task.c == nullptr)
		mkl_free(cauchy_matrix);
}

int fmm(const fmm_task *task, const fmm_context* ctx) {
	if (task == nullptr)
		return -1;
	if (ctx == nullptr)
		return -2;
	fmm(*task, *ctx);
	return 0;
}

int fmm(const fmm_task *task, const uint64_t p) {
	if (task == nullptr)
		return -1;
	fmm_context ctx{};
	init_fmm_context(&ctx, p);
	fmm(*task, ctx);
	free_fmm_context(&ctx);
	return 0;
}