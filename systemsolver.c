//William Sun
//Dec 3, 2014
//Solves for the coefficients in a system of equations
//Using stochastic gradient descent

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int ind(int i, int j, int n){
	return i*n + j;
}

/*print will print a nx1 array*/
void printx(double *r, int n){
	int i;
	for (i=0; i<n; i++)
		printf("%lf ", r[i]);
	printf("\n");
	printf("\n");
}

void dumb_solve (double *a, double *y, int n, double eps, int numit, double *x, int *niter, double *discreps);

void multiply (double *left, double *right, double *result, int m, int n);

void print (double *matrix, int m, int n);

void transpose (double *matrix, int n);

/* Find the norm of the vector, namely sum all elements and then
 * take the square root
 */
double norm (double *vector, int n);

void subtract (double *left, double *right, double *result, int n);

/* Calculate the gradient at the given point x.
 * The gradient is given by del [f (x)] = 2At (Ax - b)
 */
double *gradient (double *a, double *ax_b, int n);

/* Objective function that we are trying to minimize. The function is:
 * f (x) = || Ax - b || ^ 2
 * This function calculates the value of f (x) at the specified x
 */
double fOfx (double *ax_b, int n);

/* Using the closed formula derived (see attached report) calculate gamma.
 * Gamma is the optimal 'step size' at each iteration, given where we are
 * in the function */
double stepSize (double *a, double *ax_b, int n); 

void dumb_solve (double *a, double *y, int n, double eps, int numit, double *x,
					int *niter, double *discreps) {
	int i;
	double *ax				= (double *) malloc (sizeof (double) * n);
	double *ax_b			= (double *) malloc (sizeof (double) * n);
	double *x_n				= (double *) malloc (sizeof (double) * n);
	double *grad			= NULL;
	double step_size		= 0.0;
	double approximation	= 0.0;

	/* Initial approximation of solution is 0 */
	for (i = 0; i < n; i++) {x_n [i] = 0;}

	for (*niter = 0; (*niter < numit); (*niter)++) {
		/* Calculate (Ax - b) the value of which will be re-used for many
		 * of the functions below */
		multiply (a, x_n, ax, n, 1);
		subtract (ax, y, ax_b, n);
		/* We've reached the required accuracy, break */
		approximation = fOfx (ax_b, n);
		if (approximation < eps)
			break;
		discreps [*niter] = approximation;
		/* Calculate the new approximation */
		grad = gradient (a, ax_b, n);
		step_size = stepSize (a, ax_b, n);
		for (i = 0; i < n; i++) {grad [i] = step_size * grad [i];}
		subtract (x_n, grad, x_n, n);
		free (grad);
	}

	/* Copy over answer to solution vector x */
	for (i = 0; i < n; i++) {x [i] = x_n [i];}

	free (x_n); free (ax); free (ax_b);
	return;
}

double stepSize (double *a, double *ax_b, int n) {
	int i;
	double *at			= (double *) malloc (sizeof (double) * (n * n));
	double *aat			= (double *) malloc (sizeof (double) * (n * n));
	double *nom			= (double *) malloc (sizeof (double) * n);
	double *denom		= (double *) malloc (sizeof (double) * n);
	double nominator	= 0.0;
	double denominator	= 0.0;

	/* Calculate A transpose */
	for (i = 0; i < (n * n); i++) {at [i] = a [i];}
	transpose (at, n);

	/* Calculate At * (Ax - b) */
	multiply (at, ax_b, nom, n, 1);

	/* Calculate norm squared of At * (Ax - b), i.e. nominator */
	for (i = 0; i < n; i++) {nominator += nom [i] * nom [i];}

	/* Calculate A*At */
	multiply (a, at, aat, n, n);

	/* Calculate A*At * (Ax - b) */
	multiply (aat, ax_b, denom, n, 1);

	/* Calculate the norm squared of A*At * (Ax - b), i.e. denominator */
	for (i = 0; i < n; i++) {denominator += denom [i] * denom [i];}

	free (at); free (aat); free (nom); free (denom);
	return (nominator / (2.0 * denominator));
}

double fOfx (double *ax_b, int n) {
	int i;
	double out = 0.0;
	/* Dot (Ax - b) with itself which is equal to || Ax - b || ^ 2 */
	for (i = 0; i < n; i++) {out += ax_b [i] * ax_b [i];}
	return out;
}

void multiply (double *left, double *right, double *result, int m, int n) {
	int leftRow, leftCol;
	int rightRow, rightCol;
	double tempResult = 0;

	for (leftRow = 0; leftRow < m; leftRow++) {
		for (rightCol = 0; rightCol < n; rightCol++) {
			for (leftCol = rightRow = 0; leftCol < m; leftCol++, rightRow++)
				tempResult+=left[leftRow*m+leftCol]*right[rightRow*n+rightCol];
			result [leftRow * n + rightCol] = tempResult;
			tempResult = 0;
		}
	}
	return;
}

double *gradient (double *a, double *ax_b, int n) {
	int i;
	double *at			= (double *) malloc (sizeof (double) * (n * n));
	double *gradient	= (double *) malloc (sizeof (double) * n);

	/* Calculate 2 * At */
	for (i = 0; i < (n * n); i++) {at [i] = 2 * a [i];}
	transpose (at, n);
	/* Calculate 2At(Ax - b) */
	multiply (at, ax_b, gradient, n, 1);

	free (at);
	return gradient;
}

void print (double *matrix, int m, int n) {
	int i, j;
	printf ("\n");
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++)
			printf ("%.9f ", matrix [i * n + j]);
		printf ("\n");
	}
	printf ("\n");
	return;
}

void subtract (double *left, double *right, double *result, int n) {
	int i;
	for (i = 0; i < n; i++)
		result [i] = left [i] - right [i];
	return;
}

double norm (double *vector, int n) {
	double result = 0.0;
	int i = 0;
	for (i = 0; i < n; i++)
		result += (vector[i] * vector [i]);
	return sqrt (result);
}

void transpose (double *matrix, int n) {
	double temp;
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			if (j == i) {continue;} /* skip diagonal elements */
			temp = matrix [i * n + j];
			matrix [i * n + j] = matrix [j * n + i];
			matrix [j * n + i] = temp;
		}
	}
	return;
}

int main()
{
	/*double eps = 0.0;
	int numit = 10;
	int n=2;
	double a[4]={0.0};
	double x[2]={0.0};
	double y[2]={0.0};
	//double discreps[10]={0.0};

	a[ind(0,0,n)] = 10.0;
	a[ind(0,1,n)] = 3.0;
	a[ind(1,0,n)] = 2.0;
	a[ind(1,1,n)] = 1.0;
	x[0] = 1.0; 
	x[1] = 2.0;
	y[0] = 1.0;
	y[1] = 1.0;
	dumb_solve(a,y,2,x);*/
	double eps = 0.001;
	int numit = 100000;
	int niter = 100;
	double discreps[100]={0.0};

	int n=3;
	double *temp = (double*) malloc(n*sizeof(double));
	double a[9]={0.0};
	double x[3]={0.0};
	double y[3]={0.0};

	a[ind(0,0,n)] = 10.0;
	a[ind(0,1,n)] = 1.0;
	a[ind(0,2,n)] = 0.0;
	a[ind(1,0,n)] = 2.0;
	a[ind(1,1,n)] = 5.0;
	a[ind(1,2,n)] = 4.0;
	a[ind(2,0,n)] = 1.0;
	a[ind(2,1,n)] = 3.0;
	a[ind(2,2,n)] = 1.0;
	x[0] = 0.0; 
	x[1] = 0.0;
	x[2] = 0.0;
	y[0] = 2.0;
	y[1] = 1.0;
	y[2] = 0.0; 
	//dumb_solve(a,y,n,eps,numit,x,&niter,discreps);
	print(x,n,1);
	double temp2 = stepSize(a,a,n);
	temp = gradient(a,a,n);
	print(temp,n,n);
}